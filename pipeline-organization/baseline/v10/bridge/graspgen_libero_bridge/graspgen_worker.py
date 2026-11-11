"""GraspGen worker, isolated from LIBERO's MuJoCo / robosuite environment.

All point clouds and returned poses use the *same world coordinate frame* and
meters. The worker subtracts and restores the observed point-cloud mean exactly
once. It does not require an object pose or a bundled object manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from .pointcloud import sample_points


class GraspGenWorker:
    def __init__(self, config, grasp_repo):
        self.config_path = Path(config).resolve()
        self.grasp_repo = Path(grasp_repo).resolve()
        self.sampler = None
        self.cfg = None
        self.model_identity = None

    def load(self):
        if self.sampler is not None:
            return
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("USE_TF", "0")
        source = self.grasp_repo / "third_party" / "GraspGen"
        if not (source / "grasp_gen").is_dir():
            raise FileNotFoundError("GraspGen source not found: " + str(source))
        if not self.config_path.is_file():
            raise FileNotFoundError(str(self.config_path))
        sys.path.insert(0, str(source))
        import torch
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

        if not torch.cuda.is_available():
            raise RuntimeError("GraspGen requires a CUDA device in its model worker")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = load_grasp_cfg(str(self.config_path))
        if cfg.data.gripper_name != "franka_panda":
            raise ValueError("The bridge currently supports the official Franka Panda model")
        if int(cfg.data.num_points) != 2048:
            raise ValueError("Expected the official 2048-point Panda checkpoint")
        self.sampler = GraspGenSampler(cfg)
        self.cfg = cfg
        self.model_identity = {
            "config": str(self.config_path),
            "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            "generator_checkpoint": str(cfg.eval.checkpoint),
            "discriminator_checkpoint": str(cfg.discriminator.checkpoint),
            "grasp_repo": str(self.grasp_repo),
            "gripper": str(cfg.data.gripper_name),
            "input_points": int(cfg.data.num_points),
        }

    def handle(self, request):
        command = request.get("command", "infer")
        if command == "ping":
            return {"ok": True, "success": True, "loaded": self.sampler is not None,
                    "model": self.model_identity}
        if command != "infer":
            raise ValueError("Unsupported GraspGen command: " + str(command))
        self.load()
        import torch
        from grasp_gen.dataset.dataset import collate

        seed = int(request.get("seed", 0))
        count = int(request.get("num_grasps", 200))
        if not 1 <= count <= 2048:
            raise ValueError("num_grasps must be in [1, 2048]")
        points_path = Path(request["points_path"]).resolve()
        output_path = Path(request["output_path"]).resolve()
        with np.load(str(points_path), allow_pickle=False) as data:
            source_points = np.asarray(data["points_world"], dtype=np.float32)
        if source_points.ndim != 2 or source_points.shape[1] != 3:
            raise ValueError("points_world must have shape (N, 3)")
        if len(source_points) < 16 or not np.isfinite(source_points).all():
            raise ValueError("Need at least 16 finite world-coordinate target points")
        points = sample_points(source_points, 2048, seed)
        if float(np.linalg.norm(np.ptp(points, axis=0))) < 1e-5:
            raise ValueError("Target point cloud has negligible spatial extent")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        pc = torch.as_tensor(points, device="cuda", dtype=torch.float32)
        center = pc.mean(dim=0)
        centered = pc - center[None]
        datum = {
            "task": "pick",
            "inputs": torch.cat([centered, torch.zeros_like(centered)], dim=-1),
            "points": centered,
        }
        self.sampler.model.grasp_generator.num_grasps_per_object = count
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction, _, _ = self.sampler.model.infer(collate([datum]))
        torch.cuda.synchronize()
        generation_s = time.perf_counter() - started
        raw = prediction["grasps_pred"][0].detach().cpu().numpy().astype(np.float32)
        scores = prediction["grasp_confidence"][0, :, 0].detach().cpu().numpy().astype(np.float32)
        if raw.shape != (count, 4, 4) or scores.shape != (count,):
            raise ValueError("Unexpected GraspGen output shapes: %r / %r" %
                             (raw.shape, scores.shape))
        center_np = center.detach().cpu().numpy().astype(np.float32)
        poses = raw.copy()
        poses[:, :3, 3] += center_np
        # The official model does not consistently populate the homogeneous row.
        poses[:, 3, :] = [0, 0, 0, 1]
        if not np.isfinite(poses).all() or not np.isfinite(scores).all():
            raise ValueError("Nonfinite GraspGen prediction")
        rotations = poses[:, :3, :3].astype(np.float64)
        orthogonality = np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)))
        if orthogonality > 0.01 or np.max(np.abs(np.linalg.det(rotations) - 1)) > 0.01:
            raise ValueError("Predicted orientations are not valid rotations")
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(count, dtype=np.int32)
        ranks[order] = np.arange(1, count + 1, dtype=np.int32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Saving all proposals preserves model output provenance; the planner
        # makes a separate, explicit choice among sorted candidates.
        with output_path.open("wb") as handle:
            np.savez_compressed(
                handle, poses_world_grasp=poses, scores=scores,
                points_world=points, center_world=center_np,
                poses_centered_grasp_raw=raw,
                generation_index=np.arange(count, dtype=np.int32),
                original_rank=ranks,
            )
        result = {
            "ok": True, "success": True, "output_path": str(output_path), "num_grasps": count,
            "seed": seed, "input_points": int(len(source_points)),
            "sampled_points": 2048, "generation_s": generation_s,
            "elapsed_s": generation_s, "generation_calls": 1,
            "coordinate_frame": "world", "units": "meters",
            "preprocess": "point-cloud mean subtracted and restored exactly once",
            "model": self.model_identity,
        }
        metadata_path = output_path.with_suffix(output_path.suffix + ".json")
        metadata_path.write_text(json.dumps(result, indent=2) + "\n")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-repo", required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--ready")
    parser.add_argument("--request", help="Single-request JSON file for diagnostics")
    args = parser.parse_args()
    worker = GraspGenWorker(args.config, args.grasp_repo)
    worker.load()
    if args.serve:
        if not args.ready:
            parser.error("--serve requires --ready")
        from .rpc import serve
        serve(worker.handle, args.ready)
    elif args.request:
        print(json.dumps(worker.handle(json.loads(Path(args.request).read_text()))))
    else:
        parser.error("Choose --serve or --request")


if __name__ == "__main__":
    main()
