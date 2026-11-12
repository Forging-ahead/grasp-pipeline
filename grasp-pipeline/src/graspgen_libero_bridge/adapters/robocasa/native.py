"""Native RoboCasa lifecycle, measured robot state and visible RGB-D target data.

Only reset creates a scene. Observation, planning and skill transitions never
reset it. Simulator truth is kept explicitly separate from camera observations.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .actions import ActionLayout
from .errors import EpisodeFailure, EpisodeStopped


def matrix(position, rotation):
    result = np.eye(4)
    result[:3, 3] = np.asarray(position, float)
    result[:3, :3] = np.asarray(rotation, float).reshape(3, 3)
    return result


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Cannot serialize runtime metadata: " + type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git_identity(directory):
    try:
        head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
                                            text=True, stderr=subprocess.DEVNULL, timeout=10).strip())
        return {"commit": head, "tracked_source_modified": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_source_modified": None}


def prepare_render_dependencies():
    """Load geometry plugins before creating an EGL context.

    On the pinned simulation stack, importing trimesh after the first render
    corrupts subsequent RGB, depth and segmentation buffers. This ordering is
    also exercised by the robot-only rendering/grasp diagnostic.
    """
    import trimesh  # noqa: F401


class RoboCasaEnvironment:
    def __init__(self, config, output, *, started=None, clock=time.monotonic):
        self.config = config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.started = clock() if started is None else started
        self.wall_budget = float(config["policy"]["episode_wall_seconds_composite" if
                                 config["task"]["category"] == "composite" else "episode_wall_seconds_atomic"])
        self.env = None
        self.robot = None
        self.layout = None
        self.obs = None
        self.horizon = None
        self.native_horizon = None
        self.steps = 0
        self.revision = 0
        self.episode_id = str(self.output.resolve())
        self.stopped = False
        self.native_success_at_reset = None
        self.native_success_seen = False
        self.last_native_done = False
        self.observers = []

    def remaining_seconds(self):
        return max(0., self.wall_budget - (self.clock() - self.started))

    def check_budget(self):
        if self.stopped:
            raise EpisodeStopped("EPISODE_STOPPED", "No further actions permitted")
        if self.remaining_seconds() <= 0:
            self.stopped = True
            raise EpisodeStopped("WALL_TIMEOUT", "Episode wall-clock budget exhausted")
        if self.horizon is not None and self.steps >= self.horizon:
            self.stopped = True
            raise EpisodeStopped("STEP_LIMIT", "Configured episode action limit reached")

    def reset(self):
        if self.env is not None:
            raise RuntimeError("An episode may only reset once; create a new environment for a new attempt")
        self.check_budget()
        paths = self.config.get("paths")
        if not paths:
            raise ValueError("Actual execution requires a RoboCasa paths file")
        repo = Path(paths["robocasa_repo"]).resolve()
        if not (repo / "robocasa/__init__.py").is_file():
            raise FileNotFoundError("RoboCasa checkout not found: " + str(repo))
        sys.path.insert(0, str(repo))
        prepare_render_dependencies()
        import robocasa
        import robosuite
        import mujoco
        from robocasa.utils.env_utils import create_env
        from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS

        if not Path(robocasa.__file__).resolve().is_relative_to(repo):
            raise RuntimeError("Imported RoboCasa does not match configured checkout")
        actual = git_identity(repo)
        expected = self.config["policy"]["native_source_commit"]
        if actual["commit"] != expected or actual["tracked_source_modified"]:
            raise EpisodeFailure("ENVIRONMENT_VERSION_MISMATCH", "Use the fixed, unmodified RoboCasa source",
                                 details={"expected": expected, "actual": actual})
        task_id = self.config["task"]["task_id"]
        entry = ATOMIC_TASK_DATASETS.get(task_id, COMPOSITE_TASK_DATASETS.get(task_id))
        if entry is None or self.config["split"] not in entry:
            raise EpisodeFailure("UNSUPPORTED_TASK_BINDING", "Native registry has no task/split: " + task_id)
        self.native_horizon = int(entry["horizon"])
        if self.native_horizon <= 0 or self.native_horizon != self.config["task"]["reference_horizon_steps"]:
            raise EpisodeFailure("ENVIRONMENT_VERSION_MISMATCH", "Native horizon differs from the fixed task manifest")
        self.horizon = self.native_horizon
        if self.config["policy"].get("budget_profile") == "libero-v10-teacher":
            value = self.config["policy"].get("teacher_step_limit")
            if type(value) is not int or value != 5000:
                raise EpisodeFailure("ENVIRONMENT_VERSION_MISMATCH", "The declared v10 teacher step limit is not 5000")
            self.horizon = value
        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        policy = self.config["policy"]
        self.env = create_env(task_id, robots=policy["robot"], split=self.config["split"], seed=seed,
                              camera_names=policy["camera_names"], camera_widths=policy["camera_size"],
                              camera_heights=policy["camera_size"], camera_depths=True,
                              control_freq=policy["control_freq"])
        self.check_budget()
        self.obs = self.env.reset()
        self._augment_instance_masks()
        if len(self.env.robots) != 1:
            raise EpisodeFailure("ROBOT_MODEL_MISMATCH", "Expected one PandaOmron")
        self.robot = self.env.robots[0]
        self.layout = ActionLayout.from_env(self.env)
        self.arm = self.layout.arm
        if len(self.arm.qpos_index) != 7:
            raise EpisodeFailure("ROBOT_MODEL_MISMATCH", "Expected seven Panda arm joints")
        self.native_success_at_reset = self.check_native_success()
        self.native_success_seen = self.native_success_at_reset
        self.save_physics(self.output / "INITIAL_STATE.npz")
        xml = self.env.model.get_xml()
        (self.output / "scene.xml").write_text(xml)
        success_source = inspect.getsource(type(self.env)._check_success)
        provenance = {"benchmark": "RoboCasa365", "robocasa": actual,
                      "robocasa_module": str(robocasa.__file__), "robosuite_version": robosuite.__version__,
                      "mujoco_version": mujoco.__version__, "simulation_python": sys.executable,
                      "task_id": task_id, "split": self.config["split"], "seed": seed,
                      "native_horizon_steps": self.native_horizon, "runner_step_limit": self.horizon,
                      "budget_profile": self.config["policy"].get("budget_profile", "native"),
                      "native_ignore_done": self.env.ignore_done,
                      "control_hz": float(self.env.control_freq), "action_spec": self.layout.describe(),
                      "episode_meta": self.env.get_ep_meta(), "scene_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                      "native_success_at_reset": self.native_success_at_reset,
                      "success_source_sha256": hashlib.sha256(success_source.encode()).hexdigest(),
                      "input_mode": self.config["input_mode"], "teacher_sources": ["task_definition", "simulator_instance_segmentation", "simulator_object_poses", "simulator_collision_geometry"]}
        write_json(self.output / "ENVIRONMENT_LOCK.json", provenance)
        (self.output / "native_success.py.txt").write_text(success_source)
        write_json(self.output / "INITIAL_RNG.json", {"python": random.getstate(),
                   "numpy_legacy": np.random.get_state(), "environment": self.env.rng.bit_generator.state})
        self.check_budget()
        return self.observe()

    def _augment_instance_masks(self):
        """Kitchen does not accept robosuite's camera_segmentations keyword.

        Render native object-type/geom-ID buffers at the same physical state,
        then map compiled object body subtrees to explicit instance labels.
        No detector, hidden full cloud, or environment reset is involved.
        """
        from robosuite import macros
        from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
        model = self.env.sim.model
        lookup = np.zeros(int(model.ngeom), dtype=np.int32)
        labels = {}
        for name, body in self.env.obj_body_id.items():
            bodies = body_subtree_ids(model, int(body))
            instance = int(body) + 1
            lookup[np.isin(np.asarray(model.geom_bodyid), list(bodies))] = instance
            labels[name] = instance
        self.instance_ids = labels
        size = int(self.config["policy"]["camera_size"])
        for camera in self.config["policy"]["camera_names"]:
            raw = np.asarray(self.env.sim.render(width=size, height=size,
                             camera_name=camera, segmentation=True))
            if raw.shape != (size, size, 2):
                raise EpisodeFailure("OBSERVATION_INVALID", "Unexpected native segmentation buffer", details={"shape": list(raw.shape)})
            # MuJoCo mjtObj.mjOBJ_GEOM == 5; background IDs are negative.
            geom_ids = raw[..., 1].astype(np.int64)
            valid = (raw[..., 0] == 5) & (geom_ids >= 0) & (geom_ids < len(lookup))
            instance = np.zeros((size, size), dtype=np.int32)
            instance[valid] = lookup[geom_ids[valid]]
            if str(macros.IMAGE_CONVENTION).lower() == "opencv":
                instance = np.flip(instance, axis=0)
            self.obs[camera + "_segmentation_instance"] = instance[..., None]

    def observe(self):
        if self.env is None or self.obs is None:
            raise RuntimeError("reset is required before observe")
        return {"episode_id": self.episode_id, "revision": self.revision,
                "step": self.steps, "sim_time": float(self.env.sim.data.time),
                "images": self.obs, "robot_state": self.robot_state(),
                "teacher_state_separate": True}

    def robot_state(self):
        model, data = self.env.sim.model, self.env.sim.data
        arm = self.arm
        gripper = self.robot.gripper["right"]
        finger_names = list(gripper.joints)
        finger_ids = [int(model.joint_name2id(name)) for name in finger_names]
        finger_qpos = [int(model.jnt_qposadr[j]) for j in finger_ids]
        site = int(model.site_name2id(gripper.important_sites["grip_site"]))
        first_joint = int(arm.joint_index[0])
        arm_base = int(model.body_parentid[int(model.jnt_bodyid[first_joint])])
        if arm.origin_pos is None or arm.origin_ori is None:
            raise EpisodeFailure("ROBOT_MODEL_MISMATCH", "Controller reference frame is unavailable")
        return {"q_arm": np.asarray(data.qpos[arm.qpos_index]).tolist(),
                "qvel_arm": np.asarray(data.qvel[arm.qvel_index]).tolist(),
                "q_fingers": np.asarray(data.qpos[finger_qpos]).tolist(),
                "joint_names": [model.joint_id2name(int(j)) for j in arm.joint_index],
                "finger_joint_names": finger_names,
                "joint_limits": np.asarray(model.jnt_range[arm.joint_index]).tolist(),
                "world_T_eef": matrix(data.site_xpos[site], data.site_xmat[site]).tolist(),
                "world_T_base": matrix(data.body_xpos[arm_base], data.body_xmat[arm_base]).tolist(),
                "world_T_control_base": matrix(arm.origin_pos, arm.origin_ori).tolist(),
                "grasp_T_tcp": (np.linalg.inv(matrix(data.body_xpos[model.body_name2id(gripper.root_body)],
                                   data.body_xmat[model.body_name2id(gripper.root_body)])) @
                                   matrix(data.site_xpos[site], data.site_xmat[site])).tolist(),
                "arm_base_body": model.body_id2name(arm_base), "eef_site": gripper.important_sites["grip_site"],
                "sim_time": float(data.time), "step_count": self.steps,
                "robot_base_pos": np.asarray(self.robot.base_pos).tolist(),
                "robot_base_ori": np.asarray(self.robot.base_ori).tolist()}

    def check_native_success(self):
        return bool(self.env._check_success())

    def step(self, action):
        self.check_budget()
        action = np.asarray(action, dtype=float)
        if action.shape != self.layout.low.shape or not np.isfinite(action).all():
            raise ValueError("Malformed native action")
        if np.any(action < self.layout.low - 1e-9) or np.any(action > self.layout.high + 1e-9):
            raise ValueError("Native action outside bounds")
        before = self.robot_state()
        started = self.clock()
        self.obs, reward, native_done, native_info = self.env.step(action)
        self.steps += 1
        self.revision += 1
        self.last_native_done = bool(native_done)
        # An image/segmentation error must not erase an already executed action.
        with (self.output / "ACTION_AUDIT.jsonl").open("a") as audit:
            audit.write(json.dumps({"step": self.steps, "action": action.tolist(),
                                    "sim_time": float(self.env.sim.data.time),
                                    "native_done": bool(native_done)}) + "\n")
        self._augment_instance_masks()
        success = self.check_native_success()
        self.native_success_seen = self.native_success_seen or success
        step = {"observation": self.observe(), "reward": float(reward),
                "terminated": bool(native_done), "truncated": self.steps >= self.horizon,
                "native_success": success, "native_info": native_info,
                "step_wall_s": self.clock() - started}
        # Record the final permitted action before blocking further actions.
        for observer in self.observers:
            observer(before, action, step)
        if native_done or self.steps >= self.horizon:
            self.stopped = True
        return step

    def object_pose(self, name):
        try:
            body = int(self.env.obj_body_id[name])
        except KeyError as exc:
            raise EpisodeFailure("UNSUPPORTED_TASK_BINDING", "Unknown task object: " + name) from exc
        data = self.env.sim.data
        return matrix(data.body_xpos[body], data.body_xmat[body])

    def entity_body(self, name):
        if name in self.env.obj_body_id:
            return int(self.env.obj_body_id[name])
        entity = getattr(self.env, name, None)
        if entity is None or not hasattr(entity, "root_body"):
            raise EpisodeFailure("UNSUPPORTED_TASK_BINDING", "Unknown task object or fixture: " + str(name))
        return int(self.env.sim.model.body_name2id(entity.root_body))

    def entity_pose(self, name):
        body = self.entity_body(name)
        data = self.env.sim.data
        return matrix(data.body_xpos[body], data.body_xmat[body])

    def target_cloud(self, target, camera=None):
        from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map
        from robosuite import macros
        from graspgen_libero_bridge.perception.pointcloud import depth_to_world_points
        cameras = [camera] if camera else self.config["policy"]["camera_names"]
        if target not in self.instance_ids:
            raise EpisodeFailure("OBSERVATION_INVALID", "Target not in native instance map: " + target)
        instance_id = self.instance_ids[target]
        best = None
        diagnostics = []
        for cam in cameras:
            depth = np.asarray(self.obs[cam + "_depth"])
            segmentation = np.asarray(self.obs[cam + "_segmentation_instance"])
            if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
                segmentation = segmentation[..., 0]
            # Some EGL buffers contain isolated nonfinite/out-of-range pixels.
            # Exclude them from the cloud; preserve the raw depth in recordings.
            valid_depth = np.isfinite(depth) & (depth >= 0.) & (depth <= 1.)
            if np.count_nonzero(~valid_depth) > .01 * depth.size:
                raise EpisodeFailure("OBSERVATION_INVALID", "Camera depth buffer is corrupted",
                                     details={"camera": cam, "invalid_pixels": int(np.count_nonzero(~valid_depth)),
                                              "total_pixels": int(depth.size)})
            clean_depth = np.where(valid_depth, depth, 1.).astype(depth.dtype)
            metric = get_real_depth_map(self.env.sim, clean_depth)
            h, w = depth.shape[:2]
            intrinsic = get_camera_intrinsic_matrix(self.env.sim, cam, h, w)
            extrinsic = get_camera_extrinsic_matrix(self.env.sim, cam)
            flip = str(macros.IMAGE_CONVENTION).lower() == "opengl"
            valid_mask = valid_depth[..., 0] if valid_depth.ndim == 3 else valid_depth
            points = depth_to_world_points(metric, intrinsic, extrinsic,
                                           (segmentation == instance_id) & valid_mask, flip_vertical=flip)
            info = {"camera": cam, "target": target, "instance_id": instance_id,
                    "visible_pixels": len(points), "revision": self.revision,
                    "invalid_depth_pixels_excluded": int(np.count_nonzero(~valid_depth)),
                    "flip_vertical": flip, "units": "meters", "frame": "world",
                    "mask_source": "simulator_ground_truth_instance", "cloud_source": "visible_depth_pixels",
                    "intrinsic": intrinsic.tolist(), "world_T_camera": extrinsic.tolist()}
            diagnostics.append(info)
            if best is None or len(points) > len(best[0]):
                best = (points, info)
        if best is None or len(best[0]) < self.config["policy"]["minimum_target_points"]:
            raise EpisodeFailure("TARGET_NOT_VISIBLE", "Insufficient visible target depth; no full-geometry fallback",
                                 details={"views": diagnostics})
        return best[0], dict(best[1], checked_views=diagnostics)

    def save_physics(self, path):
        data = getattr(self.env.sim.data, "_data", self.env.sim.data)
        fields = {name: np.array(getattr(data, name), copy=True) for name in
                  ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "qfrc_applied", "xfrc_applied", "userdata")
                  if hasattr(data, name)}
        fields.update(sim_time=np.asarray(float(data.time)), physical_steps=np.asarray(self.steps))
        np.savez_compressed(path, **fields)

    def close(self):
        self.stopped = True
        if self.env is not None:
            self.env.close()
