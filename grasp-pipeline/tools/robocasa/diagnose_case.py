#!/usr/bin/env python3
"""Read-only endpoint diagnostics for a zero-action candidate-search failure.

Recreate and verify the exact native initial state, then use isolated MjData.
This is not a physical replay, a successful trajectory, or a no-path proof.
Failures after physical movement require a later full controller/task replay.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--ik-seeds", type=int, default=4)
    parser.add_argument("--approach-distances", nargs="+", type=float,
                        help="Shared approach offsets in meters; diagnostic only")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.limit <= 16 or not 1 <= args.ik_seeds <= 16:
        parser.error("Use a new output directory, 1–16 candidates and 1–16 IK seeds")
    result = json.loads((args.case / "RESULT.json").read_text())
    if result.get("physical_steps") != 0 or result.get("failure_code") != "SEARCH_BUDGET_EXHAUSTED":
        parser.error("This diagnostic only supports zero-action candidate-search failures")
    from graspgen_libero_bridge.runtime.robocasa_provenance import source_identity
    if source_identity()["source_digest"] != result["source_digest"]:
        parser.error("Use the exact actor source version from the failed case")
    config = json.loads((args.case / "CONFIG.json").read_text())
    import numpy as np
    distances = args.approach_distances or [config["policy"]["pregrasp_distance_m"]]
    if any(not np.isfinite(d) or not .01 <= d <= .3 for d in distances):
        parser.error("Approach distances must be finite and within 0.01–0.30 m")
    from scipy.spatial.transform import Rotation
    from graspgen_libero_bridge.adapters.robocasa.native import RoboCasaEnvironment, write_json
    from graspgen_libero_bridge.planning.robocasa_local import LocalModel
    from graspgen_libero_bridge.planning.robocasa_robot import self_collision_pairs
    from graspgen_libero_bridge.planning.planning_geometry import PandaMJCF

    args.output.mkdir(parents=True)
    env = RoboCasaEnvironment(config, args.output)
    report = {"scope": "isolated endpoint IK at verified native initial state; no physical replay",
              "case": str(args.case.resolve()), "source_digest": result["source_digest"],
              "physical_steps": 0, "endpoint_results": [], "success_trajectory_claimed": False}
    try:
        env.reset()
        original = json.loads((args.case / "ENVIRONMENT_LOCK.json").read_text())
        recreated = json.loads((args.output / "ENVIRONMENT_LOCK.json").read_text())
        if recreated["scene_xml_sha256"] != original["scene_xml_sha256"]:
            raise ValueError("Recreated scene differs from the failed case")
        with np.load(args.case / "INITIAL_STATE.npz", allow_pickle=False) as saved:
            for key in ("qpos", "qvel", "ctrl"):
                if not np.allclose(saved[key], getattr(env.env.sim.data, key), atol=1e-12, rtol=0):
                    raise ValueError("Initial native physics differs: " + key)
        report["exact_initial_state_verified"] = True
        local = LocalModel(env)
        robot = PandaMJCF(args.output / "robot_proxy", assets_dir=Path(config["paths"]["robosuite_assets"]))
        state = env.robot_state()
        initial = np.asarray(state["q_arm"])
        rng = np.random.default_rng(0)
        seeds = [initial]
        if args.ik_seeds > 1:
            seeds.append(np.asarray(robot.robot_config(state["q_fingers"])["kinematics"]["cspace"]["retract_config"][:7]))
        seeds.extend(np.clip(initial + rng.uniform(-1., 1., 7), local.limits[:, 0] + .001, local.limits[:, 1] - .001)
                     for _ in range(max(0, args.ik_seeds - 2)))
        stage = config["task"]["stages"][0]
        local.target = stage["target"]
        directories = sorted((args.case / "stages").glob("*/grasp_*/CANDIDATES.npz"))
        if len(directories) != 1:
            raise ValueError("Expected one recorded candidate pool before any physical action")
        with np.load(directories[0], allow_pickle=False) as data:
            poses, scores = data["poses_world_grasp"], data["scores"]
        order = np.argsort(-scores, kind="stable")[:args.limit]
        for index in order:
            for symmetry in (0., np.pi):
                grasp = poses[index].copy()
                grasp[:3, :3] = grasp[:3, :3] @ Rotation.from_euler("z", symmetry).as_matrix()
                tcp = grasp @ np.asarray(state["grasp_T_tcp"])
                goals = [("grasp", tcp, None)]
                for distance in distances:
                    pre = tcp.copy(); pre[:3, 3] -= pre[:3, 2] * distance
                    goals.append(("pregrasp", pre, distance))
                for label, goal, distance in goals:
                    local.allow_finger_target = label == "grasp"
                    attempts = []
                    for seed_index, seed in enumerate(seeds):
                        solved = local.solve(goal, start=seed)
                        contacts = solved.get("contacts", [])
                        for contact in contacts:
                            contact["geom1_name"] = env.env.sim.model.geom_id2name(contact["geom1"])
                            contact["geom2_name"] = env.env.sim.model.geom_id2name(contact["geom2"])
                        attempts.append(dict(solved, seed_index=seed_index))
                        if solved["success"]:
                            attempts[-1]["proxy_self_collisions"] = self_collision_pairs(robot, solved["q"], state["q_fingers"])
                            break
                    report["endpoint_results"].append({"candidate_index": int(index), "score": float(scores[index]),
                         "symmetry_rad": symmetry, "phase": label, "world_T_tcp": goal,
                         "approach_distance_m": distance,
                         "native_collision_free_ik_found": any(a["success"] for a in attempts), "attempts": attempts})
                    write_json(args.output / "ENDPOINT_DIAGNOSTIC.json", report)
        report["physical_steps"] = env.steps
        report["limitations"] = ["IK search failure is not an infeasibility proof", "Endpoints do not certify a valid connecting path",
                                 "No collision constraints were relaxed", "Symmetries are diagnostic variants, not substituted benchmark outcomes"]
        write_json(args.output / "ENDPOINT_DIAGNOSTIC.json", report)
        print(json.dumps({"physical_steps": env.steps, "endpoints": len(report["endpoint_results"]),
                          "native_collision_free_ik_found": sum(r["native_collision_free_ik_found"] for r in report["endpoint_results"])}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
