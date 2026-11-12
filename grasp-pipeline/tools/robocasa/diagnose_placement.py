#!/usr/bin/env python3
"""Offline geometry-only placement sweeps in a clone of a failed physics state.

Controller/task internals are not replayed. No actions are issued and no result
can be counted as task success. Run with the isolated simulation interpreter.
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
    parser.add_argument("--clearances", nargs="+", type=float, default=[.05, .10, .15])
    args = parser.parse_args()
    result = json.loads((args.case / "RESULT.json").read_text())
    if args.output.exists() or not 1 <= args.limit <= 30:
        parser.error("Use a new output directory and 1–30 placement candidates")
    if result.get("physical_steps", 0) <= 0 or result.get("error") != "No executable transfer/placement goal; object remains held":
        parser.error("Expected a failed post-lift placement search")
    from graspgen_libero_bridge.runtime.robocasa_provenance import source_identity
    if source_identity()["source_digest"] != result["source_digest"]:
        parser.error("Use the exact actor version from the failed episode")
    import numpy as np
    import mujoco
    from graspgen_libero_bridge.adapters.robocasa.native import RoboCasaEnvironment, write_json
    from graspgen_libero_bridge.adapters.robocasa.scene import export_scene
    from graspgen_libero_bridge.adapters.robocasa.goals import placement_goals
    from graspgen_libero_bridge.planning.robocasa_local import LocalModel
    from graspgen_libero_bridge.planning.robocasa_robot import self_collision_pairs
    from graspgen_libero_bridge.planning.planning_geometry import PandaMJCF
    if any(not np.isfinite(h) or not .01 <= h <= .3 for h in args.clearances):
        parser.error("Clearances must be finite and within 0.01–0.30 m")
    config = json.loads((args.case / "CONFIG.json").read_text())
    stage = config["task"]["stages"][0]
    args.output.mkdir(parents=True)
    env = RoboCasaEnvironment(config, args.output)
    report = {"scope": "offline geometry in a snapshot clone; no controller/task replay or task success",
              "source_digest": result["source_digest"], "physical_actions_issued": 0,
              "native_task_replay_claimed": False, "controller_state_restored": False, "endpoints": []}
    try:
        env.reset()
        before = json.loads((args.case / "ENVIRONMENT_LOCK.json").read_text())
        clone = json.loads((args.output / "ENVIRONMENT_LOCK.json").read_text())
        if before["scene_xml_sha256"] != clone["scene_xml_sha256"]:
            raise ValueError("Clone scene/asset identity differs")
        initial_orientation = env.object_pose(stage["target"])[:3, :3].copy()
        data = getattr(env.env.sim.data, "_data", env.env.sim.data)
        model = getattr(env.env.sim.model, "_model", env.env.sim.model)
        with np.load(args.case / "FINAL_STATE.npz", allow_pickle=False) as saved:
            for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "qfrc_applied", "xfrc_applied", "userdata"):
                if name in saved:
                    getattr(data, name)[...] = saved[name]
            data.time = float(saved["sim_time"])
        mujoco.mj_forward(model, data)
        env.save_physics(args.output / "CLONED_PHYSICS.npz")
        local = LocalModel(env)
        local.target = stage["target"]; local.allow_finger_target = True
        local.attach([stage["target"]])
        state = env.robot_state()
        scene = export_scene(env, stage["target"])
        robot = PandaMJCF(args.output / "robot_proxy", assets_dir=Path(config["paths"]["robosuite_assets"]))
        q = np.asarray(state["q_arm"])
        retract = np.asarray(robot.robot_config(state["q_fingers"])["kinematics"]["cspace"]["retract_config"][:7])
        rng = np.random.default_rng(0)
        seeds = [q, retract, *[np.clip(q + rng.uniform(-1, 1, 7), local.limits[:, 0] + .001,
                                     local.limits[:, 1] - .001) for _ in range(2)]]
        tcp_T_object = np.linalg.inv(np.asarray(state["world_T_eef"])) @ env.object_pose(stage["target"])
        for index, candidate in enumerate(placement_goals(env, stage, scene["target"]["points"], initial_orientation)):
            if index >= args.limit:
                break
            for height in args.clearances:
                desired = candidate["world_T_object"].copy(); desired[2, 3] += height
                goal = desired @ np.linalg.inv(tcp_T_object)
                attempts = []
                for seed_index, seed in enumerate(seeds):
                    solved = local.solve(goal, start=seed)
                    attempts.append(dict(solved, seed_index=seed_index))
                    if solved["success"]:
                        attempts[-1]["proxy_self_collisions"] = self_collision_pairs(robot, solved["q"], state["q_fingers"])
                        break
                report["endpoints"].append({"candidate": index, "clearance_m": height,
                    "yaw_degrees": candidate["yaw_degrees"], "offset_uv": candidate["offset_uv"],
                    "world_T_tcp": goal, "native_collision_free_ik_found": any(a["success"] for a in attempts),
                    "attempts": attempts})
                write_json(args.output / "PLACEMENT_DIAGNOSTIC.json", report)
        assert env.steps == 0
        report["complete"] = True
        write_json(args.output / "PLACEMENT_DIAGNOSTIC.json", report)
        print(json.dumps({"endpoints": len(report["endpoints"]), "native_collision_free_ik_found":
                          sum(e["native_collision_free_ik_found"] for e in report["endpoints"]), "physical_actions": 0}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
