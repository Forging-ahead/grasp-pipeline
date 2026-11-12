#!/usr/bin/env python3
"""PandaOmron-only diagnostic using robosuite assets; no kitchen-task score.

Run with the isolated simulation interpreter. This checks the actual composite
controller and planner model when external kitchen assets are unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

SOURCE_ROOT = str(Path(__file__).resolve().parents[2] / "src")
sys.path.insert(0, SOURCE_ROOT)
os.environ["PYTHONPATH"] = SOURCE_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--grasp", action="store_true", help="Run real RGB-D GraspGen/cuRobo grasp-and-lift diagnostic; no kitchen-task score")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to reuse an output directory")
    args.output.mkdir(parents=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import numpy as np
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from graspgen_libero_bridge.adapters.robocasa.actions import ActionLayout
    from graspgen_libero_bridge.adapters.robocasa.config import load_config
    from graspgen_libero_bridge.adapters.robocasa.native import RoboCasaEnvironment, write_json, prepare_render_dependencies
    from graspgen_libero_bridge.planning.robocasa_local import LocalModel
    from graspgen_libero_bridge.planning.robocasa_robot import check_robot_model
    from graspgen_libero_bridge.control.robocasa_executor import RoboCasaExecutor
    from graspgen_libero_bridge.runtime.robocasa_provenance import source_identity, digest_json

    identity = source_identity()
    result = {"scope": "PandaOmron robot/controller diagnostic in robosuite Lift; NOT a RoboCasa task rollout",
              "passed": False, "kitchen_task_success_claimed": False, "source_digest": identity["source_digest"]}
    raw = None
    recorder = skill = planner = e = None
    try:
        config = load_config("PickPlaceCounterToStove", paths_file=args.paths)
        np.random.seed(config["seed"])
        import random
        random.seed(config["seed"])
        if args.grasp:
            config["policy"]["camera_names"] = ["agentview", "robot0_eye_in_hand"]
        write_json(args.output / "CONFIG.json", config)
        result["policy_digest"] = digest_json(config["policy"])
        prepare_render_dependencies()
        raw = robosuite.make("Lift", robots="PandaOmron",
                    controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
                    has_renderer=False, has_offscreen_renderer=args.grasp, use_camera_obs=args.grasp,
                    camera_names=config["policy"]["camera_names"], camera_depths=args.grasp,
                    camera_heights=256, camera_widths=256,
                    control_freq=config["policy"]["control_freq"], ignore_done=True, seed=config["seed"])
        obs = raw.reset()
        raw.obj_body_id = {"cube": raw.cube_body_id}
        raw.objects = {"cube": raw.cube}
        e = RoboCasaEnvironment(config, args.output)
        e.env, e.obs, e.robot = raw, obs, raw.robots[0]
        e.layout = ActionLayout.from_env(raw)
        e.arm = e.layout.arm
        e.horizon = 1000
        e.save_physics(args.output / "INITIAL_STATE.npz")
        (args.output / "scene.xml").write_text(raw.model.get_xml())
        if args.grasp:
            e._augment_instance_masks()
            from graspgen_libero_bridge.adapters.robocasa.recording import TrajectoryRecorder
            recorder = TrajectoryRecorder(e)
        else:
            e._augment_instance_masks = lambda: None  # Control-only diagnostic has no camera input.
        state = e.robot_state()
        local = LocalModel(e)
        calibration = check_robot_model(e, local.calibration_samples())
        write_json(args.output / "ROBOT_CALIBRATION.json", calibration)
        result.update(calibration=calibration, action_spec=e.layout.describe(), initial_state=state)
        if not calibration["passed"]:
            raise RuntimeError("PandaOmron arm does not match the selected planner model")
        if args.execute:
            executor = RoboCasaExecutor(e, recorder)
            executor.monitor = local
            initial = np.asarray(state["world_T_eef"])
            goal = initial.copy(); goal[2, 3] += .025
            reply = local.solve(goal)
            if not reply["success"]:
                raise RuntimeError("Unloaded check target was not valid: " + str(reply))
            result["position_execution"] = executor.move_to(goal, phase="robot_diagnostic/up")
            result["return_execution"] = executor.move_to(initial, phase="robot_diagnostic/return")
            from scipy.spatial.transform import Rotation
            rotated = initial.copy()
            rotated[:3, :3] = Rotation.from_rotvec([0., 0., .12]).as_matrix() @ initial[:3, :3]
            reply = local.solve(rotated)
            if not reply["success"]:
                raise RuntimeError("Rotation check target was not valid: " + str(reply))
            result["rotation_execution"] = executor.move_to(rotated, phase="robot_diagnostic/rotate")
            executor.move_to(initial, phase="robot_diagnostic/return")
            executor.hold(.5, gripper=1, phase="robot_diagnostic/close")
            closed = e.robot_state()["q_fingers"]
            executor.hold(.5, gripper=-1, phase="robot_diagnostic/open")
            opened = e.robot_state()["q_fingers"]
            result.update(closed_fingers=closed, opened_fingers=opened, physical_steps=e.steps)
            if np.linalg.norm(np.asarray(opened)) <= np.linalg.norm(np.asarray(closed)) + .005:
                raise RuntimeError("Gripper did not physically open after the close/open commands")
        if args.grasp:
            from graspgen_libero_bridge.planning.robocasa_planner import RoboCasaPlanner
            from graspgen_libero_bridge.policy.robocasa_skills import PickPlaceSkill
            executor = RoboCasaExecutor(e, recorder)
            executor.monitor = local
            planner = RoboCasaPlanner(e, local)
            skill = PickPlaceSkill(e, executor, local, recorder, planner)
            from graspgen_libero_bridge.planning.robocasa_robot import prepare_start_state
            result["start_recovery"] = prepare_start_state(e, executor, local)
            directory = args.output / "grasp_diagnostic"
            carried = skill._attempt_grasp({"target": "cube"}, directory)
            result.update(grasp_and_lift_passed=True, physical_steps=e.steps,
                          native_Lift_success=e.check_native_success(),
                          diagnostic_only=True)
        result["passed"] = True
    except Exception as exc:
        result.update(error=str(exc), failure_code=getattr(exc, "code", "ROBOT_DIAGNOSTIC_ERROR"),
                      details=getattr(exc, "details", {}), traceback=traceback.format_exc())
    finally:
        if skill is not None:
            skill.close()
        if planner is not None:
            planner.close()
        if recorder is not None:
            recorder.close()
        if e is not None:
            result["physical_steps"] = e.steps
            if e.env is not None and e.layout is not None:
                e.save_physics(args.output / "FINAL_STATE.npz")
        if raw is not None:
            raw.close()
        result["source_unchanged_during_run"] = source_identity()["source_digest"] == identity["source_digest"]
        write_json(args.output / "ROBOT_DIAGNOSTIC.json", result)
        print(json.dumps({"passed": result["passed"], "output": str(args.output), "error": result.get("error")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
