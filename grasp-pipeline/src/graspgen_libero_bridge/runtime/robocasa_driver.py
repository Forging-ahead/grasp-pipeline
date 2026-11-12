"""Native probe, unloaded validation and real-task dispatch."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["probe", "validate", "run"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    from graspgen_libero_bridge.runtime.robocasa_provenance import source_identity, digest_json
    identity = source_identity()
    (args.output / "SOURCE_IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
    result = {"schema_version": "robocasa-result-1", "benchmark": "RoboCasa365",
              "mode": args.mode, "task_id": config["task"]["task_id"], "seed": config["seed"],
              "split": config["split"], "input_mode": config["input_mode"],
              "status": "running", "pipeline_success": False, "training_candidate": False,
              "source_digest": identity["source_digest"], "policy_digest": digest_json(config["policy"]),
              "native_success_at_reset": None, "native_success_at_end": None,
              "native_success_seen": False, "physical_steps": 0, "video_available": False}
    result["candidate_source"] = ("recorded_identical_pool" if config.get("candidate_replay_case") else
                                  "partial_cloud_geometric" if config["policy"].get("grasp_generator") == "geometric" else
                                  "fresh_graspgen_inference")
    result["paired_diagnostic"] = bool(config.get("candidate_replay_case"))
    result["budget_profile"] = config["policy"].get("budget_profile", "native")
    environment = recorder = None
    try:
        from graspgen_libero_bridge.adapters.robocasa.native import RoboCasaEnvironment, write_json
        from graspgen_libero_bridge.adapters.robocasa.recording import TrajectoryRecorder
        from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
        from graspgen_libero_bridge.planning.robocasa_local import LocalModel
        from graspgen_libero_bridge.planning.robocasa_robot import check_robot_model
        from graspgen_libero_bridge.control.robocasa_executor import RoboCasaExecutor
        from graspgen_libero_bridge.adapters.robocasa.capabilities import capabilities

        environment = RoboCasaEnvironment(config, args.output, started=started)
        write_json(args.output / "CAPABILITIES.json", capabilities(config))
        environment.reset()
        recorder = TrajectoryRecorder(environment)
        result["native_success_at_reset"] = environment.native_success_at_reset
        state = environment.robot_state()
        write_json(args.output / "ROBOT_STATE.json", state)
        write_json(args.output / "TASK_BINDING.json", config["task"])
        if args.mode in {"validate", "run"}:
            local = LocalModel(environment)
            samples = local.calibration_samples()
            write_json(args.output / "FK_SAMPLES.json", samples)
            calibration = check_robot_model(environment, samples)
            write_json(args.output / "ROBOT_CALIBRATION.json", calibration)
            if not calibration["passed"]:
                raise EpisodeFailure("ROBOT_MODEL_MISMATCH", "Planner robot FK failed calibration", details=calibration)
            executor = RoboCasaExecutor(environment, recorder)
            executor.monitor = local
            if args.mode == "validate" and args.execute:
                import numpy as np
                from scipy.spatial.transform import Rotation
                initial = np.asarray(state["world_T_eef"])
                checks = []
                for axis in range(3):
                    for sign in (1, -1):
                        goal = initial.copy()
                        goal[axis, 3] += sign * .02
                        solved = local.solve(goal)
                        if not solved["success"]:
                            checks.append({"axis": axis, "sign": sign, "executed": False, "ik": solved})
                            continue
                        outcome = executor.move_to(goal, phase="validate/position")
                        checks.append({"axis": axis, "sign": sign, "executed": True, "outcome": outcome})
                        executor.move_to(initial, phase="validate/return")
                for axis in range(3):
                    for sign in (1, -1):
                        goal = initial.copy()
                        delta = np.zeros(3); delta[axis] = sign * .12
                        goal[:3, :3] = Rotation.from_rotvec(delta).as_matrix() @ initial[:3, :3]
                        solved = local.solve(goal)
                        if not solved["success"]:
                            checks.append({"rotation_axis": axis, "sign": sign, "executed": False, "ik": solved})
                            continue
                        outcome = executor.move_to(goal, phase="validate/rotation")
                        checks.append({"rotation_axis": axis, "sign": sign, "executed": True, "outcome": outcome})
                        executor.move_to(initial, phase="validate/return")
                executor.hold(.5, gripper=1, phase="validate/close")
                executor.hold(.5, gripper=-1, phase="validate/open")
                write_json(args.output / "CONTROL_CHECKS.json", checks)
                if sum(c["executed"] for c in checks) < 10:
                    raise EpisodeFailure("CONTROL_CHECK_INCOMPLETE", "Too few safe targets were executed", details={"checks": checks})
        if args.mode == "run":
            from graspgen_libero_bridge.policy.robocasa_skills import run_policy
            result.update(run_policy(environment, executor, local, recorder))
        else:
            result.update(status="probe_passed" if args.mode == "probe" else "validation_passed",
                          validation_executed=args.execute,
                          validation_scope="native lifecycle" if args.mode == "probe" else
                          ("CPU FK and unloaded control" if args.execute else "CPU FK only"))
    except Exception as exc:
        code = getattr(exc, "code", "ASSET_MISSING" if isinstance(exc, FileNotFoundError) else "ENVIRONMENT_ERROR")
        result.update(status="failed", failure_code=code,
                      error=str(exc), details=getattr(exc, "details", {}), traceback=traceback.format_exc())
        result["failure_stage"] = recorder.phase if recorder is not None else "environment_initialization"
    finally:
        if environment is not None and environment.env is not None:
            try:
                result.update(physical_steps=environment.steps, native_horizon=environment.native_horizon,
                              runner_step_limit=environment.horizon,
                              within_native_horizon=environment.steps <= environment.native_horizon,
                              native_budget_success=bool(result.get("pipeline_success") and environment.check_native_success()
                                                         and environment.steps <= environment.native_horizon),
                              native_success_at_reset=environment.native_success_at_reset,
                              native_success_at_end=environment.check_native_success(),
                              native_success_seen=environment.native_success_seen,
                              native_done=environment.last_native_done)
                environment.save_physics(args.output / "FINAL_STATE.npz")
            except Exception as exc:
                result["final_state_error"] = str(exc)
        if recorder is not None:
            try:
                recorder.close()
                result["video_available"] = (args.output / "VIDEO.mp4").is_file()
            except Exception as exc:
                result["recording_error"] = str(exc)
        if environment is not None:
            try:
                environment.close()
            except Exception as exc:
                result["close_error"] = str(exc)
        result["wall_s"] = time.monotonic() - started
        result["source_unchanged_during_run"] = source_identity()["source_digest"] == identity["source_digest"]
        if not result["source_unchanged_during_run"]:
            result.update(status="failed", failure_code="SOURCE_CHANGED_DURING_RUN", pipeline_success=False)
        result["native_budget_success"] = bool(result.get("pipeline_success") and result.get("native_success_at_end")
                                               and result.get("within_native_horizon"))
        result_path = args.output / "RESULT.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: result.get(key) for key in ("status", "failure_code", "pipeline_success", "physical_steps", "wall_s")}), flush=True)
    return 0 if result["status"] in {"probe_passed", "validation_passed", "success"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
