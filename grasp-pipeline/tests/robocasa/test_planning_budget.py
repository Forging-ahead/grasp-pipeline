from types import SimpleNamespace as NS

import numpy as np

from graspgen_libero_bridge.planning.robocasa_planner import RoboCasaPlanner


def test_rejected_interpolation_is_replanned_with_remaining_original_budget(tmp_path):
    clock = [0.]
    env = NS(output=tmp_path, clock=lambda: clock[0], remaining_seconds=lambda: 100., check_budget=lambda: None,
             config={"policy": {"global_planner": "curobo", "plan_timeout_s": 15., "plan_validation_retries": 1}})
    planner = RoboCasaPlanner(env, NS(validate_path=lambda q: {"valid": True}))
    planner.ensure_configuration = lambda target: None
    planner.scene = lambda target: {}
    calls = []
    def request(message):
        calls.append(dict(message))
        clock[0] += 6
        if len(calls) == 1:
            return {"success": False, "reason": "interpolated_path_validation_failed"}
        path = tmp_path / "valid.npz"
        np.savez(path, q_arm=np.zeros((2, 7)))
        return {"success": True, "path_file": str(path)}
    planner._request = request
    result = planner.global_plan("obj", np.eye(4), label="pregrasp")
    assert result["success"]
    assert [c["timeout_s"] for c in calls] == [15., 9.]
    assert [c["command"] for c in calls] == ["plan_pose", "plan_pose_graph"]
    assert calls[0]["out_dir"] != calls[1]["out_dir"]
    assert len(result["planning_attempts"]) == 2


def test_no_retry_after_budget_was_consumed_or_for_ordinary_ik_failure(tmp_path):
    for reason in ("interpolated_path_validation_failed", "curobo_plan_failed"):
        clock = [0.]
        env = NS(output=tmp_path / reason, clock=lambda: clock[0], remaining_seconds=lambda: 100., check_budget=lambda: None,
                 config={"policy": {"global_planner": "curobo", "plan_timeout_s": 15., "plan_validation_retries": 1}})
        planner = RoboCasaPlanner(env, NS())
        planner.ensure_configuration = lambda target: None
        planner.scene = lambda target: {}
        calls = []
        def request(message):
            calls.append(dict(message)); clock[0] += 15.
            return {"success": False, "reason": reason}
        planner._request = request
        assert not planner.global_plan("obj", np.eye(4), label="pregrasp")["success"]
        assert len(calls) == 1
