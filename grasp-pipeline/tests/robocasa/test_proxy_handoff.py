from types import SimpleNamespace as NS

import numpy as np
import pytest

from graspgen_libero_bridge.planning.robocasa_planner_worker import RoboCasaHandler, GraphOnlyMotionGen


def test_local_lift_path_that_enters_a_curobo_obstacle_is_rejected():
    calls = []
    planner = NS(sync=lambda scene, contact: calls.append((scene, contact)),
                 validate_path=lambda q: {"valid": False, "invalid_indices": [2]},
                 world_report={"optimizer_excluded_geometry_ids": ["grasped_target"]})
    handler = RoboCasaHandler(); handler.planner = planner
    response = handler({"command": "check_joint_path", "scene": {"revision": 10},
                        "q_arm": np.zeros((3, 7)).tolist(), "allow_target_contact": True})
    assert not response["valid"] and response["invalid_indices"] == [2]
    assert calls == [({"revision": 10}, True)]
    assert response["backend"] == "curobo_collision_constraints"


def test_proxy_check_rejects_invalid_path_before_updating_world():
    handler = RoboCasaHandler()
    handler.planner = NS(sync=lambda *args: pytest.fail("Invalid input reached the planner"))
    with pytest.raises(ValueError, match="finite Nx7"):
        handler({"command": "check_joint_path", "scene": {}, "q_arm": [[float("nan")] * 7]})


def test_graph_fallback_preserves_deadline_and_does_not_modify_original_config():
    config = NS(enable_graph=False, enable_opt=True, enable_graph_attempt=1,
                enable_finetune_trajopt=True, timeout=7., max_attempts=2)
    captured = []
    wrapped = NS(plan_single=lambda start, goal, cfg: captured.append(cfg), rollout_fn="same_collision_model")
    proxy = GraphOnlyMotionGen(wrapped)
    proxy.plan_single(None, None, config)
    assert captured[0].enable_graph and not captured[0].enable_opt
    assert captured[0].timeout == 7. and captured[0].max_attempts == 2
    assert config.enable_opt and not config.enable_graph
    assert proxy.rollout_fn == "same_collision_model"


def test_failed_graph_request_restores_bridge_backend_reference():
    original = NS()
    def error(request):
        raise RuntimeError("graph error")
    handler = RoboCasaHandler(); handler.planner = NS(motion_gen=original, plan_pose=error)
    with pytest.raises(RuntimeError, match="graph error"):
        handler({"command": "plan_pose_graph"})
    assert handler.planner.motion_gen is original


def test_joint_graph_mode_uses_the_same_numeric_goal_and_deadline():
    config = NS(enable_graph=False, enable_opt=True, enable_graph_attempt=1,
                enable_finetune_trajopt=True, timeout=4., max_attempts=2)
    goal = object(); captured = []
    wrapped = NS(plan_single_js=lambda start, target, cfg: captured.append((target, cfg)))
    GraphOnlyMotionGen(wrapped).plan_single_js(None, goal, config)
    assert captured[0][0] is goal
    assert captured[0][1].enable_graph and not captured[0][1].enable_opt
    assert captured[0][1].timeout == 4.
    assert not config.enable_graph


def test_joint_target_outside_limits_is_rejected_before_importing_cuda():
    handler = RoboCasaHandler(); handler.planner = NS(robot=NS(arm_limits=[[-1., 1.]] * 7))
    with pytest.raises(ValueError, match="outside robot limits"):
        handler({"command": "plan_joint", "q_goal": [2.] * 7})
