"""Regression cases for frame conversion, real action slices and episode exits."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.adapters.robocasa.actions import ActionLayout, physical_to_input
from graspgen_libero_bridge.adapters.robocasa.config import load_config
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeStopped
from graspgen_libero_bridge.adapters.robocasa.native import RoboCasaEnvironment


def make_layout(frame="base", kind="delta"):
    arm = NS(input_min=-np.ones(6), input_max=np.ones(6),
             output_min=np.array([-.05] * 3 + [-.5] * 3),
             output_max=np.array([.05] * 3 + [.5] * 3),
             input_ref_frame=frame, input_type=kind)
    native = NS(_action_split_indexes={"base": (0, 3), "right_gripper": (3, 4),
                "right": (4, 10), "torso": (10, 11)},
                part_controllers={"right": arm, "right_gripper": NS(), "base": NS(), "torso": NS()})
    env = NS(robots=[NS(composite_controller=native)], action_spec=(-np.ones(12), np.ones(12)))
    return env, ActionLayout.from_env(env)


def test_action_layout_is_read_not_assumed_arm_first():
    _, layout = make_layout()
    a = layout.assemble([.1] * 6, 1, base_velocity=(.2, .3, .4), torso=.5)
    np.testing.assert_allclose(a, [.2, .3, .4, 1, .1, .1, .1, .1, .1, .1, .5, -1])


def test_pose_error_rotates_into_native_base_frame():
    _, layout = make_layout()
    current = np.eye(4)
    goal = np.eye(4); goal[0, 3] = .025
    base = np.eye(4); base[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    action = layout.pose_action(goal, current, base, 1, gain=1)
    np.testing.assert_allclose(action[4:7], [0, -.5, 0], atol=1e-12)
    assert action[3] == 1


def test_rotation_is_not_discarded_and_uses_axis_angle():
    _, layout = make_layout(frame="world")
    goal = np.eye(4); goal[:3, :3] = Rotation.from_rotvec([.1, -.2, .05]).as_matrix()
    action = layout.pose_action(goal, np.eye(4), np.eye(4), -1, gain=1)
    np.testing.assert_allclose(action[7:10], [.2, -.4, .1], atol=1e-12)


def test_absolute_control_uses_metres_without_delta_scaling():
    _, layout = make_layout(frame="world", kind="absolute")
    goal = np.eye(4); goal[:3, 3] = [.4, .2, .8]
    action = layout.pose_action(goal, np.eye(4), np.eye(4), -1)
    np.testing.assert_allclose(action[4:7], [.4, .2, .8])


def test_reject_overlapping_or_unknown_native_parts():
    env, _ = make_layout()
    env.robots[0].composite_controller._action_split_indexes["base"] = (3, 6)
    with pytest.raises(ValueError, match="Overlapping"):
        ActionLayout.from_env(env)


def test_nonfinite_or_out_of_bounds_action_is_rejected():
    _, layout = make_layout()
    for values in ([np.nan] * 6, [2.] * 6):
        with pytest.raises(ValueError):
            layout.assemble(values, -1)


def test_reject_zero_scaling_range():
    _, layout = make_layout()
    layout.arm.output_min = layout.arm.output_max.copy()
    with pytest.raises(ValueError, match="scaling"):
        physical_to_input(np.zeros(6), layout.arm)


def test_native_horizon_last_action_is_recorded_and_no_extra_action_executes(tmp_path):
    cfg = load_config("PickPlaceCounterToStove")
    e = RoboCasaEnvironment(cfg, tmp_path, clock=lambda: 0.)
    _, e.layout = make_layout()
    calls, records = [], []
    def step(action):
        calls.append(action.copy())
        return {}, 0., False, {}
    e.env = NS(step=step, sim=NS(data=NS(time=0.)))
    e.obs = {}; e.horizon = 1
    e._augment_instance_masks = lambda: None
    e.robot_state = lambda: {"q_arm": [0] * 7}
    e.check_native_success = lambda: False
    e.observers.append(lambda *args: records.append(args))
    out = e.step(np.zeros(12))
    assert out["truncated"] and len(calls) == len(records) == 1
    with pytest.raises(EpisodeStopped):
        e.step(np.zeros(12))
    assert len(calls) == 1


def test_wall_budget_stops_before_touching_environment(tmp_path):
    cfg = load_config("PickPlaceCounterToStove")
    clock = [0.]
    e = RoboCasaEnvironment(cfg, tmp_path, clock=lambda: clock[0])
    clock[0] = cfg["policy"]["episode_wall_seconds_atomic"]
    with pytest.raises(EpisodeStopped) as error:
        e.step(np.zeros(12))
    assert error.value.code == "WALL_TIMEOUT"
    assert e.steps == 0


def test_native_termination_does_not_imply_task_success(tmp_path):
    e = RoboCasaEnvironment(load_config("PickPlaceCounterToStove"), tmp_path, clock=lambda: 0.)
    _, e.layout = make_layout()
    e.env = NS(step=lambda action: ({}, 0., True, {}), sim=NS(data=NS(time=0.)))
    e.obs = {}; e.horizon = 10
    e._augment_instance_masks = lambda: None
    e.robot_state = lambda: {}; e.check_native_success = lambda: False
    out = e.step(np.zeros(12))
    assert out["terminated"] and not out["native_success"] and e.stopped


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"gpu": -1}, {"split": "all"}])
def test_invalid_episode_identity_is_rejected(kwargs):
    with pytest.raises(ValueError):
        load_config("PickPlaceCounterToStove", **kwargs)


def test_unknown_task_and_unavailable_split_are_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        load_config("not_a_real_task")
    with pytest.raises(ValueError, match="no registered"):
        load_config("ArrangeBreadBasket", split="pretrain")
