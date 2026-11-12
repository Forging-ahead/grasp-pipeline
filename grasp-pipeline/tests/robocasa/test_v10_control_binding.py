from types import SimpleNamespace as NS

import numpy as np

from graspgen_libero_bridge.adapters.robocasa.config import load_config
from graspgen_libero_bridge.control.controller import Controller
from graspgen_libero_bridge.control.robocasa_executor import RoboCasaExecutor
from graspgen_libero_bridge.control.robocasa_v10_controller import V10MotionBinding
from test_actions_and_limits import make_layout


def test_native_controller_calls_original_move_to_with_inherited_integral_and_budgets(tmp_path):
    assert V10MotionBinding.move_to is Controller.move_to
    policy = load_config("PickPlaceCounterToStove")["policy"]
    assert policy["libero_v10_rules"]["position_integral_gain"] == .8
    assert policy["libero_v10_rules"]["place_tracking_extra_steps"] == 240
    _, layout = make_layout(frame="world")
    pose = np.eye(4)
    env = NS(output=tmp_path, steps=0, config={"policy": policy}, layout=layout, env=NS(control_freq=20))
    env.robot_state = lambda: {"world_T_eef": pose.copy(), "world_T_control_base": np.eye(4)}
    def step(action):
        # Persistent small load bias leaves a proportional-only equilibrium
        # outside 2.5 mm; the original bounded integral should overcome it.
        pose[:3, 3] += .5 * action[4:7] * .05
        pose[0, 3] -= .0015
        env.steps += 1
        return {}
    env.step = step
    executor = RoboCasaExecutor(env)
    goal = np.eye(4); goal[0, 3] = .03
    result = executor.move_to(goal, phase="place/waypoint_1", position_tolerance=.0025)
    assert result["reached"] and env.steps < 120
    assert np.linalg.norm(pose[:3, 3] - goal[:3, 3]) <= .0025
    assert result["controller"] == "unchanged_libero_v10_Controller.move_to"


def test_native_world_to_base_action_conversion_preserves_v10_command(tmp_path):
    policy = load_config("PickPlaceCounterToStove")["policy"]
    _, layout = make_layout(frame="base")
    base = np.eye(4); base[:2, :2] = [[0, -1], [1, 0]]
    calls = []
    env = NS(output=tmp_path, steps=0, config={"policy": policy}, layout=layout, env=NS(control_freq=20))
    env.robot_state = lambda: {"world_T_eef": np.eye(4), "world_T_control_base": base}
    env.step = lambda action: calls.append(action) or {}
    binding = V10MotionBinding(RoboCasaExecutor(env), "test", None)
    binding.step(np.array([.32, 0, 0, 0, 0, .2, 1.]))
    np.testing.assert_allclose(calls[0][4:7], [0, -.32, 0], atol=1e-12)
    np.testing.assert_allclose(calls[0][7:10], [0, 0, .2], atol=1e-12)
    assert calls[0][3] == 1.
