from types import SimpleNamespace as NS

import numpy as np

from graspgen_libero_bridge.control.robocasa_executor import StableNativeGoal, RoboCasaExecutor


def test_repeated_observation_reads_cannot_fake_a_second_of_stability():
    env = NS(steps=10, env=NS(control_freq=20), check_native_success=lambda: True)
    stable = StableNativeGoal(env, 1.)
    assert not any(stable() for _ in range(100))
    for step in range(11, 30):
        env.steps = step; assert not stable()
    env.steps = 30; assert stable()


def test_false_or_missing_physical_observations_reset_stability():
    good = [True]
    env = NS(steps=0, env=NS(control_freq=20), check_native_success=lambda: good[0])
    stable = StableNativeGoal(env, .15)
    assert not stable()
    env.steps = 1; assert not stable()
    env.steps = 5; assert not stable()  # Unobserved frames cannot be assumed good.
    good[0] = False; env.steps = 6; assert not stable()
    good[0] = True
    for step in (7, 8, 9):
        env.steps = step; assert not stable()
    env.steps = 10; assert stable()


def test_goal_stop_preserves_measured_completion_without_claiming_tcp_target_reached():
    env = NS(steps=600, config={"policy": {"move_timeout_s": 20}}, clock=lambda: 0., remaining_seconds=lambda: 0.,
             check_budget=lambda: (_ for _ in ()).throw(AssertionError("No extra action after the last permitted frame")))
    result = RoboCasaExecutor(env).move_to(np.eye(4), phase="optional_home", stop_when=lambda: True)
    assert result["task_goal_reached"] and not result["reached"]
    assert result["step"] == 600 and result["stop_reason"] == "native_goal_stable"
