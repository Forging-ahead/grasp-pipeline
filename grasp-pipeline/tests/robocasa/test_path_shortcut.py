import numpy as np

from graspgen_libero_bridge.planning.robocasa_shortcut import joint_length, shortcut_path


def detour():
    result = np.zeros((4, 7))
    result[:, :2] = [[0, 0], [0, 1], [1, 1], [1, 0]]
    return result


def test_free_space_shortcut_reduces_motion_and_preserves_endpoints():
    original = detour()
    result, report = shortcut_path(original, lambda q: True, max_checks=8, remaining_seconds=lambda: 1.)
    np.testing.assert_array_equal(result[[0, -1]], original[[0, -1]])
    assert joint_length(result) < joint_length(original)
    assert report["accepted_shortcuts"] == 1
    np.testing.assert_array_equal(original, detour())


def test_a_shortcut_cannot_cut_through_a_blocked_joint_region():
    def safe(q):
        return not np.any((q[:, 0] > .4) & (q[:, 0] < .6) & (q[:, 1] < .2))
    result, report = shortcut_path(detour(), safe, max_checks=8, remaining_seconds=lambda: 1.)
    assert safe(result)
    assert not report["checks"][0]["valid"]


def test_exhausted_budget_cannot_launch_more_collision_queries():
    def unexpected(q):
        raise AssertionError("No query is allowed after the deadline")
    result, report = shortcut_path(detour(), unexpected, max_checks=8, remaining_seconds=lambda: 0.)
    np.testing.assert_array_equal(result, detour())
    assert report["checks"] == []
