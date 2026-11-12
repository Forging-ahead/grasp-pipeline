from types import SimpleNamespace as NS

import numpy as np
import pytest

from graspgen_libero_bridge.policy.robocasa_skills import grasp_precheck_from_pregrasp


def test_contact_pose_is_checked_from_planned_branch_not_distant_initial_posture(tmp_path):
    start = np.zeros(7)
    reachable_branch = np.arange(7) * .1
    file = tmp_path / "path.npz"
    np.savez(file, q_arm=np.array([start, reachable_branch]))
    local = NS(allow_finger_target=False)
    def solve(goal, start=None):
        # The observed failure: single-start IK picks a colliding elbow branch.
        if start is not None:
            assert local.allow_finger_target
        return {"success": start is not None and np.allclose(start, reachable_branch)}
    local.solve = solve
    assert not solve(np.eye(4))["success"]
    assert grasp_precheck_from_pregrasp(local, file, np.eye(4))["success"]
    assert not local.allow_finger_target


def test_failed_precheck_restores_contact_permission(tmp_path):
    file = tmp_path / "path.npz"; np.savez(file, q_arm=np.zeros((2, 7)))
    def error(*args, **kwargs):
        raise RuntimeError("IK error")
    local = NS(allow_finger_target=False, solve=error)
    with pytest.raises(RuntimeError, match="IK error"):
        grasp_precheck_from_pregrasp(local, file, np.eye(4))
    assert not local.allow_finger_target
