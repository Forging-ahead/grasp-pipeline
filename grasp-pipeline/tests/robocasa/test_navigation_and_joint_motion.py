import numpy as np
import pytest
from types import SimpleNamespace as NS
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.planning.robocasa_navigation import astar_grid
from graspgen_libero_bridge.adapters.robocasa.articulation import joint_motion_pose, native_joints
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure


def test_astar_goes_through_gap_instead_of_through_wall():
    blocked = np.zeros((12, 12), dtype=bool)
    blocked[5, :] = True; blocked[5, 9] = False
    path = astar_grid(blocked, (1, 1), (10, 1))
    assert path[0] == (1, 1) and path[-1] == (10, 1)
    assert (5, 9) in path
    assert all(not blocked[p] for p in path)


def test_astar_does_not_cut_between_two_touching_obstacles():
    blocked = np.array([[False, True], [True, False]])
    assert astar_grid(blocked, (0, 0), (1, 1)) is None


def test_blocked_endpoint_is_not_silently_moved_to_another_goal():
    blocked = np.zeros((4, 4), dtype=bool); blocked[3, 3] = True
    assert astar_grid(blocked, (0, 0), (3, 3)) is None


def test_hinge_pose_rotates_about_measured_anchor_and_preserves_handle_grasp():
    initial = np.eye(4); initial[:3, 3] = [2., 1., 0.]
    result = joint_motion_pose(initial, [0, 0, 1], [1, 1, 0], np.pi / 2, 3)
    np.testing.assert_allclose(result[:3, 3], [1, 2, 0], atol=1e-12)
    np.testing.assert_allclose(result[:3, :3], Rotation.from_euler("z", 90, degrees=True).as_matrix(), atol=1e-12)


def test_slider_uses_world_joint_axis_without_changing_tool_orientation():
    initial = np.eye(4); initial[:3, 3] = [1, 2, 3]
    result = joint_motion_pose(initial, [0, -2, 0], [4, 4, 4], .25, 2)
    np.testing.assert_allclose(result[:3, 3], [1, 1.75, 3])
    np.testing.assert_array_equal(result[:3, :3], initial[:3, :3])


def test_invalid_joint_is_rejected_before_planning():
    with pytest.raises(ValueError, match="axis"):
        joint_motion_pose(np.eye(4), [0, 0, 0], [0, 0, 0], 1, 3)
    with pytest.raises(ValueError, match="hinge/slide"):
        joint_motion_pose(np.eye(4), [0, 0, 1], [0, 0, 0], 1, 0)


def test_fixture_front_uses_measured_rotation_not_native_scalar_yaw_metadata():
    model = NS(joint_name2id=lambda name: 0, jnt_type=np.array([2]),
               jnt_range=np.array([[0., .4]]), jnt_qposadr=np.array([0]), jnt_bodyid=np.array([1]),
               nbody=2, body_parentid=np.array([0, 0]), ngeom=1, geom_bodyid=np.array([1]),
               geom_id2name=lambda geom: "drawer_handle", geom_contype=np.array([1]), geom_conaffinity=np.array([1]))
    data = NS(qpos=np.array([0.]), geom_xpos=np.array([[1., 2., .8]]),
              xaxis=np.array([[1., 0., 0.]]), xanchor=np.array([[1., 2., .8]]))
    fixture = NS(door_joint_names=["drawer_joint"], rot=np.pi / 2)
    measured = np.eye(4); measured[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
    environment = NS(env=NS(drawer=fixture, sim=NS(model=model, data=data)), entity_pose=lambda name: measured)
    spec = native_joints(environment, {"fixture": "drawer", "goal": "open"})[0]
    np.testing.assert_allclose(spec["fixture_front_world"], [1., 0., 0.], atol=1e-12)
    assert spec["goal"] == pytest.approx(.392)


def test_jointless_open_shelf_uses_native_already_open_semantics_without_fabricating_a_joint():
    fixture = NS(door_joint_names=[], is_open=lambda env: True)
    environment = NS(env=NS(cab=fixture, sim=NS(model=NS())))
    assert native_joints(environment, {"fixture": "cab", "goal": "open"}) == []
    with pytest.raises(EpisodeFailure, match="no target joints"):
        native_joints(environment, {"fixture": "cab", "goal": "close"})
