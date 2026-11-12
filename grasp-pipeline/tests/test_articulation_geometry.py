"""Regression cases for door orbit inference and cylindrical-handle symmetry."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from graspgen_libero_bridge.adapters.libero.articulation import (
    ArticulationPreviewMonitor, constrained_pose, grip_candidates, handle_motion_pose)


def info():
    return dict(joint_type=3, axis=np.array([0., 0., 1.]), anchor=np.zeros(3),
        handle_axis=np.array([0., 0., 1.]), cylindrical_handle=True,
        handle_center=np.array([.2, 0., 0.]), handle_rotation=np.eye(3),
        handle_half_sizes=np.array([.01, .01, .075]), q_address=0)


class ArticulationGeometryTests(unittest.TestCase):
    def test_round_vertical_handle_can_follow_door_without_equal_wrist_rotation(self):
        metadata = info()
        reference = np.eye(4)
        reference[0, 3] = .2
        desired = handle_motion_pose(reference, metadata, -.7, True)
        np.testing.assert_allclose(desired[:3, 3], [.2*np.cos(.7), -.2*np.sin(.7), 0.])
        np.testing.assert_allclose(desired[:3, :3], np.eye(3), atol=1e-12)
        rigid = handle_motion_pose(reference, metadata, -.7, False)
        self.assertGreater(Rotation.from_matrix(rigid[:3, :3]).magnitude(), .69)

    def test_knob_cannot_use_circular_bar_freedom(self):
        metadata = dict(info(), rotary_knob=True)
        reference = np.eye(4)
        expected = constrained_pose(reference, metadata['axis'], metadata['anchor'], .6, 3)
        np.testing.assert_allclose(handle_motion_pose(reference, metadata, .6, True), expected)

    def test_orbit_inference_does_not_mistake_fixed_wrist_for_stationary_door(self):
        metadata = info()
        reference = np.eye(4)
        reference[0, 3] = .2
        delta = -.5
        data = SimpleNamespace(qpos=np.zeros(1), site_xpos=np.array([[.2*np.cos(delta), .2*np.sin(delta), 0.]]),
            site_xmat=np.eye(3).reshape(1, 9))
        sim = SimpleNamespace(data=data, forward=lambda: None)
        adapter = SimpleNamespace(env=SimpleNamespace(sim=sim),
            task_env=SimpleNamespace(robots=[SimpleNamespace(eef_site_id=0)]))
        measured = []
        monitor = SimpleNamespace(check=lambda contact: measured.append(float(data.qpos[0])) or [])
        coupled = ArticulationPreviewMonitor(adapter, monitor, metadata, reference, 0., -1.)
        coupled.check(True)
        self.assertAlmostEqual(measured[0], delta)

    def test_original_candidates_are_preserved_and_offsets_remain_on_bar(self):
        metadata = info()
        original = list(grip_candidates(metadata, -1., 0.))
        self.assertEqual(len(original), 6)
        for _, grip in original:
            np.testing.assert_allclose(grip[:3, 3], metadata['handle_center'])
        shifted = list(grip_candidates(metadata, -1., 0., axial_fractions=(.5,-.5)))
        self.assertEqual(len(shifted), 12)
        for pre, grip in shifted:
            np.testing.assert_allclose(grip[:2, 3], metadata['handle_center'][:2])
            self.assertAlmostEqual(abs(grip[2, 3]), .0375)
            self.assertAlmostEqual(np.linalg.norm(pre[:3, 3]-grip[:3, 3]), .07)
            np.testing.assert_allclose(grip[:3, :3].T@grip[:3, :3], np.eye(3), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
