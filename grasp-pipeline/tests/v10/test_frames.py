import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from graspgen_libero_bridge.common.frames import pose_error, pose_to_action, grasp_to_control


class FramesTests(unittest.TestCase):
    def test_world_frame_rotation_error(self):
        cur = np.eye(4)
        cur[:3, :3] = Rotation.from_euler('x', .7).as_matrix()
        des = cur.copy()
        r = np.array([.02, -.03, .04])
        des[:3, :3] = Rotation.from_rotvec(r).as_matrix() @ cur[:3, :3]
        np.testing.assert_allclose(pose_error(des, cur)[1], r, atol=1e-10)

    def test_grasp_offset_uses_local_approach_axis(self):
        g = np.eye(4)
        g[:3, :3] = Rotation.from_euler('y', np.pi/2).as_matrix()
        out = grasp_to_control(g)
        np.testing.assert_allclose(out[:3, 3], [.097, 0, 0], atol=1e-10)
        np.testing.assert_allclose(out[:3, :3], g[:3, :3])

    def test_scale_and_saturation(self):
        des = np.eye(4)
        des[:3, 3] = [.1, -.01, 0]
        action = pose_to_action(des, np.eye(4), -1, gain=1)
        np.testing.assert_allclose(action, [1, -.2, 0, 0, 0, 0, -1])


if __name__ == '__main__':
    unittest.main()
