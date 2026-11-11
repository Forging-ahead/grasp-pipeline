"""Geometry and camera-contract tests, independent of CUDA / MuJoCo."""
import unittest
from types import SimpleNamespace

import numpy as np

from graspgen_libero_bridge.pointcloud import (
    depth_to_world_points, object_geom_ids, sample_mujoco_object_points,
    sample_points,
)


class PointCloudContracts(unittest.TestCase):
    def test_camera_backprojection_and_world_frame(self):
        depth = np.array([[2.0, 4.0], [0.0, np.nan]])
        intrinsic = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
        pose = np.array([[0, -1, 0, 1], [1, 0, 0, 2],
                         [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
        points = depth_to_world_points(depth, intrinsic, pose)
        np.testing.assert_allclose(points, [[1, 2, 5], [1, 4, 7]])

    def test_depth_and_instance_mask_flip_together(self):
        depth = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [False, False]])
        result = depth_to_world_points(depth, np.eye(3), np.eye(4),
                                       mask=mask, flip_vertical=True)
        np.testing.assert_allclose(result, [[0, 1, 1]])

    def test_single_row_channel_depth_preserves_image_axes(self):
        depth = np.array([[[1.0], [2.0]]])
        points = depth_to_world_points(depth, np.eye(3), np.eye(4))
        np.testing.assert_allclose(points, [[0, 0, 1], [2, 0, 2]])

    def test_sampling_is_seeded_and_handles_sparse_cloud(self):
        source = np.array([[0, 0, 0], [1, 2, 3], [np.nan, 0, 0]])
        a = sample_points(source, 2048, 17)
        b = sample_points(source, 2048, 17)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (2048, 3))
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(np.count_nonzero(a[:, 0]), 0)
        self.assertLess(np.count_nonzero(a[:, 0]), len(a))

    def _mesh_env(self):
        # A compiled 2m x 1m rectangle, rotated and translated in world. A
        # separate oversized collision box must not contaminate visual points.
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        model = SimpleNamespace(
            nbody=3, body_parentid=np.array([0, 0, 1]), ngeom=2,
            geom_bodyid=np.array([2, 2]), geom_group=np.array([1, 0]),
            geom_type=np.array([7, 6]), geom_contype=np.array([0, 1]),
            geom_conaffinity=np.array([0, 1]), geom_dataid=np.array([0, -1]),
            geom_size=np.array([[1, 1, 1], [10, 10, 10]]),
            mesh_vertadr=np.array([2]), mesh_vertnum=np.array([4]),
            mesh_faceadr=np.array([0]), mesh_facenum=np.array([2]),
            mesh_vert=np.array([[-99, -99, -99], [-99, -99, -99],
                                [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]]),
            mesh_face=np.array([[0, 1, 2], [0, 2, 3]]),
        )
        data = SimpleNamespace(
            geom_xmat=np.array([rotation.ravel(), np.eye(3).ravel()]),
            geom_xpos=np.array([[1, 2, 3], [0, 0, 0]]),
        )
        return SimpleNamespace(sim=SimpleNamespace(model=model, data=data),
                               obj_body_id={"target": 1})

    def test_compiled_mesh_scale_and_transform_applied_once(self):
        env = self._mesh_env()
        points, metadata = sample_mujoco_object_points(
            env, "target", count=20000, seed=4, return_metadata=True)
        self.assertEqual(metadata["geom_ids"], [0])
        self.assertAlmostEqual(metadata["surface_area_m2"], 2.0)
        np.testing.assert_allclose(points[:, 2], 3)
        self.assertGreaterEqual(points[:, 0].min(), 0)
        self.assertLessEqual(points[:, 0].max(), 1)
        self.assertGreaterEqual(points[:, 1].min(), 2)
        self.assertLessEqual(points[:, 1].max(), 4)
        np.testing.assert_allclose(points.mean(axis=0), [0.5, 3, 3], atol=0.02)
        self.assertTrue(metadata["privileged_input"])

    def test_visual_and_collision_geometry_are_separate_modes(self):
        env = self._mesh_env()
        self.assertEqual(object_geom_ids(env, "target", "visual"), [0])
        self.assertEqual(object_geom_ids(env, "target", "collision"), [1])
        points = sample_mujoco_object_points(env, "target", count=2000,
                                            seed=9, geometry="collision")
        np.testing.assert_allclose(np.max(np.abs(points), axis=1), 10, atol=1e-5)

    def test_visual_mesh_and_primitive_both_contribute_surface_points(self):
        env = self._mesh_env()
        model, data = env.sim.model, env.sim.data
        model.ngeom = 3
        model.geom_bodyid = np.array([2, 2, 2])
        model.geom_group = np.array([1, 0, 1])
        model.geom_type = np.array([7, 6, 6])
        model.geom_contype = np.array([0, 1, 0])
        model.geom_conaffinity = np.array([0, 1, 0])
        model.geom_dataid = np.array([0, -1, -1])
        model.geom_size = np.array([[1, 1, 1], [10, 10, 10], [.5, .5, .5]])
        data.geom_xmat = np.vstack([data.geom_xmat, np.eye(3).ravel()])
        data.geom_xpos = np.vstack([data.geom_xpos, [5, 2, 3]])

        points, metadata = sample_mujoco_object_points(
            env, "target", count=20000, seed=18, return_metadata=True)
        self.assertEqual(metadata["geom_ids"], [0, 2])
        self.assertAlmostEqual(metadata["surface_area_m2"], 8.0)
        # The 1m cube contributes 6m² and the visual rectangle 2m². Sampling
        # should retain both parts in that area ratio, never the 10m collision
        # proxy deliberately provided by the fixture.
        on_cube = points[:, 0] > 4
        self.assertAlmostEqual(float(on_cube.mean()), .75, delta=.02)
        self.assertGreater(np.count_nonzero(~on_cube), 0)
        np.testing.assert_allclose(
            np.max(np.abs(points[on_cube] - [5, 2, 3]), axis=1), .5, atol=1e-5)
        self.assertGreaterEqual(points[~on_cube, 0].min(), 0)
        self.assertLessEqual(points[~on_cube, 0].max(), 1)
        self.assertLessEqual(points[:, 0].max(), 5.5)


if __name__ == "__main__":
    unittest.main()
