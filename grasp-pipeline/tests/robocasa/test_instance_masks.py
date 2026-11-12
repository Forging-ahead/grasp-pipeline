from types import SimpleNamespace as NS

import numpy as np

from graspgen_libero_bridge.adapters.robocasa.config import load_config
from graspgen_libero_bridge.adapters.robocasa.native import RoboCasaEnvironment


def test_geom_segmentation_maps_complete_object_subtrees_and_excludes_sites(tmp_path, monkeypatch):
    from robosuite import macros
    monkeypatch.setattr(macros, "IMAGE_CONVENTION", "opengl")
    config = load_config("PickPlaceCounterToStove")
    config["policy"].update(camera_size=8, camera_names=["test"])
    e = RoboCasaEnvironment(config, tmp_path)
    model = NS(ngeom=3, nbody=4, body_parentid=np.array([0, 0, 1, 0]), geom_bodyid=np.array([1, 2, 3]))
    raw = np.full((8, 8, 2), -1, dtype=int)
    raw[0, :4] = [[5, 0], [5, 1], [5, 2], [6, 1]]
    e.env = NS(obj_body_id={"target": 1, "other": 3}, sim=NS(model=model, render=lambda **kwargs: raw))
    e.obs = {}
    e._augment_instance_masks()
    mask = e.obs["test_segmentation_instance"][..., 0]
    np.testing.assert_array_equal(mask[0, :4], [2, 2, 4, 0])
    assert e.steps == e.revision == 0


def test_native_opencv_images_receive_matching_mask_row_orientation(tmp_path, monkeypatch):
    from robosuite import macros
    monkeypatch.setattr(macros, "IMAGE_CONVENTION", "opencv")
    config = load_config("PickPlaceCounterToStove")
    config["policy"].update(camera_size=8, camera_names=["test"])
    e = RoboCasaEnvironment(config, tmp_path)
    model = NS(ngeom=1, nbody=2, body_parentid=np.array([0, 0]), geom_bodyid=np.array([1]))
    raw = np.full((8, 8, 2), -1, dtype=int); raw[0, 0] = [5, 0]
    e.env = NS(obj_body_id={"target": 1}, sim=NS(model=model, render=lambda **kwargs: raw))
    e.obs = {}; e._augment_instance_masks()
    mask = e.obs["test_segmentation_instance"][..., 0]
    assert mask[-1, 0] == 2 and mask[0, 0] == 0


def test_one_corrupt_depth_pixel_is_excluded_without_destroying_a_valid_cloud(tmp_path, monkeypatch):
    from robosuite import macros
    import robosuite.utils.camera_utils as cameras
    monkeypatch.setattr(macros, "IMAGE_CONVENTION", "opengl")
    def metric(sim, values):
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
        return values + .1
    monkeypatch.setattr(cameras, "get_real_depth_map", metric)
    monkeypatch.setattr(cameras, "get_camera_intrinsic_matrix", lambda *args: np.array([[10., 0., 4.], [0., 10., 4.], [0., 0., 1.]]))
    monkeypatch.setattr(cameras, "get_camera_extrinsic_matrix", lambda *args: np.eye(4))
    config = load_config("PickPlaceCounterToStove")
    config["policy"].update(camera_size=16, camera_names=["test"])
    e = RoboCasaEnvironment(config, tmp_path)
    e.env = NS(sim=NS())
    e.instance_ids = {"target": 2}
    depth = np.full((16, 16, 1), .5)
    depth[0, 0] = -50.
    e.obs = {"test_depth": depth, "test_segmentation_instance": np.full((16, 16, 1), 2)}
    points, meta = e.target_cloud("target")
    assert len(points) == 255 and meta["invalid_depth_pixels_excluded"] == 1
    assert depth[0, 0, 0] == -50., "Raw recordings must retain the original renderer output"
