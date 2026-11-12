from types import SimpleNamespace as NS
import numpy as np

from graspgen_libero_bridge.adapters.robocasa import goals


def test_native_pan_interior_is_used_instead_of_bbox_including_the_handle(monkeypatch):
    names = ["pan_reg_bbox", "pan_reg_int"]
    model = NS(nbody=2, body_parentid=[0, 0], ngeom=2, geom_bodyid=[1, 1], geom_type=[6, 5],
               geom_size=np.array([[.12, .20, .014], [.1071, .0068, 0.]]),
               body_name2id=lambda name: 1, geom_id2name=lambda i: names[i])
    data = NS(geom_xpos=np.array([[0., -.082, .8], [0., 0., .8]]), geom_xmat=np.array([np.eye(3), np.eye(3)]))
    env = NS(env=NS(sim=NS(model=model, data=data), objects={"pan": NS(root_body="pan_root")}))
    monkeypatch.setattr(goals, "_ray_surface", lambda *a: .793)
    regions = goals.v10_placement_regions(env, {"destination": "pan", "destination_kind": "object"})
    assert len(regions) == 1 and regions[0]["name"] == "pan_reg_int"
    np.testing.assert_allclose(regions[0]["frame"][:2, 3], [0, 0])
    np.testing.assert_allclose(regions[0]["half_size"], [.1071, .1071, .0068])


def test_circle_opening_rejects_box_corners_outside_actual_pan_radius():
    region = {"frame": np.eye(4), "half_size": np.array([.1, .1, .01]), "shape": "cylinder"}
    points = np.array([[.08, .08, 0], [-.08, -.08, 0], [.08, -.08, 0]])
    assert not goals.fits_native_opening(np.eye(4), points, region)
    assert goals.fits_native_opening(np.eye(4), points * .5, region)
