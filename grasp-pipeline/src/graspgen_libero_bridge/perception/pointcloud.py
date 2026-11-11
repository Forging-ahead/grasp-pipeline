"""Target point clouds from RGB-D observations or MuJoCo truth geometry.

The two input modes deliberately have separate functions. MuJoCo geometry uses
complete, privileged object information; RGB-D uses only visible depth pixels,
with the supplied instance mask itself recorded as simulator ground truth.
This module imports NumPy only until a robosuite environment is actually used.
"""
from __future__ import annotations

import numpy as np


def sample_points(points, count=2048, seed=0):
    """Deterministically resample finite XYZ points, replacing only if needed."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if count <= 0:
        raise ValueError("count must be positive")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("No finite points")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), int(count), replace=len(points) < count)
    return points[indices].astype(np.float32)


def depth_to_world_points(depth_m, intrinsic, world_from_camera, mask=None,
                          flip_vertical=False, min_depth=0.001, max_depth=10.0):
    """Back-project metric depth using a +X right, +Y down, +Z forward camera.

    ``flip_vertical=True`` converts a bottom-up OpenGL observation to the
    top-down pinhole convention. Depth and mask are flipped together. The
    extrinsic must already be a camera-convention-corrected 4x4 matrix; use
    robosuite's ``get_camera_extrinsic_matrix`` without another axis flip.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError("depth must have shape (H, W) or (H, W, 1)")
    k = np.asarray(intrinsic, dtype=np.float64)
    transform = np.asarray(world_from_camera, dtype=np.float64)
    if k.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("Expected a 3x3 intrinsic and 4x4 camera transform")
    if not np.isfinite(k).all() or not np.isfinite(transform).all():
        raise ValueError("Camera calibration must be finite")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Focal lengths must be positive")
    selected = np.ones(depth.shape, dtype=bool) if mask is None else np.asarray(mask).astype(bool)
    if selected.ndim == 3 and selected.shape[-1] == 1:
        selected = selected[..., 0]
    if selected.shape != depth.shape:
        raise ValueError("Depth and mask dimensions differ")
    if flip_vertical:
        depth = depth[::-1]
        selected = selected[::-1]
    valid = selected & np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    uv1 = np.column_stack([cols, rows, np.ones(len(rows))])
    rays = uv1 @ np.linalg.inv(k).T
    camera = rays * z[:, None]
    return (camera @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def robosuite_target_points(env, observation, camera, target_name, count=2048,
                           seed=0, flip_vertical=None):
    """Visible target points using depth + a simulator-provided instance mask.

    This is an RGB-D + ground-truth-mask input setting, not RGB-only perception.
    The environment must enable ``camera_depths`` and instance segmentation.
    """
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map,
    )
    import robosuite.macros as macros

    depth = np.asarray(observation[camera + "_depth"])
    segmentation = np.asarray(observation[camera + "_segmentation_instance"])
    if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
        segmentation = segmentation[..., 0]
    mapping = getattr(env, "instance_to_id", None)
    if mapping is None or target_name not in mapping:
        # The renderer enumerates model instances from zero and reserves zero
        # in the rendered segmentation for background by adding one.
        base = getattr(env, "env", env)
        mapping = {name: i + 1 for i, name in enumerate(base.model.instances_to_ids)}
    if target_name not in mapping:
        raise KeyError("Target absent from instance mapping: " + str(target_name))
    mask = segmentation == int(mapping[target_name])
    metric_depth = get_real_depth_map(env.sim, depth)
    height, width = np.asarray(metric_depth).shape[:2]
    intrinsic = get_camera_intrinsic_matrix(env.sim, camera, height, width)
    extrinsic = get_camera_extrinsic_matrix(env.sim, camera)
    if flip_vertical is None:
        # robosuite returns raw OpenGL rows for 'opengl' and flips those rows
        # for 'opencv'; pinhole back-projection requires the latter convention.
        convention = str(getattr(macros, "IMAGE_CONVENTION", "opengl")).lower()
        if convention not in ("opengl", "opencv"):
            raise ValueError("Unknown robosuite IMAGE_CONVENTION: " + convention)
        flip_vertical = convention == "opengl"
    points = depth_to_world_points(metric_depth, intrinsic, extrinsic, mask,
                                  flip_vertical=bool(flip_vertical))
    if len(points) < 16:
        raise ValueError("Fewer than 16 valid visible pixels for " + str(target_name))
    return sample_points(points, count, seed), {
        "input_mode": "rgbd_gt_instance_mask", "camera": camera,
        "target": target_name, "visible_pixels": int(len(points)),
        "instance_id": int(mapping[target_name]),
        "flip_vertical": bool(flip_vertical), "coordinate_frame": "world",
        "units": "meters",
    }


def _unwrap_env(env):
    base = env
    while not hasattr(base, "obj_body_id") and hasattr(base, "env"):
        base = base.env
    return base


def object_geom_ids(env, target_name, geometry="visual"):
    """Geometry IDs in the named object's body subtree.

    ``visual`` uses all group-1 visual geoms in robosuite, preserving mixed
    mesh/primitive objects while avoiding duplicate group-0 collision geoms.
    ``collision`` instead selects contact-enabled geometry for diagnostics.
    """
    base = _unwrap_env(env)
    model = base.sim.model
    if target_name not in base.obj_body_id:
        raise KeyError("Unknown MuJoCo object: " + str(target_name))
    root = int(base.obj_body_id[target_name])
    bodies = {root}
    for body in range(model.nbody):
        parent = int(model.body_parentid[body])
        if parent in bodies and body != parent:
            bodies.add(body)
    geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies]
    if not geoms:
        raise ValueError("Object has no geometry: " + str(target_name))
    if geometry == "visual":
        visual = [i for i in geoms if int(model.geom_group[i]) == 1]
        return visual or geoms
    if geometry == "collision":
        return [i for i in geoms if int(model.geom_contype[i]) or int(model.geom_conaffinity[i])]
    if geometry == "all":
        return geoms
    raise ValueError("geometry must be visual, collision, or all")


def _triangles_from_geom(model, data, geom_id):
    """Return world-space surface triangles for a compiled MuJoCo geom."""
    kind = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    if kind == 7:  # mjGEOM_MESH: compiled vertices already include asset scaling.
        mesh = int(model.geom_dataid[geom_id])
        start, length = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fstart, flength = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        vertices = np.asarray(model.mesh_vert[start:start + length], dtype=np.float64)
        faces = np.asarray(model.mesh_face[fstart:fstart + flength], dtype=np.int64)
        if faces.size == 0 or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("Invalid compiled MuJoCo mesh indices")
        triangles = vertices[faces]
    elif kind == 6:  # box with half extents
        vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                             for z in (-1, 1)], dtype=np.float64) * size
        faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                          [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        triangles = vertices[faces]
    else:
        # trimesh is already installed in the model environment and can be
        # installed independently in the LIBERO environment without PyTorch.
        import trimesh
        if kind == 2:
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=size[0])
        elif kind == 3:
            mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1])
            # trimesh's capsule convention can vary across versions; center it
            # from its actual bounds, preserving MuJoCo's z-axis convention.
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
        elif kind == 4:
            mesh = trimesh.creation.icosphere(subdivisions=3)
            mesh.vertices *= size
        elif kind == 5:
            mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=64)
        else:
            raise ValueError("Unsupported target geom type: " + str(kind))
        triangles = np.asarray(mesh.triangles, dtype=np.float64)
    return triangles @ rotation.T + translation


def sample_mujoco_object_points(env, target_name, count=2048, seed=0,
                               geometry="visual", return_metadata=False):
    """Sample complete simulator geometry; this is privileged-input diagnosis.

    MuJoCo's compiled mesh vertices and runtime geom transforms are used, so
    MJCF asset scale/centering are not applied for a second time. Geometry is
    sampled in proportion to triangle area and returned in world coordinates.
    """
    if int(count) <= 0:
        raise ValueError("count must be positive")
    base = _unwrap_env(env)
    ids = object_geom_ids(base, target_name, geometry)
    if not ids:
        raise ValueError("Selected object has no " + geometry + " geoms")
    triangles = np.concatenate([
        _triangles_from_geom(base.sim.model, base.sim.data, i) for i in ids
    ], axis=0)
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    good = np.isfinite(areas) & (areas > 1e-14)
    triangles, areas = triangles[good], areas[good]
    if len(triangles) == 0:
        raise ValueError("Target has zero finite surface area")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(triangles), int(count), p=areas / areas.sum())
    uv = rng.random((int(count), 2))
    outside = uv.sum(axis=1) > 1
    uv[outside] = 1 - uv[outside]
    selected = triangles[indices]
    points = (selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) +
              uv[:, 1:] * (selected[:, 2] - selected[:, 0])).astype(np.float32)
    metadata = {
        "input_mode": "simulator_geometry", "target": target_name,
        "geometry": geometry, "geom_ids": [int(i) for i in ids],
        "surface_area_m2": float(areas.sum()), "points": int(count),
        "coordinate_frame": "world", "units": "meters",
        "privileged_input": True,
    }
    return (points, metadata) if return_metadata else points
