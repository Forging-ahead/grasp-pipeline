"""Export the current compiled MuJoCo collision world for cuRobo.

Each mesh geom is exported as the convex hull of its *compiled* vertices. This
matches MuJoCo's per-mesh convex collision semantics without applying asset
scale or mesh-centering transforms twice. Separate geoms remain separate parts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .env_adapter import body_subtree_ids


def _matrix(position, rotation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    result[:3, 3] = np.asarray(position, dtype=float)
    return result


def _pose_list(matrix):
    from scipy.spatial.transform import Rotation
    t = np.asarray(matrix, dtype=float)
    q = Rotation.from_matrix(t[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return np.r_[t[:3, 3], q].tolist()


def _corners(bounds):
    low, high = np.asarray(bounds, dtype=float)
    return np.asarray([[x, y, z] for x in (low[0], high[0])
                       for y in (low[1], high[1]) for z in (low[2], high[2])])


def _export_hull(vertices, directory, mesh_id):
    from scipy.spatial import ConvexHull
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid compiled mesh vertices")
    digest = hashlib.sha256(vertices.astype("<f8").tobytes()).hexdigest()
    path = directory / f"mesh_{mesh_id}_{digest[:20]}.obj"
    if not path.exists():
        hull = ConvexHull(vertices)
        if hull.volume <= 1e-15:
            raise ValueError("Collision mesh has no three-dimensional convex hull")
        used = np.asarray(hull.vertices, dtype=np.int64)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        faces = np.asarray(hull.simplices, dtype=np.int64).copy()
        for face, plane in zip(faces, hull.equations):
            a, b, c = vertices[face]
            if np.dot(np.cross(b - a, c - a), plane[:3]) < 0:
                face[1], face[2] = face[2], face[1]
        text = ["# MuJoCo compiled mesh; per-geom convex collision hull; meters"]
        text.extend("v %.17g %.17g %.17g" % tuple(v) for v in vertices[used])
        text.extend("f %d %d %d" % tuple(f + 1) for f in remap[faces])
        temporary = path.with_suffix(".tmp")
        temporary.write_text("\n".join(text) + "\n")
        temporary.replace(path)
    return path, digest


def _contact_geom_ids(model):
    result = {i for i in range(int(model.ngeom))
              if int(model.geom_contype[i]) or int(model.geom_conaffinity[i])}
    # Explicit contact pairs can override the ordinary bit-mask filtering.
    for name in ("pair_geom1", "pair_geom2"):
        values = getattr(model, name, None)
        if values is not None:
            result.update(map(int, np.asarray(values).reshape(-1)))
    return result


def _geometry(model, data, geom_id, mesh_dir, plane_extent):
    kind = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    pose = _matrix(data.geom_xpos[geom_id], data.geom_xmat[geom_id])
    name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
    result = dict(name=str(name), geom_id=int(geom_id), body_id=int(model.geom_bodyid[geom_id]),
                  pose=_pose_list(pose), source_mujoco_type=kind)
    if kind == 7:  # mesh; compiled vertices include original MJCF asset scaling.
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start:start + count], dtype=float)
        path, digest = _export_hull(vertices, mesh_dir, mesh_id)
        result.update(type="mesh", mesh_path=str(path.resolve()), scale=[1., 1., 1.],
                      source_mesh_id=mesh_id, compiled_vertices_sha256=digest,
                      representation="convex_hull_of_compiled_mesh")
        bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    elif kind == 6:
        result.update(type="box", dims=(2 * size).tolist())
        bounds = np.stack((-size, size))
    elif kind == 5:
        result.update(type="cylinder", radius=float(size[0]), length=float(2 * size[1]))
        extent = np.array([size[0], size[0], size[1]])
        bounds = np.stack((-extent, extent))
    elif kind == 2:
        result.update(type="sphere", radius=float(size[0]))
        bounds = np.stack((-np.repeat(size[0], 3), np.repeat(size[0], 3)))
    elif kind == 3:
        extent = np.array([size[0], size[0], size[1] + size[0]])
        result.update(type="box", dims=(2 * extent).tolist(),
                      representation="conservative_capsule_enclosing_box")
        bounds = np.stack((-extent, extent))
    elif kind == 4:
        result.update(type="box", dims=(2 * size).tolist(),
                      representation="conservative_ellipsoid_enclosing_box")
        bounds = np.stack((-size, size))
    elif kind == 0:
        # Infinite collision planes need a finite planning representation. The
        # proxy covers a region far larger than the Panda's entire workspace.
        thickness = .10
        extent = max(float(plane_extent), float(np.max(size[:2])), 3.)
        pose = pose.copy()
        pose[:3, 3] -= pose[:3, 2] * thickness / 2
        dims = np.array([2 * extent, 2 * extent, thickness])
        result.update(type="box", dims=dims.tolist(), pose=_pose_list(pose),
                      representation="finite_workspace_plane_proxy", plane_half_extent_m=extent)
        bounds = np.stack((-dims / 2, dims / 2))
    elif kind == 1:
        field_id = int(model.geom_dataid[geom_id])
        extent = np.asarray(model.hfield_size[field_id], dtype=float)
        dims = np.array([2 * extent[0], 2 * extent[1], extent[2] + extent[3]])
        pose = pose.copy()
        pose[:3, 3] += pose[:3, 2] * (extent[2] - extent[3]) / 2
        result.update(type="box", dims=dims.tolist(), pose=_pose_list(pose),
                      representation="conservative_hfield_enclosing_box")
        bounds = np.stack((-dims / 2, dims / 2))
    else:
        raise ValueError(f"Unsupported MuJoCo collision geom {name!r}, type {kind}; geometry was not omitted")
    if not np.isfinite(bounds).all() or (bounds[1] <= bounds[0]).any():
        raise ValueError("Invalid collision geometry extent: " + str(name))
    result["bounds_local"] = bounds.tolist()
    corners_world = _corners(bounds) @ pose[:3, :3].T + pose[:3, 3]
    return result, corners_world


def export_scene(adapter, run_dir):
    """Return current state plus every nonrobot contact geom, target separately."""
    state = adapter.state()
    model, data = adapter.env.sim.model, adapter.env.sim.data
    run_dir = Path(run_dir).resolve()
    mesh_dir = run_dir / "collision_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    root_id = int(model.body_name2id(adapter.robot.robot_model.root_body))
    robot_bodies = body_subtree_ids(model, root_id)
    target_name = str(adapter.config.get("target_object", adapter.target_object))
    if target_name not in adapter.task_env.obj_body_id:
        raise KeyError("Target absent from task: " + target_name)
    target_root = int(adapter.task_env.obj_body_id[target_name])
    target_bodies = body_subtree_ids(model, target_root)
    target_pose = np.asarray(adapter.target_pose(target_name), dtype=float)
    target_inverse = np.linalg.inv(target_pose)
    obstacles, target_geometries, target_corners = [], [], []
    omitted_robot = []
    contact_ids = _contact_geom_ids(model)
    for geom_id in sorted(contact_ids):
        body = int(model.geom_bodyid[geom_id])
        if body in robot_bodies:
            omitted_robot.append(int(geom_id))
            continue
        geometry, corners = _geometry(
            model, data, geom_id, mesh_dir,
            plane_extent=float(adapter.config.get("planning_plane_extent_m", 20.)))
        if body in target_bodies:
            target_geometries.append(geometry)
            target_corners.append(corners)
        else:
            obstacles.append(geometry)
    if not target_geometries:
        raise ValueError("Target has no collision-enabled geometry: " + target_name)
    world_points = np.concatenate(target_corners)
    object_points = world_points @ target_inverse[:3, :3].T + target_inverse[:3, 3]
    target = dict(
        name=target_name, body_id=target_root, world_T_object=target_pose.tolist(),
        geometries=target_geometries,
        bounds=np.stack((object_points.min(axis=0), object_points.max(axis=0))).tolist(),
        points=object_points.tolist(), points_frame="object", points_kind="collision_AABB_corners",
        bounds_world=np.stack((world_points.min(axis=0), world_points.max(axis=0))).tolist(),
    )
    scene = dict(state)
    scene.update(
        obstacles=obstacles, target=target, merged_model_xml=str(adapter.merged_xml_path),
        collision_export=dict(
            contact_geom_count=len(contact_ids), robot_geom_ids=omitted_robot,
            exported_obstacle_count=len(obstacles), exported_target_count=len(target_geometries),
            mesh_semantics="per-geom convex hull of compiled MuJoCo vertices; asset scaling applied once",
            robot_base_body=adapter.robot.robot_model.root_body,
            robot_body_ids=sorted(robot_bodies),
            conservative_approximations=[dict(name=g["name"], representation=g["representation"])
                                         for g in obstacles + target_geometries
                                         if g.get("representation", "").startswith(("conservative_", "finite_workspace_"))],
        ),
    )
    return scene
