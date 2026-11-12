"""Task goals from live fixture regions or object support geometry, never seed coordinates."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
from graspgen_libero_bridge.adapters.robocasa.native import matrix


def support_geom_ids(environment, stage):
    env = environment.env
    name = stage["destination"]
    entity = env.objects[name] if stage["destination_kind"] == "object" else getattr(env, name)
    root = int(env.sim.model.body_name2id(entity.root_body))
    bodies = body_subtree_ids(env.sim.model, root)
    return {int(g) for g, body in enumerate(env.sim.model.geom_bodyid) if int(body) in bodies}


def _ray_surface(environment, xy, top, allowed):
    import mujoco
    model = getattr(environment.env.sim.model, "_model", environment.env.sim.model)
    data = getattr(environment.env.sim.data, "_data", environment.env.sim.data)
    point = np.array([*xy, top], dtype=float)
    geom = np.array([-1], dtype=np.int32)
    for _ in range(30):
        distance = mujoco.mj_ray(model, data, point, np.array([0., 0., -1.]), None, 1, -1, geom)
        if distance < 0:
            return None
        point[2] -= distance
        if int(geom[0]) in allowed:
            return float(point[2])
        point[2] -= 1e-4
    return None


def placement_goals(environment, stage, object_points, initial_orientation):
    env = environment.env
    destination = stage["destination"]
    points = np.asarray(object_points, float)
    supports = support_geom_ids(environment, stage)
    regions = []
    if stage["destination_kind"] == "fixture":
        fixture = getattr(env, destination)
        for name, (p0, px, py, pz) in fixture.get_int_sites(relative=False).items():
            p0, px, py, pz = map(lambda p: np.asarray(p, float), (p0, px, py, pz))
            u, v, w = px - p0, py - p0, pz - p0
            if np.linalg.norm(u) <= 0 or np.linalg.norm(v) <= 0:
                continue
            regions.append({"name": name, "center": p0 + .5 * (u + v),
                            "u": u, "v": v, "height": float(np.linalg.norm(w)),
                            "ray_top": float(p0[2] + (.5 * w[2] if w[2] > .03 else .03))})
    else:
        entity = env.objects[destination]
        pose = environment.object_pose(destination)
        size = np.asarray(entity.size, float)
        regions.append({"name": "object_support", "center": pose[:3, 3],
                        "u": pose[:3, 0] * size[0], "v": pose[:3, 1] * size[1],
                        "height": float(size[2]), "ray_top": float(pose[2, 3] + size[2] + .10)})
    if not regions:
        raise EpisodeFailure("UNSUPPORTED_TASK_BINDING", "Destination exposes no placement region: " + destination)
    for region in regions:
        for a, b in [(0., 0.), (.2, 0.), (-.2, 0.), (0., -.2), (0., .2)]:
            center = region["center"] + a * region["u"] + b * region["v"]
            surface = _ray_surface(environment, center[:2], region["ray_top"], supports)
            if surface is None:
                continue
            for yaw in (0., 45., -45., 90., -90., 180.):
                rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ initial_orientation
                rotated = points @ rotation.T
                goal = np.eye(4)
                goal[:3, :3] = rotation
                goal[:2, 3] = center[:2]
                goal[2, 3] = surface - rotated[:, 2].min() + environment.config["policy"]["release_clearance_m"]
                # A fixture's horizontal region must contain the full footprint.
                if stage["destination_kind"] == "fixture":
                    local = rotated + goal[:3, 3] - region["center"]
                    u, v = region["u"], region["v"]
                    if np.max(np.abs(local @ u / np.dot(u, u))) > .48 or np.max(np.abs(local @ v / np.dot(v, v))) > .48:
                        continue
                    if region["height"] > .03 and np.max(local[:, 2]) > region["height"] - .005:
                        continue
                yield {"world_T_object": goal, "support_geom_ids": sorted(supports),
                       "region": region["name"], "surface_height_m": surface,
                       "yaw_degrees": yaw, "offset_uv": [a, b],
                       "source": "native_region_and_compiled_geometry_vertical_ray"}


def check_placement(environment, stage):
    import robocasa.utils.object_utils as OU
    env = environment.env
    target, destination = stage["target"], stage["destination"]
    if stage["destination_kind"] == "object":
        return bool(OU.check_obj_in_receptacle(env, target, destination))
    fixture = getattr(env, destination)
    # Use native helpers for containment versus support, rather than a distance-only success.
    if "cabinet" in type(fixture).__name__.lower() or "drawer" in type(fixture).__name__.lower():
        return bool(OU.obj_inside_of(env, target, fixture))
    return bool(OU.check_obj_fixture_contact(env, target, fixture))


def v10_placement_regions(environment, stage):
    """Map native *interior* regions to v10 gravity-aligned opening boxes.

    An object's bbox can include a pan handle and need not be centered on its
    opening. Compiled reg_int geometry, when present, is the native region.
    """
    from graspgen_libero_bridge.adapters.libero.placement_geometry import gravity_frame
    env = environment.env
    regions = []
    if stage["destination_kind"] == "fixture":
        fixture = getattr(env, stage["destination"])
        for name, corners in fixture.get_int_sites(relative=False).items():
            p0, px, py, pz = (np.asarray(p, float) for p in corners)
            edges = np.stack([px - p0, py - p0, pz - p0], axis=1)
            lengths = np.linalg.norm(edges, axis=0)
            if np.any(lengths <= 1e-6):
                continue
            frame = np.eye(4); frame[:3, :3] = edges / lengths
            frame[:3, 3] = p0 + edges.sum(axis=1) / 2
            if np.linalg.det(frame[:3, :3]) < 0:
                frame[:3, 1] *= -1
            frame, size = gravity_frame(frame, lengths / 2)
            regions.append({"name": name, "frame": frame, "half_size": size, "shape": "box",
                            "container": "cabinet" in type(fixture).__name__.lower() or "drawer" in type(fixture).__name__.lower(),
                            "source": "native_fixture_get_int_sites"})
    else:
        entity = env.objects[stage["destination"]]
        model, data = env.sim.model, env.sim.data
        root = model.body_name2id(entity.root_body)
        bodies = body_subtree_ids(model, root)
        ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
               and "reg_int" in (model.geom_id2name(g) or "")]
        if not ids:
            ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
                   and (model.geom_id2name(g) or "").endswith("reg_bbox")]
        for geom in ids:
            kind = int(model.geom_type[geom])
            if kind not in {5, 6}:
                continue
            size = np.asarray(model.geom_size[geom], float).copy()
            if kind == 5:
                size = np.array([size[0], size[0], size[1]])
            frame, size = gravity_frame(matrix(data.geom_xpos[geom], data.geom_xmat[geom]), size)
            is_container = "reg_int" in model.geom_id2name(geom)
            regions.append({"name": model.geom_id2name(geom), "frame": frame, "half_size": size,
                            "shape": "cylinder" if kind == 5 else "box", "container": is_container,
                            "source": "native_compiled_reg_int" if is_container else "native_bbox_support_surface"})
    supports = support_geom_ids(environment, stage)
    checked = []
    for region in regions:
        frame, size = region["frame"], region["half_size"]
        top = frame[2, 3] + (0. if region["container"] else size[2]) + .03
        surface = _ray_surface(environment, frame[:2, 3], top, supports)
        if surface is not None:
            region.update(surface_height_m=surface, support_geom_ids=sorted(supports))
            checked.append(region)
    if not checked:
        raise EpisodeFailure("UNSUPPORTED_TASK_BINDING", "No native placement region with a physical support")
    return checked


def fits_native_opening(pose, points, region):
    from graspgen_libero_bridge.adapters.libero.placement_geometry import fits_opening
    frame, size = region["frame"], region["half_size"]
    if not fits_opening(pose, points, frame, size):
        return False
    if region["shape"] == "cylinder":
        world = np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]
        local = (world - frame[:3, 3]) @ frame[:3, :3]
        gravity = frame[:3, :3].T @ np.array([0., 0., -1.])
        return all(bool(np.all(np.linalg.norm((local + ((level - local[:, 2]) / gravity[2])[:, None] * gravity)[:, :2],
                                              axis=1) <= size[0] - .002)) for level in (-size[2], size[2]))
    return True
