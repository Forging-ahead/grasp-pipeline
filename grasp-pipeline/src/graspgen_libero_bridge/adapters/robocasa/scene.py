"""Compiled MuJoCo geometry with the mobile platform retained as obstacles."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
from graspgen_libero_bridge.adapters.libero.scene import _geometry, _contact_geom_ids


def export_scene(environment, target, carried_contents=()):
    env = environment.env
    model, data = env.sim.model, env.sim.data
    state = environment.robot_state()
    arm_root = int(model.body_name2id(state["arm_base_body"]))
    arm_bodies = body_subtree_ids(model, arm_root)
    target_root = environment.entity_body(target)
    target_bodies = body_subtree_ids(model, target_root)
    for name in carried_contents:
        target_bodies |= body_subtree_ids(model, environment.entity_body(name))
    pose = environment.entity_pose(target)
    inverse = np.linalg.inv(pose)
    directory = environment.output / "collision_meshes"
    directory.mkdir(exist_ok=True)
    obstacles, target_geometries, corners = [], [], []
    for geom in sorted(_contact_geom_ids(model)):
        body = int(model.geom_bodyid[geom])
        if body in arm_bodies:
            continue
        geometry, points = _geometry(model, data, geom, directory, plane_extent=20.)
        if body in target_bodies:
            target_geometries.append(geometry)
            corners.append(points)
        else:
            obstacles.append(geometry)
    if not corners:
        raise ValueError("Target has no collision-enabled geometry: " + target)
    world_points = np.concatenate(corners)
    object_points = world_points @ inverse[:3, :3].T + inverse[:3, 3]
    state.update(episode_id=environment.episode_id, revision=environment.revision,
                 obstacles=obstacles, target={"name": target, "body_id": target_root,
                 "world_T_object": pose.tolist(), "geometries": target_geometries,
                 "bounds": np.stack((object_points.min(0), object_points.max(0))).tolist(),
                 "bounds_world": np.stack((world_points.min(0), world_points.max(0))).tolist(),
                 "points": object_points.tolist(), "points_frame": "object",
                 "points_kind": "collision_AABB_corners_not_grasp_input"},
                 collision_export={"source": "compiled_mujoco_contact_geometry",
                 "mobile_platform_and_torso_retained_as_obstacles": True,
                 "arm_base_body": state["arm_base_body"],
                 "arm_body_ids": sorted(arm_bodies), "omitted_target": False})
    state["target"]["carried_contents"] = list(carried_contents)
    return state
