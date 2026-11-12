"""Planar base A* with isolated full-robot MuJoCo path validation."""
from __future__ import annotations

import heapq
import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
from graspgen_libero_bridge.adapters.libero.scene import _geometry, _contact_geom_ids
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
from graspgen_libero_bridge.adapters.robocasa.native import matrix


def astar_grid(blocked, start, goal):
    blocked = np.asarray(blocked, bool)
    start, goal = tuple(start), tuple(goal)
    if blocked.ndim != 2:
        raise ValueError("Expected a 2D occupancy grid")
    def valid(node):
        return all(0 <= node[i] < blocked.shape[i] for i in range(2)) and not blocked[node]
    if not valid(start) or not valid(goal):
        return None
    queue = [(0., 0., start)]
    cost, previous = {start: 0.}, {}
    directions = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
    while queue:
        _, distance, node = heapq.heappop(queue)
        if distance > cost[node] + 1e-12:
            continue
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return path[::-1]
        for dx, dy in directions:
            other = (node[0] + dx, node[1] + dy)
            if not valid(other):
                continue
            if dx and dy and (not valid((node[0] + dx, node[1])) or not valid((node[0], node[1] + dy))):
                continue
            candidate = distance + float(np.hypot(dx, dy))
            if candidate + 1e-12 < cost.get(other, float("inf")):
                cost[other], previous[other] = candidate, node
                heuristic = float(np.hypot(other[0] - goal[0], other[1] - goal[1]))
                heapq.heappush(queue, (candidate + heuristic, candidate, other))
    return None


def mobile_pose(environment, data=None):
    data = environment.env.sim.data if data is None else data
    name = environment.robot.robot_model.base.correct_naming("center")
    site = environment.env.sim.model.site_name2id(name)
    return matrix(data.site_xpos[site], data.site_xmat[site])


def base_grid_path(environment, target_position):
    env = environment.env
    model, data = env.sim.model, env.sim.data
    start = mobile_pose(environment)[:2, 3]
    target = np.asarray(target_position, float)[:2]
    prefix = environment.robot.robot_model.base.naming_prefix
    robot_root = model.body_name2id(environment.robot.robot_model.root_body)
    robot_bodies = body_subtree_ids(model, robot_root)
    mesh_dir = environment.output / "navigation_meshes"; mesh_dir.mkdir(exist_ok=True)
    base_boxes, obstacle_boxes = [], []
    for geom in sorted(_contact_geom_ids(model)):
        geometry, corners = _geometry(model, data, geom, mesh_dir, plane_extent=20.)
        bounds = np.stack((corners.min(0), corners.max(0)))
        body = int(model.geom_bodyid[geom])
        if (model.body_id2name(body) or "").startswith(prefix):
            base_boxes.append(bounds)
        elif body not in robot_bodies and int(model.geom_type[geom]) != 0:
            obstacle_boxes.append(bounds)
    if not base_boxes:
        raise EpisodeFailure("ROBOT_MODEL_MISMATCH", "Cannot derive the native base footprint")
    base_boxes = np.asarray(base_boxes)
    radius = max(float(np.linalg.norm(point[:2] - start)) for box in base_boxes for point in box)
    radius += environment.config["policy"]["base_footprint_margin_m"]
    bottom, top = float(base_boxes[:, 0, 2].min()), float(base_boxes[:, 1, 2].max())
    boxes = [b for b in obstacle_boxes if b[1, 2] > bottom + .02 and b[0, 2] < top + .02]
    lower = np.minimum(start, target) - 2.
    upper = np.maximum(start, target) + 2.
    resolution = environment.config["policy"]["base_grid_resolution_m"]
    shape = np.ceil((upper - lower) / resolution).astype(int) + 1
    if int(np.prod(shape)) > 1_000_000:
        raise EpisodeFailure("NAVIGATION_FAILED", "Navigation grid budget exceeded")
    blocked = np.zeros(shape, dtype=bool)
    for bounds in boxes:
        a = np.floor((bounds[0, :2] - radius - lower) / resolution).astype(int)
        b = np.ceil((bounds[1, :2] + radius - lower) / resolution).astype(int)
        a = np.maximum(a, 0); b = np.minimum(b, shape - 1)
        if np.all(b >= a):
            blocked[a[0]:b[0] + 1, a[1]:b[1] + 1] = True
    start_cell = np.rint((start - lower) / resolution).astype(int)
    goal_cell = np.rint((target - lower) / resolution).astype(int)
    path = astar_grid(blocked, start_cell, goal_cell)
    np.savez_compressed(environment.output / "NAVIGATION_GRID.npz", blocked=blocked, lower=lower,
                        resolution=resolution, start=start, goal=target, footprint_radius_m=radius)
    if path is None:
        raise EpisodeFailure("NAVIGATION_FAILED", "No base-grid route under the measured footprint model",
                             details={"start_blocked": bool(blocked[tuple(start_cell)]),
                                      "goal_blocked": bool(blocked[tuple(goal_cell)]), "radius_m": radius})
    points = lower + np.asarray(path) * resolution
    points[0] = start; points[-1] = target
    return points


def preview_base_path(environment, local, points, target_yaw):
    base_controller = environment.robot.composite_controller.part_controllers["base"]
    qpos = np.asarray(base_controller.qpos_index, int)
    qvel = np.asarray(base_controller.qvel_index, int)
    if len(qpos) != 3 or len(qvel) != 3:
        raise EpisodeFailure("ROBOT_MODEL_MISMATCH", "Expected native three-joint planar base")
    local.sync()
    site_name = environment.robot.robot_model.base.correct_naming("center")
    site = environment.env.sim.model.site_name2id(site_name)
    initial_yaw = Rotation.from_matrix(mobile_pose(environment)[:3, :3]).as_euler("xyz")[2]
    targets = [(point, initial_yaw) for point in points]
    difference = np.arctan2(np.sin(target_yaw - initial_yaw), np.cos(target_yaw - initial_yaw))
    targets += [(points[-1], initial_yaw + u * difference) for u in np.linspace(0., 1., max(2, int(abs(difference) / .07) + 1))]
    jp, jr = np.zeros((3, local.model.nv)), np.zeros((3, local.model.nv))
    arm_q = local.data.qpos[local.qpos_ids].copy()
    for index, (xy, yaw) in enumerate(targets):
        environment.check_budget()
        for _ in range(30):
            current = mobile_pose(environment, local.data)
            current_yaw = Rotation.from_matrix(current[:3, :3]).as_euler("xyz")[2]
            angle = np.arctan2(np.sin(yaw - current_yaw), np.cos(yaw - current_yaw))
            error = np.r_[xy - current[:2, 3], angle]
            if np.linalg.norm(error[:2]) < .002 and abs(angle) < .005:
                break
            local.mj.mj_jacSite(local.model, local.data, jp, jr, site)
            jac = np.stack([jp[0, qvel], jp[1, qvel], jr[2, qvel]])
            delta = np.linalg.solve(jac.T @ jac + 1e-8 * np.eye(3), jac.T @ error)
            local.data.qpos[qpos] += np.clip(delta, -.1, .1)
            local.set_q(arm_q)
        else:
            return {"valid": False, "reason": "BASE_IK_NOT_CONVERGED", "waypoint": index}
        failures = local.contacts()
        if failures:
            return {"valid": False, "reason": "FULL_ROBOT_COLLISION", "waypoint": index, "contacts": failures}
    return {"valid": True, "waypoints": len(targets), "backend": "isolated_full_robot_mujoco_preview"}
