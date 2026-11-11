"""Release-pose search for open containers; does not waive robot collisions."""
import numpy as np


def can_retrace_after_ik_failure(reply):
    return reply.get("state_restored") is True and reply.get("reason") in {
        "cartesian_ik_iteration_limit", "cartesian_ik_collision_blocked",
        "cartesian_waypoint_ik_failed", "cartesian_ik_stalled"}


def container_release_candidates(object_goal, eef_T_object, points_object,
                                 region_pose, region_size, offsets, clearance=.002):
    object_goal = np.asarray(object_goal, dtype=float)
    eef_T_object = np.asarray(eef_T_object, dtype=float)
    points = np.asarray(points_object, dtype=float)
    region = np.asarray(region_pose, dtype=float)
    size = np.asarray(region_size, dtype=float)
    for matrix in (object_goal, eef_T_object, region):
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("Invalid pose matrix")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
            raise ValueError("Invalid pose rotation")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Need object surface points")
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Invalid container region size")
    heights = sorted(set(float(v) for v in offsets))
    if not heights or not np.isfinite(heights).all() or heights[0] < 0 or heights[-1] > .12:
        raise ValueError("Release offsets must be in [0, .12] meters")
    if not 0 <= clearance < float(np.min(size[:2])):
        raise ValueError("Invalid opening clearance")
    # Resting simulator assets may lean slightly. Check the entire vertical
    # gravity corridor through the opening and floor in the actual site frame.
    gravity_local = region[:3, :3].T @ np.array([0., 0., -1.])
    if -gravity_local[2] < np.cos(np.deg2rad(10.)):
        raise ValueError("Tilted containers require a different placement policy")
    candidates = []
    for height in heights:
        pose = object_goal.copy()
        pose[2, 3] += height
        world = points @ pose[:3, :3].T + pose[:3, 3]
        local = (world-region[:3, 3]) @ region[:3, :3]
        fits = True
        for plane in (-size[2], size[2]):
            t = (plane-local[:, 2]) / gravity_local[2]
            crossing = local + t[:, None]*gravity_local
            fits = fits and bool(np.all(np.abs(crossing[:, :2]) <= size[:2]-clearance))
        if not fits:
            continue
        candidates.append(dict(offset_m=height, object_pose=pose,
                               eef_pose=pose @ np.linalg.inv(eef_T_object),
                               opening_footprint_clear=True))
    return candidates
