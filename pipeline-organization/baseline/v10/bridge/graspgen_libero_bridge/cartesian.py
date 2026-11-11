"""MuJoCo-checked Cartesian contact-phase plans for the actual joint branch.

This planner performs static IK previews only. It never steps the environment,
changes a controller goal, or executes a grasped object's transport. A held
free-joint object follows its measured grip-site relation in the preview; all
integrator inputs are restored before returning. Execution remains OSC control.
Collision checks use the supplied monitor and its current contact permissions.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
import traceback
from contextlib import contextmanager

import numpy as np


def _site_pose(data, site):
    result = np.eye(4)
    result[:3, :3] = np.asarray(data.site_xmat[site]).reshape(3, 3)
    result[:3, 3] = np.asarray(data.site_xpos[site])
    return result


def _body_pose(data, body):
    result = np.eye(4)
    result[:3, :3] = np.asarray(data.body_xmat[body]).reshape(3, 3)
    result[:3, 3] = np.asarray(data.body_xpos[body])
    return result


def _snapshot(sim):
    data = getattr(sim.data, "_data", sim.data)
    arrays = {}
    for name in ("qpos", "qvel", "act", "ctrl", "qacc", "qacc_warmstart",
                 "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat",
                 "userdata", "eq_active", "plugin_state"):
        value = getattr(data, name, None)
        if value is not None:
            arrays[name] = np.array(value, copy=True)
    return copy.deepcopy(sim.get_state()), arrays, float(data.time)


def _restore(sim, snapshot):
    state, arrays, original_time = snapshot
    data = getattr(sim.data, "_data", sim.data)
    sim.set_state(state)
    for name, value in arrays.items():
        getattr(data, name)[...] = value
    sim.forward()
    # Restore integration warm-start / force inputs even if mj_forward updated
    # derived accelerations. Body/site transforms already match restored qpos.
    for name, value in arrays.items():
        getattr(data, name)[...] = value
    if float(data.time) != original_time:
        raise RuntimeError("Cartesian preview failed to restore simulation time")
    for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
        if name in arrays and not np.array_equal(np.asarray(getattr(data, name)), arrays[name]):
            raise RuntimeError("Cartesian preview failed to restore " + name)


@contextmanager
def preview_path_endpoint(adapter, monitor, path_file):
    """Temporarily inspect a planned endpoint, including the held free body.

    No dynamics, controller goal, recorder, or contact permissions are changed.
    Nested Cartesian plans can therefore check a continuation before either
    segment is physically executed. Every integrator input is restored on exit.
    """
    from scipy.spatial.transform import Rotation
    sim = adapter.env.sim
    model, data = sim.model, sim.data
    robot = adapter.task_env.robots[0]
    indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
    with np.load(path_file, allow_pickle=False) as path:
        endpoint = np.asarray(path["q_arm"][-1], dtype=float)
    if indexes.shape != (7,) or endpoint.shape != (7,) or not np.isfinite(endpoint).all():
        raise ValueError("Endpoint preview requires seven finite arm joints")
    snapshot = _snapshot(sim)
    try:
        address = relation = None
        if getattr(monitor, "held", False):
            body = int(adapter.task_env.obj_body_id[monitor.target])
            while True:
                first, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
                joints = range(first, first + count) if first >= 0 else ()
                free = next((j for j in joints if int(model.jnt_type[j]) == 0), None)
                if free is not None:
                    address = int(model.jnt_qposadr[free])
                    relation = np.linalg.inv(_site_pose(data, robot.eef_site_id)) @ _body_pose(data, body)
                    break
                parent = int(model.body_parentid[body])
                if body == 0 or parent == body:
                    raise ValueError("Held target does not have a free-joint ancestor")
                body = parent
        data.qpos[indexes] = endpoint
        sim.forward()
        if address is not None:
            pose = _site_pose(data, robot.eef_site_id) @ relation
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            data.qpos[address:address + 7] = np.r_[pose[:3, 3], quat]
            sim.forward()
        yield
    finally:
        _restore(sim, snapshot)


def plan_cartesian(adapter, monitor, goal_pose, output_dir,
                   allow_finger_target=True, max_linear_step=.005):
    """Plan a straight/SLERP site motion with actual-MuJoCo collision checks.

    Returns ``{success, path_file?, reason?, ...}``. On success, ``PATH.npz``
    contains ``q_arm[N,7]`` and ``world_T_eef[N,4,4]`` including the measured
    start. Every stored state and intermediate IK joint segment is checked by
    ``monitor.check``. This is sampled contact-aware validation, not a continuous
    collision certificate. No supplied monitor permission is broadened here.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(success=False, planner="mujoco_cartesian_dls",
                  state_restored=False, collision_checks=0, accepted_ik_steps=0,
                  allow_finger_target=bool(allow_finger_target),
                  path_validation="sampled actual MuJoCo contacts; existing monitor tolerance and phase permissions",
                  limitation="Kinematic contact-phase preview; actual execution must use closed-loop control and contact monitoring")
    snapshot = None
    path_q, path_pose, path_target = [], [], []
    try:
        import mujoco
        from scipy.spatial.transform import Rotation

        step = float(max_linear_step)
        if not np.isfinite(step) or not .0001 <= step <= .02:
            raise ValueError("max_linear_step must be between 0.0001 and 0.02 meters")
        goal = np.asarray(goal_pose, dtype=float)
        if goal.shape != (4, 4) or not np.isfinite(goal).all():
            raise ValueError("goal_pose must be a finite 4x4 transform")
        if not np.allclose(goal[3], [0., 0., 0., 1.], atol=1e-8):
            raise ValueError("Invalid goal homogeneous row")
        if not np.allclose(goal[:3, :3].T @ goal[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(goal[:3, :3]) < .99999:
            raise ValueError("Invalid goal rotation")
        sim = adapter.env.sim
        robot = adapter.task_env.robots[0]
        model, data = sim.model, sim.data
        native_model = getattr(model, "_model", model)
        native_data = getattr(data, "_data", data)
        q_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        dof_indexes = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
        if q_indexes.shape != (7,) or dof_indexes.shape != (7,):
            raise ValueError("Cartesian contact planner requires one seven-joint Panda")
        site = int(robot.eef_site_id)
        names = list(robot.robot_model.joints)
        joint_ids = [int(model.joint_name2id(name)) for name in names]
        limits = np.asarray([model.jnt_range[j] for j in joint_ids], dtype=float)
        if limits.shape != (7, 2):
            raise ValueError("Missing Panda joint limits")
        lower, upper = limits[:, 0] + 1e-6, limits[:, 1] - 1e-6
        snapshot = _snapshot(sim)
        initial_q = np.array(data.qpos[q_indexes], dtype=float, copy=True)
        start = _site_pose(data, site)
        delta_rotation = Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).as_rotvec()
        distance = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
        angle = float(np.linalg.norm(delta_rotation))
        waypoints = max(1, int(np.ceil(distance / step)), int(np.ceil(angle / .03)))
        if waypoints > 600:
            raise ValueError("Cartesian contact-phase request exceeds 600 small waypoints")
        result.update(linear_distance_m=distance, rotation_distance_rad=angle,
                      linear_waypoints=waypoints, max_linear_step_m=step,
                      held=bool(getattr(monitor, "held", False)),
                      allowed_support_geoms=sorted(getattr(monitor, "allowed_support_geoms", ())),
                      joint_names=names, position_tolerance_m=.0004,
                      rotation_tolerance_rad=.003)

        held = bool(getattr(monitor, "held", False))
        free_address = None
        eef_to_free_body = None
        target_name = getattr(monitor, "target", adapter.target_object)
        if held:
            body = int(adapter.task_env.obj_body_id[target_name])
            current = body
            while True:
                first = int(model.body_jntadr[current])
                count = int(model.body_jntnum[current])
                joints = range(first, first + count) if first >= 0 else ()
                free_joint = next((j for j in joints if int(model.jnt_type[j]) == 0), None)
                if free_joint is not None:
                    free_address = int(model.jnt_qposadr[free_joint])
                    eef_to_free_body = np.linalg.inv(start) @ _body_pose(data, current)
                    break
                parent = int(model.body_parentid[current])
                if parent == current or current == 0:
                    raise ValueError("Held target does not have a free-joint ancestor")
                current = parent
            result["held_target"] = target_name
            result["eef_T_target"] = (np.linalg.inv(start) @ adapter.target_pose(target_name)).tolist()

        def preview(q):
            data.qpos[q_indexes] = q
            sim.forward()
            pose = _site_pose(data, site)
            target_pose = None
            if held:
                free_pose = pose @ eef_to_free_body
                quat = Rotation.from_matrix(free_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
                data.qpos[free_address:free_address + 7] = np.r_[free_pose[:3, 3], quat]
                sim.forward()
                target_pose = adapter.target_pose(target_name).copy()
            failures = monitor.check(allow_finger_target=bool(allow_finger_target))
            result["collision_checks"] += 1
            return pose, target_pose, failures

        def error(desired, actual):
            return np.r_[desired[:3, 3] - actual[:3, 3],
                         Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).as_rotvec()]

        weights = np.array([1., 1., 1., .35, .35, .35])
        q = initial_q.copy()
        actual, target_actual, failures = preview(q)
        if failures:
            result.update(reason="cartesian_start_collision", collisions=failures)
        elif (q < limits[:, 0] - 1e-5).any() or (q > limits[:, 1] + 1e-5).any():
            result.update(reason="cartesian_start_outside_joint_limits")
        else:
            path_q.append(q.copy())
            path_pose.append(actual.copy())
            if held:
                path_target.append(target_actual.copy())
            failed = False
            for waypoint_id in range(1, waypoints + 1):
                u = waypoint_id / waypoints
                desired = np.eye(4)
                desired[:3, 3] = start[:3, 3] * (1 - u) + goal[:3, 3] * u
                desired[:3, :3] = Rotation.from_rotvec(delta_rotation * u).as_matrix() @ start[:3, :3]
                reference_q = q.copy()
                converged = False
                for iteration in range(80):
                    actual, _, failures = preview(q)
                    if failures:
                        result.update(reason="cartesian_collision", collisions=failures,
                                      failed_waypoint=waypoint_id)
                        failed = True
                        break
                    residual = error(desired, actual)
                    if np.linalg.norm(residual[:3]) <= .0004 and np.linalg.norm(residual[3:]) <= .003:
                        converged = True
                        break
                    jac_p = np.zeros((3, int(native_model.nv)))
                    jac_r = np.zeros_like(jac_p)
                    mujoco.mj_jacSite(native_model, native_data, jac_p, jac_r, site)
                    jacobian = np.concatenate((jac_p[:, dof_indexes], jac_r[:, dof_indexes]))
                    weighted_j = weights[:, None] * jacobian
                    weighted_error = weights * residual
                    damping = .006
                    normal = weighted_j @ weighted_j.T + np.eye(6) * damping ** 2
                    inverse = weighted_j.T @ np.linalg.solve(normal, np.eye(6))
                    dq = inverse @ weighted_error
                    # Keep the local redundant-joint branch near its previous
                    # waypoint; this does not substitute a different global plan.
                    nullspace = np.eye(7) - inverse @ weighted_j
                    dq += nullspace @ (.04 * (reference_q - q))
                    largest = float(np.max(np.abs(dq)))
                    if largest > .035:
                        dq *= .035 / largest
                    current_cost = float(np.dot(weighted_error, weighted_error))
                    accepted = None
                    collision_rejections = []
                    for scale in (1., .5, .25, .125, .0625, .03125):
                        trial = np.clip(q + scale * dq, lower, upper)
                        if np.max(np.abs(trial - q)) < 1e-9:
                            continue
                        trial_pose, _, trial_bad = preview(trial)
                        if trial_bad:
                            collision_rejections = trial_bad
                            continue
                        trial_error = weights * error(desired, trial_pose)
                        trial_cost = float(np.dot(trial_error, trial_error))
                        if trial_cost >= current_cost - max(1e-15, current_cost * 1e-7):
                            continue
                        # Check the actual joint interpolation too, including
                        # the held object's coupled pose, before accepting it.
                        n = max(1, int(np.ceil(np.max(np.abs(trial - q)) / .01)),
                                int(np.ceil(np.sum(np.abs(trial - q)) * 1.5 / step)))
                        segment = []
                        for v in np.linspace(0, 1, n + 1)[1:]:
                            intermediate = q + (trial - q) * v
                            pose_i, object_i, bad_i = preview(intermediate)
                            if bad_i:
                                collision_rejections = bad_i
                                segment = []
                                break
                            segment.append((intermediate.copy(), pose_i.copy(),
                                            object_i.copy() if object_i is not None else None))
                        if segment:
                            accepted = trial, segment
                            break
                    if accepted is None:
                        result.update(reason="cartesian_ik_collision_blocked" if collision_rejections else "cartesian_ik_stalled",
                                      failed_waypoint=waypoint_id, failed_iteration=iteration,
                                      position_error_m=float(np.linalg.norm(residual[:3])),
                                      rotation_error_rad=float(np.linalg.norm(residual[3:])))
                        if collision_rejections:
                            result["collisions"] = collision_rejections
                        failed = True
                        break
                    q, segment = accepted
                    for q_i, pose_i, target_i in segment:
                        path_q.append(q_i)
                        path_pose.append(pose_i)
                        if held:
                            path_target.append(target_i)
                    result["accepted_ik_steps"] += 1
                if not converged and not failed:
                    result.update(reason="cartesian_ik_iteration_limit", failed_waypoint=waypoint_id)
                    failed = True
                if failed:
                    break
            if not failed:
                final_pose, _, final_bad = preview(q)
                final_error = error(goal, final_pose)
                if final_bad:
                    result.update(reason="cartesian_final_collision", collisions=final_bad)
                elif np.linalg.norm(final_error[:3]) > .0004 or np.linalg.norm(final_error[3:]) > .003:
                    result.update(reason="cartesian_final_pose_error")
                else:
                    result.update(success=True, waypoints=len(path_q),
                                  final_position_error_m=float(np.linalg.norm(final_error[:3])),
                                  final_rotation_error_rad=float(np.linalg.norm(final_error[3:])))
    except Exception as exc:
        result.update(success=False, reason="cartesian_planner_error", error=str(exc),
                      traceback=traceback.format_exc())
    finally:
        if snapshot is not None:
            try:
                _restore(adapter.env.sim, snapshot)
                result["state_restored"] = True
            except Exception as exc:
                result.update(success=False, reason="cartesian_state_restore_failed",
                              restore_error=str(exc), restore_traceback=traceback.format_exc())
        else:
            result["state_restored"] = True  # Validation/import failed before mutation.
    if result["success"]:
        path = output_dir / "PATH.npz"
        contents = dict(q_arm=np.asarray(path_q, dtype=np.float64),
                        world_T_eef=np.asarray(path_pose, dtype=np.float64))
        if path_target:
            contents["world_T_object"] = np.asarray(path_target, dtype=np.float64)
        np.savez_compressed(path, **contents)
        result["path_file"] = str(path)
    result["wall_s"] = time.monotonic() - started
    (output_dir / "CARTESIAN_RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
