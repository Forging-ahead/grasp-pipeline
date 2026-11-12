"""Cheap open-gripper endpoint screening before arm motion planning.

Move the actual compiled gripper collision geoms in an isolated MjData, keeping
their measured transform to the control site. No arm IK, dynamics, live-state
edits, grasp reranking, or new collision exemptions are involved. A passed
endpoint still needs the existing full-arm and approach-path checks.
"""
from __future__ import annotations

import time
import numpy as np

from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
from graspgen_libero_bridge.adapters.libero.scene import _contact_geom_ids


def screen_grasp_environment(adapter, candidates, minimum_clearance_m=0.,
                             allowed_finger_target_geoms=None):
    import mujoco

    started = time.monotonic()
    clearance = float(minimum_clearance_m)
    if not np.isfinite(clearance) or not 0 <= clearance <= .01:
        raise ValueError("Gripper environment clearance must be in [0, .01] m")
    sim = adapter.env.sim
    model, data = sim.model, sim.data
    native_model = getattr(model, "_model", model)
    native_data = getattr(data, "_data", data)
    if int(native_model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD):
        raise ValueError("Endpoint distance screening requires MuJoCo native CCD")
    robot = adapter.task_env.robots[0]
    root = int(model.body_name2id(robot.gripper.root_body))
    gripper_bodies = body_subtree_ids(model, root)
    robot_bodies = body_subtree_ids(model, model.body_name2id(robot.robot_model.root_body))
    target_bodies = body_subtree_ids(model, adapter.task_env.obj_body_id[adapter.target_object])
    allowed = None if allowed_finger_target_geoms is None else set(map(int, allowed_finger_target_geoms))
    target_geoms = {g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) in target_bodies}
    if allowed is not None and not allowed <= target_geoms:
        raise ValueError("Target finger permissions must stay on target geometry")
    contact_ids = _contact_geom_ids(model)
    gripper_geoms = sorted(g for g in contact_ids if int(model.geom_bodyid[g]) in gripper_bodies)
    world_geoms = sorted(g for g in contact_ids if int(model.geom_bodyid[g]) not in robot_bodies)
    if not gripper_geoms or not world_geoms:
        raise ValueError("Endpoint screening requires physical gripper and scene geometry")
    explicit = {frozenset((int(a), int(b))) for a, b in
                zip(np.asarray(model.pair_geom1), np.asarray(model.pair_geom2))}
    pairs = []
    permitted = 0
    for gripper in gripper_geoms:
        finger = "finger" in (model.body_id2name(int(model.geom_bodyid[gripper])) or "")
        for obstacle in world_geoms:
            active = ((int(model.geom_contype[gripper]) & int(model.geom_conaffinity[obstacle])) or
                      (int(model.geom_contype[obstacle]) & int(model.geom_conaffinity[gripper])) or
                      frozenset((gripper, obstacle)) in explicit)
            if not active:
                continue
            # Same final-engagement permission as ContactMonitor. The target is
            # not removed: hand/palm contact with it is still checked.
            if (finger and int(model.geom_bodyid[obstacle]) in target_bodies and
                    (allowed is None or obstacle in allowed)):
                permitted += 1
                continue
            if int(model.geom_type[gripper]) not in (2, 3, 4, 5, 6, 7) or int(model.geom_type[obstacle]) not in range(8):
                raise ValueError("Unsupported endpoint collision geometry")
            pairs.append((gripper, obstacle))
    if not pairs:
        raise ValueError("Endpoint screening found no checkable scene pairs")
    scratch = mujoco.MjData(native_model)
    scratch.geom_xpos[:] = native_data.geom_xpos
    scratch.geom_xmat[:] = native_data.geom_xmat
    site = int(robot.eef_site_id)
    rotation = np.asarray(data.site_xmat[site]).reshape(3, 3)
    position = np.asarray(data.site_xpos[site])
    local_positions = (np.asarray(data.geom_xpos)[gripper_geoms] - position) @ rotation
    local_rotations = np.einsum("ij,njk->nik", rotation.T,
        np.asarray(data.geom_xmat)[gripper_geoms].reshape(-1, 3, 3))
    kept, audits, checks = [], [], 0
    cutoff = max(.002, clearance + .001)
    witness = np.zeros(6)
    for rank, candidate in enumerate(candidates):
        pose = np.asarray(candidate["world_T_eef"], dtype=float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
                not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-8) or
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5) or
                np.linalg.det(pose[:3, :3]) < .99999):
            raise ValueError("Invalid grasp control pose for environment screening")
        scratch.geom_xpos[gripper_geoms] = local_positions @ pose[:3, :3].T + pose[:3, 3]
        scratch.geom_xmat[gripper_geoms] = np.einsum(
            "ij,njk->nik", pose[:3, :3], local_rotations).reshape(-1, 9)
        blocking = None
        for gripper, obstacle in pairs:
            distance = float(mujoco.mj_geomDistance(native_model, scratch, gripper, obstacle, cutoff, witness))
            checks += 1
            if not np.isfinite(distance):
                raise ValueError("Non-finite gripper/scene distance")
            # Zero separation is not a certificate of being collision-free;
            # touching and degenerate overlapping CCD queries must be rejected.
            if distance <= clearance + 1e-7:
                blocking = dict(gripper_geom=model.geom_id2name(gripper) or f"geom_{gripper}",
                    obstacle_geom=model.geom_id2name(obstacle) or f"geom_{obstacle}",
                    distance_m=distance, witness_world=witness.reshape(2, 3).tolist())
                break
        audits.append(dict(score_rank=rank, generation_index=candidate["generation_index"],
            score=float(candidate["score"]), passed=blocking is None, blocking_contact=blocking))
        if blocking is None:
            kept.append(candidate)
    audit = dict(enabled=True, input_candidates=len(candidates), passed=len(kept),
        rejected=len(candidates)-len(kept), minimum_clearance_m=clearance,
        numerical_tolerance_m=1e-7, pair_distance_queries=checks,
        gripper_geom_count=len(gripper_geoms), world_geom_count=len(world_geoms),
        permitted_finger_target_pairs=permitted, measured_gripper_width_m=float(adapter.state()["gripper_width"]),
        allowed_finger_target_geom_ids=None if allowed is None else sorted(allowed),
        uses_ground_truth_collision_geometry=True, score_order_preserved=True,
        live_state_mutated=False, extra_collision_exemptions=[],
        scope="open-gripper endpoint only; full-arm reachability and approach paths still require planning",
        candidates=audits, wall_s=time.monotonic()-started)
    return kept, audit
