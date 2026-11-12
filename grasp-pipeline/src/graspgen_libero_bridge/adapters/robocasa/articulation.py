"""Native joint/handle bindings and world-frame constrained motion."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
from .errors import EpisodeFailure


def joint_motion_pose(initial_pose, axis, anchor, displacement, joint_type):
    pose = np.asarray(initial_pose, float).copy()
    axis, anchor = np.asarray(axis, float), np.asarray(anchor, float)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("Joint axis is invalid")
    axis = axis / norm
    if joint_type == 2:  # MuJoCo slide
        pose[:3, 3] += axis * float(displacement)
    elif joint_type == 3:  # MuJoCo hinge
        rotation = Rotation.from_rotvec(axis * float(displacement)).as_matrix()
        pose[:3, 3] = anchor + rotation @ (pose[:3, 3] - anchor)
        pose[:3, :3] = rotation @ pose[:3, :3]
    else:
        raise ValueError("Only scalar hinge/slide articulation is supported")
    return pose


def native_joints(environment, stage):
    env = environment.env
    fixture = getattr(env, stage["fixture"])
    model = env.sim.model
    if stage["goal"] in {"open", "close"}:
        names = list(fixture.door_joint_names)
    elif stage["fixture"] == "sink":
        names = [fixture.naming_prefix + "handle_joint"]
    elif stage["fixture"] == "stove":
        names = [fixture.knob_joints[env.knob].get("name")]
    else:
        raise EpisodeFailure("UNSUPPORTED_TASK_BINDING", "No joint semantic binding for fixture")
    if not names:
        # CABINET can resolve to an open shelf. Match the native fixture's
        # already-open predicate; do not invent a handle or set a joint value.
        if stage["goal"] == "open" and callable(getattr(fixture, "is_open", None)) and fixture.is_open(env):
            return []
        raise EpisodeFailure("UNSUPPORTED_TASK_BINDING", "Fixture exposes no target joints")
    result = []
    data = getattr(env.sim.data, "_data", env.sim.data)
    for name in names:
        joint = int(model.joint_name2id(name))
        kind = int(model.jnt_type[joint])
        if kind not in {2, 3}:
            raise EpisodeFailure("UNSUPPORTED_SKILL", "Unsupported articulation joint type")
        low, high = map(float, model.jnt_range[joint])
        current = float(data.qpos[int(model.jnt_qposadr[joint])])
        if stage["goal"] in {"open", "close"}:
            closed, opened = sorted((low, high), key=abs)
            fraction = .98 if stage["goal"] == "open" else .02
            goal = closed + fraction * (opened - closed)
        elif stage["goal"] == "turn_on":
            # Native Sink.set_handle_state(on) uses the semantic on interval .40-.50 rad.
            goal = .45
        elif stage["goal"] == "turn_off":
            goal = 0.
        else:
            raise EpisodeFailure("UNSUPPORTED_TASK_BINDING", "Unknown articulation goal")
        body = int(model.jnt_bodyid[joint])
        bodies = body_subtree_ids(model, body)
        handle_geoms = [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) in bodies
                        and any(token in (model.geom_id2name(g) or "").lower() for token in ("handle", "knob"))
                        and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
        if not handle_geoms:
            raise EpisodeFailure("UNSUPPORTED_TASK_BINDING", "No collision handle geometry bound to " + name)
        center = np.mean(np.asarray(data.geom_xpos[handle_geoms]), axis=0)
        result.append({"name": name, "joint_id": joint, "joint_type": kind,
                       "initial": current, "goal": goal, "range": [low, high],
                       "axis_world": np.asarray(data.xaxis[joint]).copy(),
                       "anchor_world": np.asarray(data.xanchor[joint]).copy(),
                       "handle_center_world": center,
                       "handle_geom_ids": handle_geoms,
                       "fixture_front_world": environment.entity_pose(stage["fixture"])[:3, :3] @ np.array([0., -1., 0.])})
    return result
