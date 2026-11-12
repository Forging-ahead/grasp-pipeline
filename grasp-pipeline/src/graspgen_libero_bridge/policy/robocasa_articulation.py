"""Physical handle contact followed by joint-constrained closed-loop motion."""
from __future__ import annotations

import numpy as np

from graspgen_libero_bridge.adapters.robocasa.articulation import native_joints, joint_motion_pose
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
from graspgen_libero_bridge.adapters.robocasa.native import write_json


def execute_articulation(environment, executor, local, recorder, planner, stage, index):
    env = environment
    if not env.config["policy"]["articulation_enabled"]:
        raise EpisodeFailure("UNSUPPORTED_SKILL", "Articulation is disabled pending native validation")
    target = stage["fixture"]
    local.target = target
    joints = native_joints(env, stage)
    output = env.output / "stages" / f"{index:02d}_articulation"
    output.mkdir(parents=True)
    write_json(output / "NATIVE_JOINTS.json", joints)
    completed = []
    try:
        if not joints:
            recorder.decision("skip_articulation", {"fixture": target, "reason": "native_fixture_already_open_no_actuated_door"})
        for spec in joints:
            recorder.decision("articulate_joint", {"fixture": target, "joint": spec["name"],
                                "goal": spec["goal"], "axis_world": spec["axis_world"], "anchor_world": spec["anchor_world"]})
            model, data = env.env.sim.model, env.env.sim.data
            joint = spec["joint_id"]
            address = int(model.jnt_qposadr[joint])
            if abs(float(data.qpos[address]) - spec["goal"]) < .015:
                continue
            z = -spec["fixture_front_world"]
            z /= np.linalg.norm(z)
            for closing in (np.array([0., 0., 1.]), np.array([1., 0., 0.])):
                x = closing - z * np.dot(closing, z)
                if np.linalg.norm(x) < .1:
                    continue
                x /= np.linalg.norm(x); y = np.cross(z, x)
                grasp = np.eye(4); grasp[:3, :3] = np.stack([x, y, z], axis=1)
                grasp[:3, 3] = spec["handle_center_world"]
                pre = grasp.copy(); pre[:3, 3] -= z * env.config["policy"]["pregrasp_distance_m"]
                local.allow_finger_target = False
                reply = planner.global_plan(target, pre, label="handle_pregrasp")
                if reply.get("success"):
                    executor.follow_path(reply["path_file"], phase="handle/pregrasp", gripper=-1)
                    break
            else:
                raise EpisodeFailure("SEARCH_BUDGET_EXHAUSTED", "No collision-free handle approach")
            local.allowed_target_geoms = set(spec["handle_geom_ids"])
            local.allow_finger_target = True
            reply = planner.local_plan(grasp, label="handle_contact")
            if not reply["success"]:
                raise EpisodeFailure("COLLISION_REJECTED", "Handle contact path rejected", details=reply)
            executor.follow_path(reply["path_file"], phase="handle/contact", gripper=-1)
            executor.hold(env.config["policy"]["close_seconds"], gripper=1, phase="handle/close")
            reference = np.asarray(env.robot_state()["world_T_eef"])
            initial = float(data.qpos[address])
            increment = .015 if spec["joint_type"] == 2 else .07
            count = max(2, int(np.ceil(abs(spec["goal"] - initial) / increment)) + 1)
            progress = []
            for desired_joint in np.linspace(initial, spec["goal"], count)[1:]:
                env.check_budget()
                goal = joint_motion_pose(reference, spec["axis_world"], spec["anchor_world"],
                                         desired_joint - initial, spec["joint_type"])
                local.articulation_preview = {joint: float(desired_joint)}
                reply = planner.local_plan(goal, label="joint_follow")
                local.articulation_preview.clear()
                if not reply["success"]:
                    raise EpisodeFailure("ARTICULATION_STALLED", "Constrained IK/contact path rejected", details=reply)
                executor.follow_path(reply["path_file"], phase="handle/joint_follow", gripper=1)
                measured = float(data.qpos[address])
                progress.append({"step": env.steps, "requested": float(desired_joint), "measured": measured})
                write_json(output / (spec["name"] + "_PROGRESS.json"), progress)
                if abs(measured - desired_joint) > max(.04, 3. * increment):
                    raise EpisodeFailure("ARTICULATION_STALLED", "Handle slipped or native joint stopped progressing", details={"progress": progress})
            measured = float(data.qpos[address])
            if abs(measured - spec["goal"]) > max(.03, .05 * abs(spec["goal"] - initial)):
                raise EpisodeFailure("ARTICULATION_STALLED", "Native joint did not reach its intended state")
            executor.hold(env.config["policy"]["open_seconds"], gripper=-1, phase="handle/release")
            retreat = np.asarray(env.robot_state()["world_T_eef"])
            retreat[:3, 3] -= retreat[:3, 2] * env.config["policy"]["pregrasp_distance_m"]
            reply = planner.local_plan(retreat, label="handle_retreat")
            if not reply["success"]:
                raise EpisodeFailure("COLLISION_REJECTED", "Handle retreat rejected", details=reply)
            executor.follow_path(reply["path_file"], phase="handle/retreat", gripper=-1)
            completed.append({"joint": spec["name"], "measured": measured, "goal": spec["goal"]})
        fixture = getattr(env.env, target)
        if stage["goal"] in {"open", "close"}:
            native_subgoal = fixture.is_open(env.env, th=.95) if stage["goal"] == "open" else fixture.is_closed(env.env, th=.05)
        elif target == "sink":
            native_subgoal = bool(fixture.get_handle_state(env.env)["water_on"]) == (stage["goal"] == "turn_on")
        else:
            native_subgoal = not fixture.is_burner_on(env.env, burner_loc=env.env.knob)
        if not native_subgoal:
            raise EpisodeFailure("ARTICULATION_STALLED", "Native fixture subgoal remains false after handle execution")
        return {"skill": "articulation", "fixture": target, "goal": stage["goal"],
                "subgoal_success": True, "joints": completed, "physical_steps": env.steps,
                "already_satisfied_without_actuated_door": not bool(joints)}
    finally:
        local.articulation_preview.clear()
        local.allowed_target_geoms = None
        local.allow_finger_target = False
