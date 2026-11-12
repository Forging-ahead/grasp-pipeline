"""Continuous RoboCasa teacher skills; semantic bindings contain no spatial waypoints."""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.common.rpc import WorkerClient
from graspgen_libero_bridge.planning.grasp_candidate import control_target_distance
from graspgen_libero_bridge.planning.robocasa_planner import RoboCasaPlanner
from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
from graspgen_libero_bridge.adapters.robocasa.goals import placement_goals, check_placement
from graspgen_libero_bridge.adapters.robocasa.native import write_json
from graspgen_libero_bridge.runtime.robocasa_worker_env import worker_environment


def require_plan(reply, stage):
    if not reply.get("success"):
        raise EpisodeFailure(reply.get("reason", "SEARCH_BUDGET_EXHAUSTED"),
                             "No executable path for " + stage, details=reply)
    return reply["path_file"]


def grasp_precheck_from_pregrasp(local, path_file, grasp_tcp):
    """Check the contact pose from the branch selected by the global planner.

    Solving directly from the distant current posture can converge to a
    colliding elbow branch and incorrectly discard an executable grasp.
    This only changes isolated preview data, never the real robot state.
    """
    with np.load(path_file, allow_pickle=False) as path:
        joints = np.asarray(path["q_arm"], float)
    if joints.ndim != 2 or joints.shape[1] != 7 or not len(joints) or not np.isfinite(joints).all():
        raise ValueError("Malformed pregrasp joint path")
    permission = local.allow_finger_target
    try:
        local.allow_finger_target = True
        return local.solve(grasp_tcp, start=joints[-1])
    finally:
        local.allow_finger_target = permission


def object_contacts(environment, target):
    model, data = environment.env.sim.model, environment.env.sim.data
    bodies = body_subtree_ids(model, environment.env.obj_body_id[target])
    support = set()
    for c in data.contact[:data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if (b1 in bodies) != (b2 in bodies):
            support.add(g2 if b1 in bodies else g1)
    return support


def grasp_evidence(environment, target):
    robot = environment.robot
    gripper = robot.gripper["right"]
    native = bool(environment.env._check_grasp(gripper, environment.env.objects[target]))
    return {"native_bilateral_grasp": native, "target": target,
            "q_fingers": environment.robot_state()["q_fingers"],
            "world_T_tcp": environment.robot_state()["world_T_eef"],
            "world_T_object": environment.object_pose(target).tolist(),
            "physical_step": environment.steps}


def lift_with_attachment(env, executor, local, recorder, planner, stage, directory, initial_support):
    """Preserve v10 pre-lift attachment and bounded departure permissions.

    Ignoring observed supports is limited to the global lift fallback, as in
    LIBERO. The complete path must still pass native geometry checks, which
    continue to reject robot/world collisions.
    """
    target, contents = stage["target"], stage.get("carried_contents", [])
    support = set(initial_support) | object_contacts(env, target)
    local.allowed_support_geoms = support
    planner.attach(target, contents)
    scene = planner.scene(target)
    support_names = sorted({g["name"] for g in scene["obstacles"] if g["geom_id"] in support})
    write_json(directory / "PRE_LIFT_CONTEXT.json", {"target": target, "attached_before_lift": True,
        "departure_support_geom_ids": sorted(support), "global_lift_departure_support_names": support_names,
        "native_robot_world_contacts_still_checked": True})
    executor.ensure_joint_margin("lift", gripper=1)
    lift = np.asarray(env.robot_state()["world_T_eef"]).copy()
    lift[2, 3] += env.config["policy"]["lift_clearance_m"]
    recorder.phase = "lift_decision"
    recorder.decision("lift_object", {"target": target, "world_T_tcp": lift})
    path = planner.contact_plan(lift, label="lift")
    if not path["success"] and path.get("state_restored") and path.get("reason") in {
            "cartesian_ik_stalled", "cartesian_ik_iteration_limit", "cartesian_ik_collision_blocked", "cartesian_collision"}:
        path = planner.global_plan(target, lift, label="lift_global", allow_target_contact=True,
                                   ignore_obstacles=support_names)
    executor.follow_path(require_plan(path, "lift"), phase="lift", gripper=1)
    executor.hold(.75, gripper=1, phase="lift_verify")


class PickPlaceSkill:
    def __init__(self, environment, executor, local, recorder, planner):
        self.env, self.executor, self.local, self.recorder, self.planner = environment, executor, local, recorder, planner
        self.model = None
        self.model_wall_s = 0.

    def _candidates(self, target, directory, grasp_to_tcp):
        env = self.env
        self.recorder.phase = "perception"
        cloud, provenance = env.target_cloud(target)
        directory.mkdir(parents=True)
        points_path = directory / "CLOUD.npz"
        np.savez_compressed(points_path, points_world=cloud)
        provenance["points_sha256"] = hashlib.sha256(np.asarray(cloud, dtype="<f4").tobytes()).hexdigest()
        provenance["grasp_T_tcp"] = grasp_to_tcp
        write_json(directory / "CLOUD.json", provenance)
        seed = env.config["seed"]
        if env.config.get("candidate_replay_case"):
            from graspgen_libero_bridge.runtime.robocasa_replay import replay_candidates
            replay_candidates(env, directory, cloud)
        elif env.config["policy"].get("grasp_generator", "graspgen") == "geometric":
            points = np.asarray(cloud)
            center = np.median(points, axis=0)
            _, vectors = np.linalg.eigh(np.cov(points[:, :2].T))
            angle = np.arctan2(vectors[1, 0], vectors[0, 0])
            poses = []
            for height in np.quantile(points[:, 2], [.25, .5, .75]):
                for yaw in np.linspace(0., np.pi, 16, endpoint=False):
                    x = np.array([np.cos(angle + yaw), np.sin(angle + yaw), 0.])
                    z = np.array([0., 0., -1.]); y = np.cross(z, x)
                    tcp = np.eye(4); tcp[:3, :3] = np.stack([x, y, z], axis=1)
                    tcp[:3, 3] = [center[0], center[1], height]
                    poses.append(tcp @ np.linalg.inv(grasp_to_tcp))
            poses = np.asarray(poses)
            scores = np.linspace(1., 0., len(poses))
            np.savez_compressed(directory / "CANDIDATES.npz", poses_world_grasp=poses, scores=scores)
            write_json(directory / "GENERATOR.json", {"backend": "partial_cloud_PCA_geometry", "proposals": len(poses)})
        else:
            self.recorder.phase = "graspgen_inference"
            if self.model is None:
                paths = env.config["paths"]
                self.model = WorkerClient(paths["graspgen_python"], "graspgen_libero_bridge.models.graspgen_worker",
                    env.output / "graspgen_worker", arguments=("--config", paths["graspgen_config"], "--grasp-repo", paths["grasp_repo"]),
                    environment=worker_environment(env.output),
                    timeout=min(180., env.remaining_seconds()))
            response = self.model.request({"command": "infer", "points_path": str(points_path),
                "output_path": str(directory / "CANDIDATES.npz"), "seed": seed,
                "num_grasps": env.config["policy"]["grasp_proposals"]}, timeout=max(.1, env.remaining_seconds()))
            self.model_wall_s += float(response["generation_s"])
            write_json(directory / "GENERATOR.json", response)
        with np.load(directory / "CANDIDATES.npz", allow_pickle=False) as data:
            poses, scores = data["poses_world_grasp"].copy(), data["scores"].copy()
        order = np.argsort(-scores, kind="stable")
        if env.config["policy"]["grasp_ranking"] == "seeded_random":
            order = np.random.default_rng(seed).permutation(len(order))
        return poses, scores, order

    def _attempt_grasp(self, stage, directory):
        env, local, executor, planner = self.env, self.local, self.executor, self.planner
        target = stage["target"]
        local.target = target
        local.allow_finger_target = False
        local.allowed_support_geoms.clear()
        initial_object = env.object_pose(target)
        initial_tcp = np.asarray(env.robot_state()["world_T_eef"])
        initial_q = np.asarray(env.robot_state()["q_arm"])
        initial_support = object_contacts(env, target)
        grasp_to_tcp = np.asarray(env.robot_state()["grasp_T_tcp"])
        write_json(directory.parent / "GRASP_FRAME_CALIBRATION.json", {
            "grasp_T_tcp": grasp_to_tcp, "native_gripper": type(env.robot.gripper["right"]).__name__,
            "source": "measured native gripper root to grip_site transform",
            "reference_assumption": "official franka_panda GraspGen grasp frame is the compatible Panda gripper root"})
        poses, scores, order = self._candidates(target, directory, grasp_to_tcp)
        from graspgen_libero_bridge.policy.robocasa_v10 import V10Selection, execute_selected_grasp
        self.recorder.phase = "candidate_selection"
        scene = planner.scene(target)
        rules = env.config["policy"]["libero_v10_rules"]
        candidates, rejected = [], []
        for rank, index in enumerate(order):
            pose = poses[index].astype(float)
            pose[:3, :3] = Rotation.from_matrix(pose[:3, :3]).as_matrix()
            if pose[2, 2] > -rules["minimum_downward_approach"]:
                rejected.append({"candidate_index": int(index), "reason": "v10_downward_approach_filter"})
                continue
            tcp = pose @ grasp_to_tcp
            distance = control_target_distance(tcp, scene["target"]["bounds_world"])
            if distance > rules["grasp_control_target_margin_m"]:
                rejected.append({"candidate_index": int(index), "reason": "v10_grip_center_misses_target", "distance_m": distance})
                continue
            random_order = env.config["policy"]["grasp_ranking"] == "seeded_random"
            candidates.append({"world_T_eef": tcp.tolist(), "generation_index": int(index),
                               "score": float(-rank) if random_order else float(scores[index]),
                               "model_score": float(scores[index]), "score_source": "seeded_rank" if random_order else "discriminator"})
        write_json(directory / "SCREENED_CANDIDATES.json", {"candidates": candidates, "rejected": rejected})
        # v10 physically opens the gripper before planning with its measured aperture.
        executor.hold(1., gripper=-1, phase="prepare_gripper")
        if np.diff(np.asarray(env.robot_state()["q_fingers"]))[0] > -.07:
            raise EpisodeFailure("GRIPPER_PREPARATION_FAILED", "The measured Panda aperture did not reach 7 cm")
        executor.ensure_joint_margin("select_candidate", gripper=-1)
        selector = V10Selection(env, local, planner, target, directory / "v10", executor)
        selected = selector.choose(candidates)
        grasp_tcp = np.asarray(selected["selected"]["world_T_eef"])
        with np.load(selected["approach_path_file"], allow_pickle=False) as path:
            pregrasp = path["world_T_eef"][-1].copy()
        self.recorder.phase = "grasp_decision"
        self.recorder.decision("execute_grasp", {"target": target, "world_T_tcp": grasp_tcp,
                "world_T_pregrasp": pregrasp, "grasp_T_tcp": grasp_to_tcp})
        execute_selected_grasp(env, executor, planner, selected)
        executor.hold(env.config["policy"]["close_seconds"], gripper=1, phase="close")
        evidence = grasp_evidence(env, target)
        write_json(directory / "CLOSE_EVIDENCE.json", evidence)
        if not evidence["native_bilateral_grasp"]:
            raise EpisodeFailure("GRASP_CONTACT_FAILED", "Native bilateral contact absent after real closure", details=evidence)
        contents = stage.get("carried_contents", [])
        lift_with_attachment(env, executor, local, self.recorder, planner, stage, directory, initial_support)
        evidence = grasp_evidence(env, target)
        evidence["object_lift_m"] = float(env.object_pose(target)[2, 3] - initial_object[2, 3])
        write_json(directory / "LIFT_EVIDENCE.json", evidence)
        if not evidence["native_bilateral_grasp"] or evidence["object_lift_m"] < .5 * env.config["policy"]["lift_clearance_m"]:
            raise EpisodeFailure("HOLD_FAILED", "Object failed the real lift-and-hold check", details=evidence)
        local.allowed_support_geoms.clear()
        local.attach([target, *contents])
        planner.attach(target, contents)
        if contents:
            import robocasa.utils.object_utils as OU
            if not all(OU.check_obj_in_receptacle(env.env, name, target) for name in contents):
                raise EpisodeFailure("CONTENTS_LOST", "Container contents were lost during lift")
            scene = planner.scene(target)
        from graspgen_libero_bridge.perception.pointcloud import sample_mujoco_object_points
        points_world, geometry = sample_mujoco_object_points(env.env, target, count=4096, seed=0,
                                                             geometry="collision", return_metadata=True)
        object_pose = env.object_pose(target)
        placement_points = (points_world - object_pose[:3, 3]) @ object_pose[:3, :3]
        geometry.update(usage="v10 teacher placement geometry only; never fed to GraspGen")
        write_json(directory / "PLACEMENT_GEOMETRY.json", geometry)
        return {"initial_object_pose": initial_object, "initial_tcp_pose": initial_tcp,
                "initial_q_arm": initial_q,
                "v10_job": selector,
                "tcp_T_object": np.linalg.inv(np.asarray(env.robot_state()["world_T_eef"])) @ env.object_pose(target),
                "object_points": placement_points}

    def execute(self, stage, index):
        env, executor, local, planner = self.env, self.executor, self.local, self.planner
        if stage.get("carried_contents") and not env.config["policy"].get("carried_contents_enabled", False):
            raise EpisodeFailure("UNSUPPORTED_SKILL", "Multi-body carried contents require validated attachment support",
                                 details={"contents": stage["carried_contents"]})
        target = stage["target"]
        error = None
        for attempt in range(int(env.config["policy"]["physical_grasp_attempts"])):
            directory = env.output / "stages" / f"{index:02d}_{target}" / f"grasp_{attempt:02d}"
            try:
                carried = self._attempt_grasp(stage, directory)
                error = None
                break
            except EpisodeFailure as exc:
                error = exc
                if exc.code not in {"GRASP_CONTACT_FAILED", "HOLD_FAILED"}:
                    raise
                write_json(directory / "ATTEMPT_FAILURE.json", {"code": exc.code, "message": str(exc), "details": exc.details})
                planner.detach()
                local.allowed_support_geoms.clear()
                executor.hold(env.config["policy"]["open_seconds"], gripper=-1, phase="retry_release")
                retreat = np.asarray(env.robot_state()["world_T_eef"])
                retreat[:3, 3] -= retreat[:3, 2] * env.config["policy"]["pregrasp_distance_m"]
                reply = planner.contact_plan(retreat, label="retry_retreat")
                executor.follow_path(require_plan(reply, "retry_retreat"), phase="retry_retreat", gripper=-1)
        if error is not None:
            raise error
        from graspgen_libero_bridge.policy.robocasa_v10 import place_v10, withdraw_v10
        output = env.output / "stages" / f"{index:02d}_{target}"
        placed_state = place_v10(env, executor, planner, stage, carried, output)
        planner.detach()
        self.recorder.phase = "release_decision"
        self.recorder.decision("open_gripper", {"world_T_tcp": env.robot_state()["world_T_eef"], "target": target})
        executor.hold(env.config["policy"]["open_seconds"], gripper=-1, phase="release")
        from graspgen_libero_bridge.control.robocasa_executor import StableNativeGoal
        stable_goal = StableNativeGoal(env, env.config["policy"]["settle_seconds"]) if index == len(env.config["task"]["stages"]) - 1 else None
        escape = withdraw_v10(env, executor, planner, carried, placed_state, stable_goal=stable_goal)
        if not escape.get("task_goal_reached"):
            executor.hold(env.config["policy"]["settle_seconds"], gripper=-1, phase="settle")
        write_json(output / "POST_RELEASE_ESCAPE.json", escape)
        placed = check_placement(env, stage)
        if not placed:
            raise EpisodeFailure("PLACEMENT_UNSTABLE", "Native placement subgoal is false after release and settling")
        return {"skill": "pick_place", "target": target, "destination": stage["destination"],
                "subgoal_success": placed, "physical_steps": env.steps, "post_release_escape": escape}

    def close(self):
        if self.model is not None:
            self.model.close()


def run_policy(environment, executor, local, recorder):
    env = environment
    stages = env.config["task"]["stages"]
    from graspgen_libero_bridge.adapters.robocasa.capabilities import capabilities
    readiness = capabilities(env.config)
    if not readiness["static_task_gate_passed"]:
        raise EpisodeFailure("UNSUPPORTED_SKILL", "Required experimental skills are not enabled", details=readiness)
    if env.native_success_at_reset:
        raise EpisodeFailure("ALREADY_COMPLETE_AT_RESET", "Native goal was true before any task actions")
    planner = RoboCasaPlanner(env, local)
    pick_place = PickPlaceSkill(env, executor, local, recorder, planner)
    outcomes = []
    try:
        from graspgen_libero_bridge.policy.robocasa_v10 import record_rule_sources
        record_rule_sources(env.output)
        from graspgen_libero_bridge.planning.robocasa_robot import prepare_start_state
        recovery = prepare_start_state(env, executor, local)
        for index, stage in enumerate(stages):
            env.check_budget()
            recorder.phase = f"stage_{index:02d}/" + stage["skill"]
            recorder.decision(stage["skill"], stage, expected_subgoal="native task binding")
            if stage["skill"] == "pick_place":
                outcome = pick_place.execute(stage, index)
            elif stage["skill"] == "articulation":
                from graspgen_libero_bridge.policy.robocasa_articulation import execute_articulation
                outcome = execute_articulation(env, executor, local, recorder, planner, stage, index)
            elif stage["skill"] == "navigate":
                from graspgen_libero_bridge.policy.robocasa_navigation import execute_navigation
                outcome = execute_navigation(env, executor, local, recorder,
                          getattr(env.env, stage["native_target"]), getattr(env.env, stage["native_orientation"]))
            else:
                raise EpisodeFailure("UNSUPPORTED_SKILL", "Skill is not physically validated: " + stage["skill"], details=stage)
            outcomes.append(outcome)
            write_json(env.output / "STAGE_OUTCOMES.json", outcomes)
        success = env.check_native_success()
        if not success:
            raise EpisodeFailure("NATIVE_GOAL_FALSE", "All scheduled stages ended but final native goal is false")
        return {"status": "success", "pipeline_success": True, "training_candidate": False,
                "training_audit_pending": True, "stage_outcomes": outcomes,
                "graspgen_inference_s": pick_place.model_wall_s,
                "actual_global_planner": env.config["policy"]["global_planner"],
                "start_recovery": recovery,
                "instruction_order_verified": True}
    finally:
        pick_place.close()
        planner.close()
