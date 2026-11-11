"""Persistent cuRobo worker for the independent LIBERO bridge.

No GPU is initialized until a configure request is received.  Commands:

configure: {scene, calibration_samples, output_dir}
plan_pose: {scene, goal_pose, out_dir, timeout_s?, allow_target_contact?,
            ignore_obstacles?: [exact_geom_name]}
select:    {scene, candidates:[{world_T_eef,score,...}], out_dir,
            max_candidates?, approach_distance_m?, allow_target_contact?}
attach:    {scene}; detach: {}; status: {}

Each plan returns path_file; NPZ has q_arm[N,7], world_T_eef[N,4,4].
select also returns approach_path_file / descend_path_file.  Coordinates are
meters and world-frame control-site transforms.  No Isaac modules are used.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.planning.planning_geometry import (ARM_NAMES, FINGER_NAMES, PandaMJCF, pose_list,
                                target_spheres, transform, DEFAULT_ATTACHMENT_SLOTS,
                                SELF_COLLISION_SPHERE_LIMIT, MAX_ROBOT_SPHERES,
                                sphere_world_diagnostics)


ATTACHMENT_SLOTS = DEFAULT_ATTACHMENT_SLOTS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def arm_array(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (7,) or not np.isfinite(a).all():
        raise ValueError("q_arm must have seven finite values")
    return a


def finger_values(scene):
    if "q_fingers" in scene:
        a = np.asarray(scene["q_fingers"], dtype=float)
    else:
        width = float(scene["gripper_width"])
        a = np.asarray([width/2, -width/2])
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError('Panda q_fingers must contain two finite measured joint positions')
    # MuJoCo joint limits are soft constraints. A released finger can exceed
    # its stop by tens of micrometers during inter-skill handoff. These joints
    # are locked geometric inputs in cuRobo, not optimized arm variables:
    # retain the measured values exactly, including the small excursion, so
    # collision geometry matches physics. Do not clip or alter physical qpos.
    tolerance=.0005
    if not (-tolerance <= a[0] <= .04+tolerance and -.04-tolerance <= a[1] <= tolerance):
        raise ValueError('Panda passive finger state outside bounded soft-limit band: '
            f'q_fingers={a.tolist()}, tolerance_m={tolerance}; two positive Isaac finger joints are unsupported')
    return a


def densify(path, maximum_joint_step=.025):
    q = np.asarray(path, dtype=float)
    if q.ndim != 2 or q.shape[1] != 7 or len(q) == 0 or not np.isfinite(q).all():
        raise ValueError("Invalid cuRobo path")
    points = [q[0]]
    for a, b in zip(q[:-1], q[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b-a)) / maximum_joint_step)))
        points.extend(a+(b-a)*u for u in np.linspace(0, 1, n+1)[1:])
    return np.asarray(points, np.float32)


class BridgePlanner:
    def __init__(self, request, assets_dir=None):
        import torch
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        self.torch = torch
        self.WorldConfig = WorldConfig
        self.device = TensorDeviceType()
        self.scene = request["scene"]
        self.base = transform(self.scene["world_T_base"])
        self.base_inverse = np.linalg.inv(self.base)
        self.output_dir = Path(request.get("output_dir", request.get("out_dir", "planner_output"))).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.attachment_slots = int(request.get("attachment_slots", DEFAULT_ATTACHMENT_SLOTS))
        self.robot = PandaMJCF(self.output_dir / "robot_model", assets_dir=assets_dir,
                              max_robot_spheres=int(request.get("max_robot_spheres", MAX_ROBOT_SPHERES)))
        self.attached = None
        self.attachment_target_name = None
        self.fingers = finger_values(self.scene)
        self.config = self.robot.robot_config(self.fingers, self.attachment_slots)
        sphere_count = sum(map(len, self.config["kinematics"]["collision_spheres"].values())) + self.attachment_slots
        if sphere_count >= SELF_COLLISION_SPHERE_LIMIT:
            raise ValueError(f"Unsafe cuRobo sphere count {sphere_count}; CUDA kernel requires <1024")
        self._check_cpu_calibration(request.get("calibration_samples", []))
        self.world_data, self.world_report = self.build_world(self.scene)
        world = WorldConfig.from_dict(self.world_data)
        config = MotionGenConfig.load_from_robot_config(
            copy.deepcopy(self.config), world, self.device,
            interpolation_dt=float(request.get("interpolation_dt", .05)),
            use_cuda_graph=False, collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": max(512, len(self.world_data.get("cuboid", {}))+64),
                             "mesh": max(512, len(self.world_data.get("mesh", {}))+64)},
            num_ik_seeds=32, num_trajopt_seeds=4,
            position_threshold=.001, rotation_threshold=.005,
            collision_activation_distance=.002,
            self_collision_check=True, self_collision_opt=True)
        self.motion_gen = MotionGen(config)
        if list(self.motion_gen.kinematics.joint_names) != ARM_NAMES:
            raise ValueError(f"Unexpected cuRobo joint order: {self.motion_gen.kinematics.joint_names}")
        self._check_gpu_calibration(request["calibration_samples"])
        counts = {name: len(values) for name, values in self.robot.spheres.items()}
        counts["attached_object"] = self.attachment_slots
        names, ignored = list(counts), self.config["kinematics"]["self_collision_ignore"]
        pairs = sum(counts[a]*counts[b] for i, a in enumerate(names) for b in names[i+1:]
                    if b not in ignored.get(a, []) and a not in ignored.get(b, []))
        self.calibration["collision_budget"] = dict(total_spheres=sphere_count, attachment_slots=self.attachment_slots,
            self_collision_pairs=pairs, fast_kernel_pair_limit=16384, expected_fast_self_collision_kernel=pairs <= 16384)
        write_json(self.output_dir / "ROBOT_CONFIG.json", self.config)
        write_json(self.output_dir / "CALIBRATION.json", self.calibration)

    def _check_cpu_calibration(self, samples):
        if len(samples) < 3:
            raise ValueError("configure requires >=3 distinct MuJoCo FK calibration samples")
        qs = np.stack([arm_array(s["q_arm"]) for s in samples])
        if len(np.unique(np.round(qs, decimals=5), axis=0)) < 3 or np.max(np.ptp(qs, axis=0)) < .02:
            raise ValueError("FK calibration samples do not contain distinct joint configurations")
        report = []
        for sample in samples:
            actual = self.base @ self.robot.fk(arm_array(sample["q_arm"]), self.fingers)["control_site"]
            expected = transform(sample["world_T_eef"])
            p = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
            a = float(Rotation.from_matrix(actual[:3, :3].T @ expected[:3, :3]).magnitude())
            report.append(dict(position_m=p, rotation_rad=a))
        passed = all(r["position_m"] <= .0005 and r["rotation_rad"] <= .002 for r in report)
        self.calibration = dict(cpu=report, passed=passed, position_tolerance_m=.0005,
                                rotation_tolerance_rad=.002, frame="MuJoCo grip_site",
                                model="generated from installed robosuite Panda/PandaGripper MJCF")
        write_json(self.output_dir / "CALIBRATION_CPU.json", self.calibration)
        if not passed:
            raise ValueError(f"Generated MJCF/URDF FK mismatch: {report}")

    def fk_world(self, path):
        tensor = self.torch.as_tensor(np.asarray(path), device="cuda", dtype=self.torch.float32)
        state = self.motion_gen.kinematics.get_state(tensor)
        p = state.ee_position.detach().cpu().numpy()
        quat = state.ee_quaternion.detach().cpu().numpy()
        transforms = np.repeat(np.eye(4)[None], len(p), axis=0)
        transforms[:, :3, :3] = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
        transforms[:, :3, 3] = p
        return self.base[None] @ transforms

    def _check_gpu_calibration(self, samples):
        actual = self.fk_world(np.stack([arm_array(s["q_arm"]) for s in samples]))
        report = []
        for pose, sample in zip(actual, samples):
            expected = transform(sample["world_T_eef"])
            report.append(dict(position_m=float(np.linalg.norm(pose[:3, 3]-expected[:3, 3])),
                rotation_rad=float(Rotation.from_matrix(pose[:3, :3].T @ expected[:3, :3]).magnitude())))
        self.calibration["curobo"] = report
        self.calibration["passed"] = self.calibration["passed"] and all(
            r["position_m"] <= .0005 and r["rotation_rad"] <= .002 for r in report)
        if not self.calibration["passed"]:
            write_json(self.output_dir / "CALIBRATION.json", self.calibration)
            raise ValueError(f"cuRobo FK mismatch: {report}")

    def build_world(self, scene, allow_target_contact=False, ignore_obstacles=()):
        if not np.allclose(transform(scene["world_T_base"]), self.base, atol=1e-7):
            raise ValueError("Robot base changed; configure a new planner")
        ignored = set(ignore_obstacles)
        if len(ignored) != len(ignore_obstacles):
            raise ValueError("Duplicate ignore_obstacles")
        obstacles = list(scene.get("obstacles", []))
        known = {g["name"] for g in obstacles}
        if not ignored <= known:
            raise ValueError(f"Unknown ignored obstacles: {sorted(ignored-known)}")
        include_target = self.attached is None and not allow_target_contact
        target = scene.get("target", {})
        if include_target:
            obstacles.extend(target.get("geometries", [target] if "type" in target else []))
        data = {"cuboid": {}, "mesh": {}}
        approximations = []
        names = set()
        for geometry in obstacles:
            name = str(geometry["name"])
            if name in ignored:
                continue
            if name in names:
                raise ValueError(f"Duplicate collision geometry: {name}")
            names.add(name)
            t = self.base_inverse @ transform(geometry["pose"])
            kind = geometry["type"]
            if kind == "mesh":
                path = Path(geometry["mesh_path"]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                entry = dict(file_path=str(path), pose=pose_list(t))
                if "scale" in geometry:
                    entry["scale"] = geometry["scale"]
                data["mesh"][name] = entry
            else:
                if kind == "box":
                    dims = np.asarray(geometry["dims"], float)
                elif kind == "sphere":
                    dims = np.repeat(2*float(geometry["radius"]), 3)
                    approximations.append(dict(name=name, source=kind, representation="conservative enclosing cuboid"))
                elif kind == "cylinder":
                    r = float(geometry["radius"])
                    dims = np.array([2*r, 2*r, float(geometry["length"])])
                    approximations.append(dict(name=name, source=kind, representation="conservative enclosing cuboid"))
                else:
                    raise ValueError(f"Unsupported collision geometry type: {kind}")
                if dims.shape != (3,) or not np.isfinite(dims).all() or (dims <= 0).any():
                    raise ValueError(f"Invalid collision dimensions: {name}")
                data["cuboid"][name] = dict(dims=dims.tolist(), pose=pose_list(t))
        report = dict(obstacle_names=sorted(names), target_in_world=include_target,
            target_attached=self.attached is not None, ignored_obstacles=sorted(ignored),
            allow_target_contact=bool(allow_target_contact), approximations=approximations,
            target_contact_limitation=("whole target removed from optimizer/checker; execution must monitor arm/palm collisions"
                                       if allow_target_contact and self.attached is None else None),
            robot_collision_model="conservative per-geom AABB sphere cover; fixed base omitted",
            path_validation="sampled cuRobo sphere-world/self constraints, not a continuous exact-mesh certificate")
        return data, report

    def sync(self, scene, allow_target_contact=False, ignore_obstacles=()):
        fingers = finger_values(scene)
        if not np.allclose(fingers, self.fingers, atol=1e-6):
            self.fingers = fingers
            self.motion_gen.update_locked_joints(dict(zip(FINGER_NAMES, map(float, fingers))), copy.deepcopy(self.config))
        self.scene = scene
        self.world_data, self.world_report = self.build_world(scene, allow_target_contact, ignore_obstacles)
        self.motion_gen.update_world(self.WorldConfig.from_dict(self.world_data))
        if self.attached is not None:
            self._apply_attachment()
        current = self.fk_world(arm_array(scene["q_arm"])[None])[0]
        expected = transform(scene["world_T_eef"])
        p = float(np.linalg.norm(current[:3, 3]-expected[:3, 3]))
        a = float(Rotation.from_matrix(current[:3, :3].T @ expected[:3, :3]).magnitude())
        if p > .002 or a > .01:
            raise ValueError(f"Live control-site FK mismatch: {p}m {a}rad")

    def _apply_attachment(self):
        spheres = np.zeros((self.attachment_slots, 4), np.float32)
        spheres[:, 3] = -10.
        spheres[:len(self.attached)] = self.attached
        self.motion_gen.attach_spheres_to_robot(sphere_tensor=self.torch.as_tensor(spheres, device="cuda"), link_name="attached_object")

    def attach(self, scene):
        if self.attached is not None:
            raise ValueError("Already carrying a target; detach explicitly before attaching again")
        self.sync(scene)
        self.attached = target_spheres(scene["target"], scene["world_T_eef"], self.attachment_slots)
        self.attachment_target_name = scene["target"]["name"]
        write_json(self.output_dir / "ATTACHMENT.json", dict(target_name=self.attachment_target_name,
            spheres_control_site=self.attached.tolist(), attachment_slots=self.attachment_slots,
            world_T_eef=scene["world_T_eef"], world_T_object=scene["target"].get("world_T_object"),
            representation="complete per-geometry AABB grid sphere cover"))
        self._apply_attachment()
        self.sync(scene)
        return dict(success=True, attached_target=self.attachment_target_name, sphere_count=len(self.attached),
                    representation="conservative AABB sphere cover per target geom", contact_policy=self.world_report)

    def detach(self):
        self.motion_gen.detach_object_from_robot()
        self.attached = None
        self.attachment_target_name = None
        return dict(success=True)

    def validate_path(self, q):
        self._enable_collision_costs()
        tensor = self.torch.as_tensor(q, device="cuda", dtype=self.torch.float32)
        chunks = []
        # Avoid using a single-state .item() API for batched trajectories.
        for start in range(0, len(q), 128):
            metrics = self.motion_gen.rollout_fn.rollout_constraint(tensor[start:start+128, None, :], use_batch_env=False)
            chunks.append(metrics.feasible.detach().cpu().numpy().reshape(-1))
        feasible = np.concatenate(chunks).astype(bool)
        return dict(valid=bool(feasible.all()), samples=len(q), invalid_indices=np.flatnonzero(~feasible).tolist())

    def _enable_collision_costs(self):
        # cuRobo 0.7.8's check_start_state can return WORLD_COLLISION before
        # restoring its temporarily disabled self-collision cost.
        self.motion_gen.rollout_fn.primitive_collision_constraint.enable_cost()
        self.motion_gen.rollout_fn.robot_self_collision_constraint.enable_cost()

    def world_collision_diagnostics(self, scene):
        return sphere_world_diagnostics(self.robot, scene, self.world_data, attached=self.attached,
            robot_sphere_buffer=float(self.config["kinematics"].get("collision_sphere_buffer", 0)))

    def plan_pose(self, request, deadline=None):
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        scene = request["scene"]
        self.sync(scene, bool(request.get("allow_target_contact", False)), request.get("ignore_obstacles", []))
        out = Path(request.get("out_dir", self.output_dir / "last_plan")).resolve()
        out.mkdir(parents=True, exist_ok=True)
        goal = transform(request["goal_pose"])
        local_goal = self.base_inverse @ goal
        start = arm_array(scene["q_arm"])
        joint_state = JointState.from_position(self.torch.as_tensor(start[None], device="cuda", dtype=self.torch.float32), joint_names=ARM_NAMES)
        self._enable_collision_costs()
        try:
            start_valid, start_status = self.motion_gen.check_start_state(joint_state)
        finally:
            self._enable_collision_costs()
        policy = self.world_report
        write_json(out / "WORLD.json", self.world_data)
        write_json(out / "SCENE.json", scene)
        write_json(out / "CONTACT_POLICY.json", policy)
        if not start_valid:
            reply = dict(success=False, reason="invalid_start_state", status=str(start_status), contact_policy=policy)
            if "JOINT_LIMIT" in str(start_status):
                limits = np.asarray(self.robot.arm_limits, dtype=float)
                reply["joint_limit_diagnostics"] = dict(measured_q=start.tolist(), limits=limits.tolist(),
                    violation_rad=np.maximum(np.maximum(limits[:, 0]-start, start-limits[:, 1]), 0).tolist(),
                    recovery="Physically recover an interior state; do not clip the measured input")
            if "WORLD" in str(start_status):
                try:
                    diagnostics = self.world_collision_diagnostics(scene)
                    write_json(out / "START_COLLISION_DIAGNOSTICS.json", diagnostics)
                    reply["world_collision_diagnostics"] = diagnostics
                except Exception:
                    reply["world_collision_diagnostics_error"] = traceback.format_exc()
            write_json(out / "RESULT.json", reply)
            return reply
        timeout = float(request.get("timeout_s", 15.))
        if not np.isfinite(timeout) or not .1 <= timeout <= 120:
            raise ValueError("timeout_s must be in [0.1,120]")
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
            if timeout < .1:
                reply = dict(success=False, reason="selection_budget_exhausted", contact_policy=policy)
                write_json(out / "RESULT.json", reply)
                return reply
        started = time.monotonic()
        write_json(out / "STATUS.json", dict(stage="trajectory_optimization", timeout_s=timeout))
        try:
            result = self.motion_gen.plan_single(joint_state, Pose.from_list(pose_list(local_goal), self.device),
                # The identical start state was checked immediately above.
                MotionGenPlanConfig(max_attempts=2, timeout=timeout, enable_graph_attempt=1,
                                    check_start_validity=False))
        finally:
            self._enable_collision_costs()
        self.torch.cuda.synchronize()
        reply = dict(success=bool(result.success.item()), status=str(result.status),
                     wall_s=time.monotonic()-started, contact_policy=policy)
        if deadline is not None and time.monotonic() >= deadline:
            reply.update(solver_success=reply["success"], success=False,
                         reason="selection_budget_exhausted")
        if not reply["success"]:
            reply.setdefault("reason", "curobo_plan_failed")
        else:
            q = densify(result.get_interpolated_plan().position.detach().cpu().numpy())
            # Include actual current state explicitly for the controller.
            if np.max(np.abs(q[0]-start)) > .001:
                q = densify(np.concatenate([start[None], q]))
            write_json(out / "STATUS.json", dict(stage="path_collision_validation", samples=len(q)))
            checked = self.validate_path(q)
            poses = self.fk_world(q)
            final_p = float(np.linalg.norm(poses[-1, :3, 3]-goal[:3, 3]))
            final_a = float(Rotation.from_matrix(poses[-1, :3, :3].T @ goal[:3, :3]).magnitude())
            reply.update(path_check=checked, final_position_error_m=final_p, final_rotation_error_rad=final_a)
            if not checked["valid"] or final_p > .003 or final_a > .02:
                reply.update(success=False, reason="interpolated_path_validation_failed")
            else:
                path = out / "PATH.npz"
                np.savez_compressed(path, q_arm=q, world_T_eef=poses,
                                    gripper_width=np.float32(self.fingers[0]-self.fingers[1]))
                reply.update(path_file=str(path), waypoints=len(q), joint_names=ARM_NAMES,
                             attached_target=self.attachment_target_name)
        write_json(out / "RESULT.json", reply)
        write_json(out / "STATUS.json", dict(stage="completed", success=reply["success"]))
        return reply

    def line(self, request, deadline=None):
        """One batch IK call, then joint interpolation and full collision checks.

        This avoids launching a complete trajectory optimizer for every short
        Cartesian approach segment. Failed IK, branch jumps, curved FK paths,
        and any sampled collision are rejected rather than silently relaxed.
        """
        from curobo.types.math import Pose
        out = Path(request["out_dir"]).resolve()
        out.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        self.sync(request["scene"], bool(request.get("allow_target_contact", False)))
        self._enable_collision_costs()
        policy = self.world_report
        write_json(out / "WORLD.json", self.world_data)
        write_json(out / "CONTACT_POLICY.json", policy)
        if deadline is not None and deadline-time.monotonic() < .1:
            reply = dict(success=False, reason="selection_budget_exhausted", contact_policy=policy)
            write_json(out / "RESULT.json", reply)
            return reply
        q0 = arm_array(request["scene"]["q_arm"])
        start = transform(request["scene"]["world_T_eef"])
        goal = transform(request["goal_pose"])
        delta = goal[:3, 3]-start[:3, 3]
        distance = float(np.linalg.norm(delta))
        rotation = Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).as_rotvec()
        n = max(2, int(np.ceil(distance/.005))+1)
        parameters = np.linspace(0, 1, n)[1:]
        targets = np.repeat(np.eye(4)[None], n-1, axis=0)
        targets[:, :3, 3] = start[:3, 3] + parameters[:, None]*delta
        targets[:, :3, :3] = Rotation.from_rotvec(parameters[:, None]*rotation).as_matrix() @ start[:3, :3]
        local = self.base_inverse[None] @ targets
        quats = Rotation.from_matrix(local[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        poses = Pose(position=self.torch.as_tensor(local[:, :3, 3], device="cuda", dtype=self.torch.float32),
                     quaternion=self.torch.as_tensor(quats, device="cuda", dtype=self.torch.float32))
        seeds = self.torch.as_tensor(np.repeat(q0[None], n-1, axis=0), device="cuda", dtype=self.torch.float32)
        write_json(out / "STATUS.json", dict(stage="batch_ik", cartesian_goals=n-1, seeds_per_goal=4))
        result = self.motion_gen.ik_solver.solve_batch(poses, retract_config=seeds, seed_config=seeds[:, None, :],
                                                       num_seeds=4, return_seeds=4)
        self.torch.cuda.synchronize()
        reply = dict(success=False, contact_policy=policy, ik_wall_s=time.monotonic()-started,
                     cartesian_waypoints=n, method="single batch IK + dense joint interpolation + collision constraints")
        if deadline is not None and time.monotonic() >= deadline:
            reply["reason"] = "selection_budget_exhausted"
        else:
            success = result.success.detach().cpu().numpy().reshape(n-1, -1)
            solutions = result.solution.detach().cpu().numpy().reshape(n-1, 4, 7)
            knots = [q0]
            for index, (valid, alternatives) in enumerate(zip(success, solutions)):
                eligible = np.flatnonzero(valid)
                if not len(eligible):
                    reply.update(reason="cartesian_waypoint_ik_failed", waypoint=index+1)
                    break
                nearest = eligible[np.argmin(np.linalg.norm(alternatives[eligible]-knots[-1], axis=1))]
                q_next = alternatives[nearest]
                if np.max(np.abs(q_next-knots[-1])) > .25:
                    reply.update(reason="cartesian_ik_joint_branch_jump", waypoint=index+1)
                    break
                knots.append(q_next)
            if len(knots) == n:
                q = densify(knots, maximum_joint_step=.015)
                write_json(out / "STATUS.json", dict(stage="path_collision_validation", samples=len(q)))
                checked = self.validate_path(q)
                world_poses = self.fk_world(q)
                if distance > 1e-8:
                    projection = np.clip((world_poses[:, :3, 3]-start[:3, 3]) @ delta/(distance*distance), 0, 1)
                    line_positions = start[:3, 3]+projection[:, None]*delta
                    deviation = np.linalg.norm(world_poses[:, :3, 3]-line_positions, axis=1)
                    progress = (world_poses[:, :3, 3]-start[:3, 3]) @ delta/distance
                else:
                    projection = np.zeros(len(q))
                    deviation = np.linalg.norm(world_poses[:, :3, 3]-start[:3, 3], axis=1)
                    progress = np.zeros(len(q))
                expected_rotations = Rotation.from_rotvec(projection[:, None]*rotation).as_matrix() @ start[:3, :3]
                angles = Rotation.from_matrix(expected_rotations.transpose(0, 2, 1) @ world_poses[:, :3, :3]).magnitude()
                p_error = float(np.linalg.norm(world_poses[-1, :3, 3]-goal[:3, 3]))
                a_error = float(Rotation.from_matrix(world_poses[-1, :3, :3].T @ goal[:3, :3]).magnitude())
                reply.update(path_check=checked, max_line_deviation_m=float(deviation.max()),
                    max_orientation_deviation_rad=float(angles.max()),
                    minimum_progress_increment_m=float(np.min(np.diff(progress), initial=0)),
                    final_position_error_m=p_error, final_rotation_error_rad=a_error)
                good = checked["valid"] and deviation.max() <= .003 and angles.max() <= .015 \
                    and np.min(np.diff(progress), initial=0) >= -.001 and p_error <= .003 and a_error <= .015
                if deadline is not None and time.monotonic() >= deadline:
                    reply["reason"] = "selection_budget_exhausted"
                elif not good:
                    reply["reason"] = "cartesian_path_geometry_or_tracking_failed"
                else:
                    path = out / "PATH.npz"
                    np.savez_compressed(path, q_arm=q, world_T_eef=world_poses)
                    reply.update(success=True, path_file=str(path), waypoints=len(q))
        reply["wall_s"] = time.monotonic()-started
        write_json(out / "RESULT.json", reply)
        write_json(out / "STATUS.json", dict(stage="completed", success=reply["success"]))
        return reply

    def select(self, request):
        started = time.monotonic()
        total_budget = float(request.get("selection_timeout_s", 180.))
        if not np.isfinite(total_budget) or not 1 <= total_budget <= 900:
            raise ValueError("selection_timeout_s must be in [1,900]")
        deadline = started + total_budget
        candidates = request["candidates"]
        out = Path(request.get("out_dir", self.output_dir / "selection")).resolve()
        out.mkdir(parents=True, exist_ok=True)
        maximum = int(request.get("max_candidates", 8))
        if not 1 <= maximum <= 200:
            raise ValueError("max_candidates must be 1..200")
        distance = float(request.get("approach_distance_m", .08))
        if not .01 <= distance <= .3:
            raise ValueError("approach_distance_m must be .01..0.3")
        order = sorted(range(len(candidates)), key=lambda i: -float(candidates[i].get("score", 0)))[:maximum]
        attempts = []
        budget_exhausted = False
        for index in order:
            if deadline - time.monotonic() < .1:
                budget_exhausted = True
                break
            candidate = candidates[index]
            grasp = transform(candidate["world_T_eef"])
            pre = grasp.copy()
            pre[:3, 3] -= distance*pre[:3, 2]
            prefix = out / f"candidate_{index:03d}"
            plan_request = dict(command="plan_pose", scene=request["scene"], goal_pose=pre.tolist(),
                                out_dir=str(prefix / "approach"), timeout_s=request.get("timeout_s", 12.))
            approach = self.plan_pose(plan_request, deadline=deadline)
            attempt = dict(candidate_index=index, approach=approach)
            attempts.append(attempt)
            if not approach["success"]:
                if approach.get("reason") == "selection_budget_exhausted":
                    budget_exhausted = True
                    break
                continue
            with np.load(approach["path_file"], allow_pickle=False) as a:
                q_last = a["q_arm"][-1]
                pose_last = a["world_T_eef"][-1]
            future_scene = copy.deepcopy(request["scene"])
            future_scene.update(q_arm=q_last.tolist(), world_T_eef=pose_last.tolist())
            descend = self.line(dict(scene=future_scene, goal_pose=grasp.tolist(),
                out_dir=str(prefix / "descend"), allow_target_contact=bool(request.get("allow_target_contact", False))),
                deadline=deadline)
            attempt["descend"] = descend
            if not descend["success"]:
                if descend.get("reason") == "selection_budget_exhausted":
                    budget_exhausted = True
                    break
                continue
            descend_path = Path(descend["path_file"])
            reply = dict(success=True, selected=candidate, selected_index=index,
                approach_path_file=approach["path_file"], descend_path_file=str(descend_path),
                path_file=approach["path_file"], pregrasp_world=pre.tolist(), grasp_world=grasp.tolist(),
                attempts=attempts, descend_contact_policies=[descend.get("contact_policy")],
                selection_timeout_s=total_budget, selection_wall_s=time.monotonic()-started,
                limitation="Sampled collision validation; execution must recheck actual MuJoCo geometry and OSC tracking")
            write_json(out / "SELECTION.json", reply)
            return reply
        budget_exhausted = budget_exhausted or time.monotonic() >= deadline
        reply = dict(success=False, reason="selection_budget_exhausted" if budget_exhausted else "no_executable_grasp",
                     attempts=attempts, attempted_indices=[a["candidate_index"] for a in attempts],
                     selection_timeout_s=total_budget, selection_wall_s=time.monotonic()-started,
                     budget_type="soft deadline; in-flight non-preemptible GPU stages may overrun")
        write_json(out / "SELECTION.json", reply)
        return reply


class Handler:
    def __init__(self, assets_dir=None):
        self.planner = None
        self.assets_dir = assets_dir

    def __call__(self, request):
        command = request.get("command")
        if command == "configure":
            # Discard stale model/world/attachment when starting a new task.
            self.planner = None
            self.planner = BridgePlanner(request, assets_dir=self.assets_dir)
            return dict(success=True, calibration=self.planner.calibration,
                        robot_urdf=str(self.planner.robot.urdf_path), command=command)
        if command == "status":
            return dict(success=True, configured=self.planner is not None,
                        attached_target=None if self.planner is None else self.planner.attachment_target_name)
        if self.planner is None:
            raise RuntimeError("configure the planner before planning")
        if command in ("plan", "plan_pose"):
            return self.planner.plan_pose(request)
        if command == "select":
            return self.planner.select(request)
        if command == "attach":
            return self.planner.attach(request["scene"])
        if command == "detach":
            return self.planner.detach()
        raise ValueError(f"Unknown planner command: {command}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--ready", type=Path)
    parser.add_argument("--grasp-repo", type=Path, help="Accepted for bridge launcher compatibility; no Isaac code is imported")
    parser.add_argument("--robosuite-assets", type=Path)
    parser.add_argument("--request", type=Path, help="Run one JSON request (use --serve for a persistent planner)")
    parser.add_argument("--response", type=Path)
    args = parser.parse_args()
    handler = Handler(args.robosuite_assets)
    if args.serve:
        if args.ready is None:
            parser.error("--serve requires --ready")
        from graspgen_libero_bridge.common.rpc import serve
        serve(handler, args.ready)
    elif args.request:
        try:
            result = handler(json.loads(args.request.read_text()))
        except Exception:
            result = dict(success=False, error=traceback.format_exc())
        if args.response:
            write_json(args.response, result)
        else:
            print(json.dumps(result, indent=2))
    else:
        parser.error("Use --serve or --request")


if __name__ == "__main__":
    main()
