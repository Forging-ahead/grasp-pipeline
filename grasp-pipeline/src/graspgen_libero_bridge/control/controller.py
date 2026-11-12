"""Closed-loop default OSC execution with MuJoCo contact/path checks."""
from pathlib import Path
import json
import numpy as np
from graspgen_libero_bridge.common.frames import pose_error, pose_to_action
from graspgen_libero_bridge.common.errors import ExecutionError
from graspgen_libero_bridge.planning.cartesian import _snapshot, _restore


def descendants(model, root):
    ids = {int(root)}
    for i in range(model.nbody):
        p = i
        while p:
            if p in ids:
                ids.add(i)
                break
            p = int(model.body_parentid[p])
    return ids


class ContactMonitor:
    def __init__(self, adapter, target):
        self.adapter, self.target = adapter, target
        model = adapter.env.sim.model
        self.robot_bodies = descendants(model, model.body_name2id("robot0_base"))
        self.target_bodies = descendants(model, adapter.task_env.obj_body_id[target])
        self.tolerance = .001
        self.held = False
        self.allowed_support_geoms = set()

    def set_held(self, held, allowed_support_geoms=()):
        self.held = bool(held)
        self.allowed_support_geoms = set(allowed_support_geoms)

    def check(self, allow_finger_target=False):
        sim = self.adapter.env.sim
        model, data = sim.model, sim.data
        failures = []
        for i in range(data.ncon):
            c = data.contact[i]
            if float(c.dist) >= -self.tolerance:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            if b1 in self.robot_bodies and b2 in self.robot_bodies:
                n1, n2 = model.body_id2name(b1) or "", model.body_id2name(b2) or ""
                w1, w2 = int(model.body_weldid[b1]), int(model.body_weldid[b2])
                parent1 = int(model.body_weldid[int(model.body_parentid[w1])])
                parent2 = int(model.body_weldid[int(model.body_parentid[w2])])
                if w1 != w2 and parent1 != w2 and parent2 != w1 and not ("finger" in n1 and "finger" in n2):
                    failures.append({"geom1": model.geom_id2name(g1), "geom2": model.geom_id2name(g2), "distance_m": float(c.dist), "self_collision": True})
            if self.held and b1 not in self.robot_bodies and b2 not in self.robot_bodies:
                if (b1 in self.target_bodies) != (b2 in self.target_bodies):
                    other_g = g2 if b1 in self.target_bodies else g1
                    if (model.geom_id2name(other_g) or f"geom_{other_g}") not in self.allowed_support_geoms:
                        failures.append({"geom1": model.geom_id2name(g1), "geom2": model.geom_id2name(g2), "distance_m": float(c.dist), "held_object_collision": True})
            if (b1 in self.robot_bodies) == (b2 in self.robot_bodies):
                continue
            robot_b, other_b = (b1, b2) if b1 in self.robot_bodies else (b2, b1)
            body_name = model.body_id2name(robot_b) or ""
            if body_name in ("robot0_base", "robot0_link0"):
                continue
            if allow_finger_target and other_b in self.target_bodies and "finger" in body_name:
                continue
            failures.append({"geom1": model.geom_id2name(g1), "geom2": model.geom_id2name(g2), "distance_m": float(c.dist)})
        return failures

    def validate_path(self, path, allow_finger_target=False, max_samples=1000):
        sim = self.adapter.env.sim
        saved = _snapshot(sim)
        robot = self.adapter.task_env.robots[0]
        failures = []
        relation = np.linalg.inv(np.asarray(self.adapter.state()["world_T_eef"])) @ self.adapter.target_pose(self.target)
        target_body = self.adapter.task_env.obj_body_id[self.target]
        joint_id = int(sim.model.body_jntadr[target_body])
        target_address = int(sim.model.jnt_qposadr[joint_id]) if joint_id >= 0 else None
        try:
            indexes = np.linspace(0, len(path)-1, min(len(path), max_samples)).astype(int)
            for i in indexes:
                sim.data.qpos[robot._ref_joint_pos_indexes] = path[i]
                sim.forward()
                if self.held and target_address is not None and int(sim.model.jnt_type[joint_id]) == 0:
                    from scipy.spatial.transform import Rotation
                    site = int(robot.eef_site_id)
                    eef = np.eye(4)
                    eef[:3, :3] = np.asarray(sim.data.site_xmat[site]).reshape(3, 3)
                    eef[:3, 3] = sim.data.site_xpos[site]
                    obj = eef @ relation
                    quat = Rotation.from_matrix(obj[:3, :3]).as_quat()[[3, 0, 1, 2]]
                    sim.data.qpos[target_address:target_address+7] = np.r_[obj[:3, 3], quat]
                    sim.forward()
                bad = self.check(allow_finger_target)
                if bad:
                    failures = [{"path_index": int(i), **x} for x in bad]
                    break
        finally:
            _restore(sim, saved)
            robot.controller.update(force=True)
            robot.controller.reset_goal()
            self.adapter.task_env._update_observables(force=True)
        return {"passed": not failures, "failures": failures, "samples": len(indexes)}


class Recorder:
    def __init__(self, adapter, output):
        self.adapter = adapter
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.states, self.actions, self.phases = [], [], []
        self.phase = "initial"
        self.snapshots = []
        self.writer = None
        self.capture()

    def image(self):
        obs = self.adapter.obs
        keys = [f"{name}_image" for name in ["agentview", "robot0_eye_in_hand"]]
        images = [np.asarray(obs[k], dtype=np.uint8) for k in keys if k in obs]
        if not images:
            return None
        import robosuite.macros as macros
        if macros.IMAGE_CONVENTION == "opengl":
            images = [im[::-1] for im in images]
        return np.concatenate(images, axis=1).copy()

    def capture(self, action=None):
        state = self.adapter.state()
        self.states.append(state)
        if action is not None:
            self.actions.append(np.asarray(action).copy())
            self.phases.append(self.phase)
        im = self.image()
        if im is not None:
            import cv2
            if self.writer is None:
                h, w = im.shape[:2]
                self.writer = cv2.VideoWriter(str(self.output / "rollout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                if not self.writer.isOpened():
                    raise RuntimeError("Could not create rollout MP4")
            self.writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            if not self.snapshots or self.snapshots[-1][0] != self.phase:
                self.snapshots.append((self.phase, im))
            else:
                self.snapshots[-1] = (self.phase, im)

    def save(self):
        if self.writer is not None:
            self.writer.release()
        states = self.states
        np.savez_compressed(self.output / "trajectory.npz",
                            q_arm=np.asarray([s["q_arm"] for s in states]),
                            q_fingers=np.asarray([s["q_fingers"] for s in states]),
                            world_T_eef=np.asarray([s["world_T_eef"] for s in states]),
                            actions=np.asarray(self.actions).reshape(-1, 7),
                            phases=np.asarray(self.phases))
        (self.output / "states.json").write_text(json.dumps(states, allow_nan=False))
        if self.snapshots:
            from PIL import Image, ImageDraw
            images = self.snapshots[-8:]
            w, h = images[0][1].shape[1], images[0][1].shape[0]
            canvas = Image.new("RGB", (w, (h+24)*len(images)), "white")
            draw = ImageDraw.Draw(canvas)
            for i, (phase, im) in enumerate(images):
                y = i*(h+24)
                draw.text((5, y+5), phase, fill="black")
                canvas.paste(Image.fromarray(im), (0, y+24))
            canvas.save(self.output / "preview.png")


class Controller:
    def __init__(self, adapter, recorder, target):
        self.adapter, self.recorder = adapter, recorder
        self.monitor = ContactMonitor(adapter, target)
        cfg = adapter.task_env.robots[0].controller
        if not bool(getattr(cfg, "use_delta", True)):
            raise ValueError("This OSC adapter requires delta control")
        self.position_scale = float(np.asarray(cfg.output_max)[0])
        self.rotation_scale = float(np.asarray(cfg.output_max)[3])
        self.events = []
        robot = adapter.task_env.robots[0]
        model = adapter.env.sim.model
        self.joint_limits = np.asarray([model.jnt_range[model.joint_name2id(n)]
                                       for n in robot.robot_model.joints], dtype=float)
        self.reference_margin = float(adapter.config.get("joint_reference_margin_rad", .02))
        if not np.isfinite(self.reference_margin) or self.reference_margin <= 0 or np.any(np.diff(self.joint_limits, axis=1)[:, 0] <= 2*self.reference_margin):
            raise ValueError("Invalid joint reference margin")

    def set_joint_reference(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected a finite seven-joint reference")
        # Set the *active* controller; update_initial_joints also resets its goal.
        reference = np.clip(q, self.joint_limits[:, 0]+self.reference_margin,
                            self.joint_limits[:, 1]-self.reference_margin)
        self.adapter.task_env.robots[0].controller.initial_joint = reference.copy()
        return reference

    def ensure_joint_margin(self, phase, gripper=1, allow_finger_target=True):
        """Physically recover an interior start. Never clip measured planner inputs."""
        q = np.asarray(self.adapter.state()["q_arm"])
        low, high = self.joint_limits.T
        if np.max(np.maximum(low-q, q-high)) > .005:
            raise ExecutionError("Joint excursion too large for margin recovery", code="joint_limit_excursion",
                                 category="joint_limit", phase=phase, details={"measured_q": q.tolist()})
        margin = float(self.adapter.config.get("planning_joint_margin_rad", .0001))
        if not np.isfinite(margin) or not 0 < margin < self.reference_margin:
            raise ValueError("Planning margin must be positive and below the reference margin")
        if np.all((q >= low+margin) & (q <= high-margin)):
            return
        initial = q.copy()
        pose = np.asarray(self.adapter.state()["world_T_eef"])
        recovery_margin = float(self.adapter.config.get("joint_recovery_reference_margin_rad", .2))
        if not self.reference_margin <= recovery_margin < float(np.min(high-low))/2:
            raise ValueError("Invalid recovery joint reference margin")
        # A pure nullspace objective cannot move a joint whose limit direction
        # lies in the task subspace. Preflight a small FK-consistent motion too.
        q_target = np.clip(q, low+self.reference_margin, high-self.reference_margin)
        goal = self.joint_recovery_goal(q, q_target, allow_finger_target)
        gp, gr = pose_error(goal, pose)
        if np.linalg.norm(gp) > .02 or np.linalg.norm(gr) > .1:
            raise ExecutionError("Recovery goal exceeds permitted pose drift", code="joint_recovery_pose_bound",
                                 category="joint_limit", phase=phase)
        self.set_joint_reference(np.clip(q_target, low+recovery_margin, high-recovery_margin))
        self.recorder.phase = phase + "_joint_recovery"
        for _ in range(int(self.adapter.config.get("joint_recovery_steps", 100))):
            current = np.asarray(self.adapter.state()["world_T_eef"])
            p, r = pose_error(pose, current)
            if np.linalg.norm(p) > .02 or np.linalg.norm(r) > .1:
                break
            self.step(pose_to_action(goal, current, gripper, self.position_scale,
                                    self.rotation_scale), allow_finger_target)
            q = np.asarray(self.adapter.state()["q_arm"])
            if np.all((q >= low+margin) & (q <= high-margin)):
                self.events.append(dict(phase=phase, joint_limit_recovery=True,
                                        measured_before=initial.tolist(), measured_after=q.tolist(),
                                        recovery_goal=goal.tolist(), measured_input_not_clipped=True))
                return
        raise ExecutionError("Could not recover an interior joint state", code="joint_margin_recovery_failed",
                             category="joint_limit", phase=phase,
                             details={"measured_q": q.tolist(), "limits": self.joint_limits.tolist()})

    def joint_recovery_goal(self, measured, target, allow_finger_target):
        sim = self.adapter.env.sim
        robot = self.adapter.task_env.robots[0]
        path = np.linspace(measured, target, 16)
        check = self.monitor.validate_path(path, allow_finger_target)
        if not check["passed"]:
            raise ExecutionError("Joint recovery preflight rejected", code="joint_recovery_collision",
                                 category="joint_limit", details=check)
        saved = _snapshot(sim)
        try:
            sim.data.qpos[robot._ref_joint_pos_indexes] = target
            sim.forward()
            goal = np.eye(4)
            goal[:3, :3] = np.asarray(sim.data.site_xmat[robot.eef_site_id]).reshape(3, 3)
            goal[:3, 3] = np.asarray(sim.data.site_xpos[robot.eef_site_id])
        finally:
            _restore(sim, saved)
            robot.controller.update(force=True)
            robot.controller.reset_goal()
        return goal

    def step(self, action, allow_finger_target=False):
        obs, reward, done, info = self.adapter.step(action)
        self.recorder.capture(action)
        bad = self.monitor.check(allow_finger_target)
        if bad:
            raise ExecutionError(f"Forbidden contact: {bad}", code="forbidden_contact",
                                 category="collision", phase=self.recorder.phase, details={"contacts": bad})
        q = np.asarray(self.adapter.state()["q_arm"])
        violation = np.maximum(self.joint_limits[:, 0]-q, q-self.joint_limits[:, 1])
        if np.max(violation) > .005:
            raise ExecutionError("Joint limit excursion", code="joint_limit_excursion",
                                 category="joint_limit", phase=self.recorder.phase,
                                 details={"measured_q": q.tolist(), "violation_rad": violation.tolist()})
        if self.adapter.step_count >= self.adapter.config.get("max_steps", 1000):
            raise ExecutionError("Episode step budget exhausted", code="episode_step_limit",
                                 category="budget", phase=self.recorder.phase)
        return obs, reward, done, info

    def move_to(self, desired, *, gripper=-1, phase="move", allow_finger_target=False,
                position_tolerance=.003, rotation_tolerance=.04, max_steps=100,stop_when=None):
        self.recorder.phase = phase
        desired = np.asarray(desired)
        if int(max_steps) != max_steps or max_steps < 0:
            raise ValueError("Waypoint step budget must be a nonnegative integer")
        integral = np.zeros(3)
        dt = 1. / float(self.adapter.config.get("control_freq", 20))
        ki = float(self.adapter.config.get("position_integral_gain", .8))
        integral_bound = float(self.adapter.config.get("position_integral_bound_m", .008))
        extra = int(self.adapter.config.get("place_tracking_extra_steps", 120)) if phase == "place" else 0
        extension_radius = float(self.adapter.config.get("place_tracking_extension_radius_m", .005)) if phase == "place" else .005
        if not 0 <= extra <= 240:
            raise ValueError("Invalid bounded place tracking extension")
        if not np.isfinite(extension_radius) or not .003 <= extension_radius <= .015:
            raise ValueError("Invalid bounded place tracking extension radius")
        extended = False
        errors = []
        if not np.isfinite(ki) or ki < 0 or not 0 <= integral_bound <= .01:
            raise ValueError("Invalid bounded position integral settings")
        for i in range(max_steps + extra + 1):
            current = np.asarray(self.adapter.state()["world_T_eef"])
            p, r = pose_error(desired, current)
            errors.append(float(np.linalg.norm(p)))
            if stop_when is not None and stop_when():
                self.events.append(dict(phase=phase,event='observed_subgoal_reached',actions_used=i,
                    position_error_m=errors[-1],rotation_error_rad=float(np.linalg.norm(r))))
                return
            if np.linalg.norm(p) <= position_tolerance and np.linalg.norm(r) <= rotation_tolerance:
                if extended:
                    self.events.append(dict(phase=phase,event="near_goal_tracking_extension_completed",
                        original_budget=max_steps,extra_steps_used=i-max_steps,position_error_m=errors[-1],
                        unchanged_position_tolerance_m=position_tolerance))
                return
            # Check after the last action too: arrival on action N is not timeout.
            if i == max_steps and extra:
                improving = len(errors) >= 21 and errors[-21]-errors[-1] >= .00005
                extended = (errors[-1] <= extension_radius and np.linalg.norm(r) <= rotation_tolerance and improving)
                self.events.append(dict(phase=phase,event="near_goal_tracking_extension_check",
                    enabled=extended,original_budget=max_steps,extra_budget=extra,
                    extension_radius_m=extension_radius,unchanged_position_tolerance_m=position_tolerance,
                    position_error_m=errors[-1],progress_over_20_steps_m=errors[-21]-errors[-1] if len(errors)>=21 else None))
            if i == max_steps + extra or (i == max_steps and not extended):
                break
            integral = np.clip(integral + ki * p * dt, -integral_bound, integral_bound)
            command = desired.copy()
            command[:3, 3] += integral
            action = pose_to_action(command, current, gripper, self.position_scale, self.rotation_scale)
            self.step(action, allow_finger_target)
        raise ExecutionError(f"OSC target timeout in {phase}: position={np.linalg.norm(p):.5f}, rotation={np.linalg.norm(r):.4f}",
                             code="waypoint_timeout", category="control", phase=phase,
                             details={"position_error_m": float(np.linalg.norm(p)),
                                      "rotation_error_rad": float(np.linalg.norm(r)), "max_steps": max_steps,
                                      "extra_budget_used": extended,"recent_errors_m": errors[-30:]})

    def hold(self, steps, gripper, phase, allow_finger_target=False):
        self.recorder.phase = phase
        pose = np.asarray(self.adapter.state()["world_T_eef"])
        for _ in range(steps):
            current = np.asarray(self.adapter.state()["world_T_eef"])
            self.step(pose_to_action(pose, current, gripper, self.position_scale, self.rotation_scale), allow_finger_target)

    def follow_path(self, path_file, *, gripper=-1, phase="path", allow_finger_target=False):
        with np.load(path_file, allow_pickle=False) as d:
            q, poses = d["q_arm"].copy(), d["world_T_eef"].copy()
        check = self.monitor.validate_path(q, allow_finger_target)
        self.events.append({"phase": phase, "path_file": str(path_file), "mujoco_preflight": check})
        if not check["passed"]:
            raise ExecutionError(f"MuJoCo path preflight rejected {phase}: {check['failures']}",
                                 code="path_preflight_rejected", category="planning", phase=phase, details=check)
        selected = [0]
        for i in range(1, len(poses)):
            p, r = pose_error(poses[i], poses[selected[-1]])
            if np.linalg.norm(p) >= .012 or np.linalg.norm(r) >= .12:
                selected.append(i)
        if selected[-1] != len(poses)-1:
            selected.append(len(poses)-1)
        maximum_q_error = 0.
        for i in selected:
            self.set_joint_reference(q[i])
            self.move_to(poses[i], gripper=gripper, phase=phase, allow_finger_target=allow_finger_target,
                         position_tolerance=.004 if i != selected[-1] else .0025,
                         rotation_tolerance=.05, max_steps=int(self.adapter.config.get("waypoint_max_steps", 120)))
            actual = np.asarray(self.adapter.state()["q_arm"])
            maximum_q_error = max(maximum_q_error, float(np.max(np.abs(actual-q[i]))))
            self.events[-1]["maximum_joint_tracking_error_rad"] = maximum_q_error
            if maximum_q_error > .8:
                raise ExecutionError(f"OSC diverged from planned joint branch: {maximum_q_error:.3f} rad",
                                     code="joint_branch_deviation", category="control", phase=phase,
                                     details={"joint_error_rad": maximum_q_error, "waypoint_index": int(i)})
        self.events[-1]["maximum_joint_tracking_error_rad"] = maximum_q_error
