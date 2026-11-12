"""Closed-loop native composite control with real-action accounting."""
from __future__ import annotations

import math
import time
import numpy as np

from graspgen_libero_bridge.common.frames import pose_error
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure, EpisodeStopped


class StableNativeGoal:
    """Require consecutive physical observations, never repeated reads or gaps."""
    def __init__(self, environment, seconds):
        self.environment = environment
        self.required = max(1, int(math.ceil(seconds * float(environment.env.control_freq))) + 1)
        self.last_step = None
        self.count = 0
        self.complete = False

    def __call__(self):
        step = self.environment.steps
        if step == self.last_step:
            return self.complete
        consecutive = self.last_step is not None and step == self.last_step + 1
        self.last_step = step
        if self.environment.check_native_success():
            self.count = self.count + 1 if consecutive else 1
        else:
            self.count = 0
        self.complete = self.count >= self.required
        return self.complete


class RoboCasaExecutor:
    def __init__(self, environment, recorder=None):
        self.environment = environment
        self.recorder = recorder
        self.policy = environment.config["policy"]
        self.gripper_command = -1.
        self.monitor = None

    def _step(self, action):
        result = self.environment.step(action)
        if self.monitor is not None:
            failures = self.monitor.check()
            if failures:
                raise EpisodeFailure("COLLISION_REJECTED", "Forbidden contact during physical execution",
                                     details={"contacts": failures})
        return result

    def move_to(self, goal, *, phase, gripper=None, timeout=None, joint_reference=None, stability_steps=3, stop_when=None,
                position_tolerance=None):
        env = self.environment
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (4, 4) or not np.isfinite(goal).all():
            raise ValueError("Expected finite target transform")
        if gripper is not None:
            self.gripper_command = float(gripper)
        if joint_reference is not None:
            reference = np.asarray(joint_reference, float)
            limits = np.asarray(env.robot_state()["joint_limits"])
            tolerance = self.policy.get("joint_soft_limit_tolerance_rad", .001)
            if (reference.shape != (7,) or not np.isfinite(reference).all()
                    or np.any(reference < limits[:, 0] - tolerance) or np.any(reference > limits[:, 1] + tolerance)):
                raise ValueError("Invalid planned null-space joint reference")
            # Interior controller reference; never clip observations or edit qpos.
            margin = self.policy.get("libero_v10_rules", {}).get("joint_reference_margin_rad",
                                         self.policy.get("joint_limit_margin_rad", .01))
            reference = np.clip(reference, limits[:, 0] + margin, limits[:, 1] - margin)
            env.arm.initial_joint = reference.copy()
        if self.recorder is not None:
            self.recorder.phase = phase
            self.recorder.decision("move_tcp", {"world_T_tcp": goal, "gripper": self.gripper_command,
                                   "nullspace_joint_reference": None if joint_reference is None else reference})
        if self.policy.get("rule_profile") == "libero-v10":
            from graspgen_libero_bridge.control.robocasa_v10_controller import move_with_v10
            return move_with_v10(self, goal, phase=phase, gripper=self.gripper_command,
                                 position_tolerance=position_tolerance or self.policy["position_tolerance_m"],
                                 stop_when=stop_when)
        deadline = env.clock() + min(timeout or self.policy["move_timeout_s"], env.remaining_seconds())
        consecutive = 0
        while True:
            if stop_when is not None and stop_when():
                return {"reached": False, "task_goal_reached": True, "step": env.steps,
                        "stop_reason": "native_goal_stable"}
            env.check_budget()
            state = env.robot_state()
            position, rotation = pose_error(goal, state["world_T_eef"])
            if np.linalg.norm(position) <= (position_tolerance or self.policy["position_tolerance_m"]) and np.linalg.norm(rotation) <= self.policy["rotation_tolerance_rad"]:
                consecutive += 1
                if consecutive >= stability_steps:
                    return {"reached": True, "position_error_m": float(np.linalg.norm(position)),
                            "rotation_error_rad": float(np.linalg.norm(rotation)), "step": env.steps}
            else:
                consecutive = 0
            if env.clock() >= deadline:
                raise EpisodeFailure("EXECUTION_TIMEOUT", "TCP did not converge during " + phase,
                                     details={"goal": goal.tolist(), "state": state,
                                              "position_error_m": float(np.linalg.norm(position)),
                                              "rotation_error_rad": float(np.linalg.norm(rotation))})
            action = env.layout.pose_action(goal, state["world_T_eef"], state["world_T_control_base"],
                                           self.gripper_command, gain=self.policy["pose_gain"])
            self._step(action)

    def ensure_joint_margin(self, phase, *, gripper=1):
        """v10 interior-start recovery through physical, native-checked actions."""
        env, local = self.environment, self.monitor
        rules = self.policy.get("libero_v10_rules", {})
        if not rules:
            return
        state = env.robot_state()
        q = np.asarray(state["q_arm"])
        low, high = np.asarray(state["joint_limits"]).T
        if np.max(np.maximum(low - q, q - high)) > self.policy["joint_soft_limit_tolerance_rad"]:
            raise EpisodeFailure("JOINT_LIMIT", "Measured joint excursion exceeds the bounded recovery tolerance")
        margin = rules["planning_joint_margin_rad"]
        if np.all((q >= low + margin) & (q <= high - margin)):
            return
        target = np.clip(q, low + rules["joint_reference_margin_rad"], high - rules["joint_reference_margin_rad"])
        checked = local.validate_path(np.linspace(q, target, 16))
        if not checked["valid"]:
            raise EpisodeFailure("JOINT_LIMIT", "Native joint-margin recovery preflight failed", details=checked)
        local.sync(); local.set_q(target)
        goal = local.tcp().copy()
        initial_pose = np.asarray(state["world_T_eef"])
        p, r = pose_error(goal, initial_pose)
        if np.linalg.norm(p) > .02 or np.linalg.norm(r) > .1:
            raise EpisodeFailure("JOINT_LIMIT", "Recovery exceeds the v10 TCP drift bounds")
        reference_margin = rules["joint_recovery_reference_margin_rad"]
        env.arm.initial_joint = np.clip(target, low + reference_margin, high - reference_margin)
        for _ in range(rules["joint_recovery_steps"]):
            current = env.robot_state()
            p, r = pose_error(initial_pose, current["world_T_eef"])
            if np.linalg.norm(p) > .02 or np.linalg.norm(r) > .1:
                break
            if self.recorder is not None:
                self.recorder.phase = phase + "/joint_recovery"
            action = env.layout.pose_action(goal, current["world_T_eef"], current["world_T_control_base"],
                                           gripper, gain=self.policy["pose_gain"])
            self._step(action)
            measured = np.asarray(env.robot_state()["q_arm"])
            if np.all((measured >= low + margin) & (measured <= high - margin)):
                return
        raise EpisodeFailure("JOINT_LIMIT", "Physical joint-margin recovery did not reach an interior state")

    def hold(self, seconds, *, gripper=None, phase="hold"):
        env = self.environment
        if gripper is not None:
            self.gripper_command = float(gripper)
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Hold duration must be nonnegative")
        goal = np.asarray(env.robot_state()["world_T_eef"])
        if self.recorder is not None:
            self.recorder.phase = phase
            self.recorder.decision("gripper_hold", {"seconds": seconds, "command": self.gripper_command,
                                                    "world_T_tcp": goal})
        for _ in range(int(math.ceil(seconds * float(env.env.control_freq)))):
            state = env.robot_state()
            action = env.layout.pose_action(goal, state["world_T_eef"], state["world_T_control_base"], self.gripper_command,
                                           gain=self.policy["pose_gain"])
            self._step(action)

    def follow_path(self, path_file, *, phase, gripper=None, stop_when=None):
        with np.load(path_file, allow_pickle=False) as path:
            poses = np.asarray(path["world_T_eef"], dtype=float)
            joints = np.asarray(path["q_arm"], dtype=float)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
            raise ValueError("Malformed planned TCP path")
        # Track a subsampled path with an angular/linear bound, retaining its endpoint.
        if joints.shape != (len(poses), 7):
            raise ValueError("Joint and TCP path lengths differ")
        if self.monitor is not None:
            measured = np.asarray(self.environment.robot_state()["q_arm"])
            difference = float(np.max(np.abs(joints[0] - measured)))
            if difference > .8:
                raise EpisodeFailure("JOINT_BRANCH_DEVIATION", "Measured branch differs from the path start",
                                     details={"joint_error_rad": difference})
            count = max(1, int(np.ceil(difference / .01)))
            prefix = np.linspace(measured, joints[0], count + 1)[:-1]
            preflight = self.monitor.validate_path(np.concatenate([prefix, joints]))
            if not preflight["valid"]:
                raise EpisodeFailure("PATH_PREFLIGHT_REJECTED", "Native execution preflight rejected " + phase,
                                     details=preflight)
        kept = [0]
        for index, pose in enumerate(poses[1:], 1):
            p, r = pose_error(pose, poses[kept[-1]])
            if np.linalg.norm(p) >= .012 or np.linalg.norm(r) >= .12 or np.max(np.abs(joints[index] - joints[kept[-1]])) >= .10:
                kept.append(index)
        if kept[-1] != len(poses) - 1:
            kept.append(len(poses) - 1)
        for n, index in enumerate(kept):
            outcome = self.move_to(poses[index], phase=phase + "/waypoint_" + str(n), gripper=gripper,
                         joint_reference=joints[index], stability_steps=3 if n == len(kept) - 1 else 1, stop_when=stop_when,
                         position_tolerance=.0025 if n == len(kept) - 1 else .004)
            if outcome.get("task_goal_reached"):
                return {"executed_waypoints": n, "source_waypoints": len(poses), "path_completed": False,
                        "task_goal_reached": True, "stop_reason": "native_goal_stable", "step": self.environment.steps}
            deviation = float(np.max(np.abs(np.asarray(self.environment.robot_state()["q_arm"]) - joints[index])))
            if deviation > .8:
                raise EpisodeFailure("JOINT_BRANCH_DEVIATION", "OSC diverged from the planned joint branch",
                                     details={"joint_error_rad": deviation, "waypoint_index": int(index)})
        return {"executed_waypoints": len(kept), "source_waypoints": len(poses), "path_completed": True,
                "task_goal_reached": False}
