"""Bind the unchanged v10 feedback controller to native composite actions."""
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.common.frames import pose_error
from graspgen_libero_bridge.control.controller import Controller


class NativeGoalReached(Exception):
    pass


class V10MotionBinding:
    move_to = Controller.move_to

    def __init__(self, executor, phase, stop_when):
        self.executor, self.environment = executor, executor.environment
        self.phase, self.stop_when = phase, stop_when
        config = dict(executor.policy["libero_v10_rules"], control_freq=executor.environment.env.control_freq)
        self.adapter = SimpleNamespace(config=config, state=self.state)
        self.recorder = SimpleNamespace(phase=phase)
        self.position_scale = float(executor.environment.layout.arm.output_max[0])
        self.rotation_scale = float(executor.environment.layout.arm.output_max[3])
        self.events = []

    def state(self):
        if self.stop_when is not None and self.stop_when():
            raise NativeGoalReached()
        return self.environment.robot_state()

    def step(self, action, allow_finger_target=False):
        # The original controller emits a normalized world-frame seven-vector.
        # Reconstruct its commanded pose, then let the *native* action layout
        # transform that delta into the actual arm controller's input frame.
        state = self.environment.robot_state()
        current = np.asarray(state["world_T_eef"])
        commanded = current.copy()
        commanded[:3, 3] += np.asarray(action[:3]) * self.position_scale
        commanded[:3, :3] = Rotation.from_rotvec(np.asarray(action[3:6]) * self.rotation_scale).as_matrix() @ current[:3, :3]
        native = self.environment.layout.pose_action(commanded, current, state["world_T_control_base"], float(action[6]), gain=1.)
        return self.executor._step(native)


def move_with_v10(executor, goal, *, phase, gripper, position_tolerance, stop_when):
    binding = V10MotionBinding(executor, phase, stop_when)
    try:
        binding.move_to(goal, gripper=gripper, phase=phase.split("/")[0],
                        allow_finger_target=bool(executor.monitor and executor.monitor.allow_finger_target),
                        position_tolerance=position_tolerance,
                        rotation_tolerance=executor.policy["rotation_tolerance_rad"],
                        max_steps=executor.policy["libero_v10_rules"]["waypoint_max_steps"])
    except NativeGoalReached:
        return {"reached": False, "task_goal_reached": True, "step": executor.environment.steps,
                "stop_reason": "native_goal_stable"}
    finally:
        if binding.events:
            from graspgen_libero_bridge.adapters.robocasa.native import write_json
            output = executor.environment.output / "v10_control_events" / (str(executor.environment.steps) + ".json")
            write_json(output, {"phase": phase, "events": binding.events})
    state = executor.environment.robot_state()
    p, r = pose_error(goal, state["world_T_eef"])
    return {"reached": True, "position_error_m": float(np.linalg.norm(p)),
            "rotation_error_rad": float(np.linalg.norm(r)), "step": executor.environment.steps,
            "controller": "unchanged_libero_v10_Controller.move_to"}
