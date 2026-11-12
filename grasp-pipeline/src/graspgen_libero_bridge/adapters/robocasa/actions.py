"""Read the native composite action layout and preserve all six arm coordinates."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value, size, label):
    result = np.broadcast_to(np.asarray(value, dtype=float), (size,)).copy()
    if not np.isfinite(result).all():
        raise ValueError(label + " contains nonfinite values")
    return result


def physical_to_input(values, controller):
    values = np.asarray(values, dtype=float)
    n = len(values)
    imin = _vector(controller.input_min, n, "input_min")
    imax = _vector(controller.input_max, n, "input_max")
    omin = _vector(controller.output_min, n, "output_min")
    omax = _vector(controller.output_max, n, "output_max")
    if np.any(imax <= imin) or np.any(omax <= omin):
        raise ValueError("Invalid controller scaling range")
    return np.clip((values - omin) / (omax - omin) * (imax - imin) + imin, imin, imax)


@dataclass
class ActionLayout:
    low: np.ndarray
    high: np.ndarray
    slices: dict
    arm: object
    torso: object
    mode_index: int

    @classmethod
    def from_env(cls, env):
        if len(env.robots) != 1:
            raise ValueError("RoboCasa integration requires one PandaOmron")
        controller = env.robots[0].composite_controller
        parts = controller.part_controllers
        slices = {name: tuple(map(int, bounds))
                  for name, bounds in controller._action_split_indexes.items()}
        required = {"right": 6, "right_gripper": 1, "base": 3, "torso": 1}
        if set(slices) != set(required):
            raise ValueError("Unrecognized native controller parts: " + str(sorted(slices)))
        low, high = (np.asarray(a, dtype=float).copy() for a in env.action_spec)
        if low.ndim != 1 or low.shape != high.shape or not np.isfinite([low, high]).all():
            raise ValueError("Invalid native action bounds")
        covered = []
        for name, (start, end) in slices.items():
            if end - start != required[name] or start < 0 or end > len(low):
                raise ValueError("Unexpected native action slice: " + name)
            covered.extend(range(start, end))
        missing = sorted(set(range(len(low))) - set(covered))
        if len(covered) != len(set(covered)) or missing != [len(low) - 1]:
            raise ValueError("Overlapping parts or missing hybrid base-mode field")
        arm = parts["right"]
        if getattr(arm, "input_type", None) not in {"delta", "absolute"}:
            raise ValueError("Unsupported arm input type")
        if getattr(arm, "input_ref_frame", None) not in {"base", "world"}:
            raise ValueError("Unsupported arm reference frame")
        return cls(low, high, slices, arm, parts["torso"], missing[0])

    def describe(self):
        return {"dimension": len(self.low), "slices": {k: list(v) for k, v in self.slices.items()},
                "low": self.low.tolist(), "high": self.high.tolist(),
                "arm_input_type": self.arm.input_type,
                "arm_reference_frame": self.arm.input_ref_frame,
                "arm_rotation": "axis_angle_radians", "hybrid_mode_index": self.mode_index,
                "arm_input_min": np.asarray(self.arm.input_min).tolist(),
                "arm_input_max": np.asarray(self.arm.input_max).tolist(),
                "arm_output_min": np.asarray(self.arm.output_min).tolist(),
                "arm_output_max": np.asarray(self.arm.output_max).tolist()}

    def assemble(self, arm, gripper, *, base_velocity=(0., 0., 0.), torso=0., mode=-1.):
        values = {"right": _vector(arm, 6, "arm"),
                  "right_gripper": _vector(gripper, 1, "gripper"),
                  "base": _vector(base_velocity, 3, "base"),
                  "torso": _vector(torso, 1, "torso")}
        result = np.empty_like(self.low)
        for name, (start, end) in self.slices.items():
            result[start:end] = values[name]
        result[self.mode_index] = float(mode)
        if not np.isfinite(result).all() or np.any(result < self.low - 1e-9) or np.any(result > self.high + 1e-9):
            raise ValueError("Action outside native bounds")
        return result

    def pose_action(self, goal_world, current_world, world_from_control_base, gripper,
                    *, gain=.8, base_velocity=(0., 0., 0.), torso=0.):
        goal, current = np.asarray(goal_world, float), np.asarray(current_world, float)
        if goal.shape != (4, 4) or current.shape != (4, 4):
            raise ValueError("TCP poses must be 4x4 transforms")
        if self.arm.input_ref_frame == "base":
            local = np.linalg.inv(np.asarray(world_from_control_base, float))
            goal, current = local @ goal, local @ current
        if self.arm.input_type == "delta":
            error = np.r_[goal[:3, 3] - current[:3, 3],
                          Rotation.from_matrix(goal[:3, :3] @ current[:3, :3].T).as_rotvec()]
            arm_action = physical_to_input(gain * error, self.arm)
        else:
            arm_action = np.r_[goal[:3, 3], Rotation.from_matrix(goal[:3, :3]).as_rotvec()]
        return self.assemble(arm_action, gripper, base_velocity=base_velocity, torso=torso)
