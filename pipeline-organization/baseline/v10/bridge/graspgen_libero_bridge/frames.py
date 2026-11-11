"""SE(3) boundary conventions; all internal transforms are world-frame matrices."""
import numpy as np
from scipy.spatial.transform import Rotation


def grasp_to_control(pose, offset=.097):
    """Native Panda: approach +z, closing +/-x. Robosuite grip site shares axes."""
    shift = np.eye(4)
    shift[2, 3] = offset
    return np.asarray(pose, dtype=float) @ shift


def pose_error(desired, current):
    desired, current = np.asarray(desired), np.asarray(current)
    return desired[:3, 3] - current[:3, 3], Rotation.from_matrix(desired[:3, :3] @ current[:3, :3].T).as_rotvec()


def pose_to_action(desired, current, gripper, position_scale=.05, rotation_scale=.5, gain=.8):
    position, rotation = pose_error(desired, current)
    return np.r_[np.clip(gain * position / position_scale, -1, 1),
                 np.clip(gain * rotation / rotation_scale, -1, 1), float(gripper)]


def interpolate_pose(a, b, u):
    a, b = np.asarray(a), np.asarray(b)
    out = np.eye(4)
    out[:3, 3] = (1-u) * a[:3, 3] + u * b[:3, 3]
    r = Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec()
    out[:3, :3] = Rotation.from_rotvec(r * u).as_matrix() @ a[:3, :3]
    return out
