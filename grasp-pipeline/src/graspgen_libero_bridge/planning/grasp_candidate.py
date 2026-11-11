"""Necessary target-engagement check, using the already privileged scene bounds.

A collision-free empty grasp is not an executable pick. The control site is the
Panda grip centre; reject proposals whose centre misses even the target's
collision AABB plus a small tolerance. This does not certify force closure.
"""
import numpy as np

def control_target_distance(control_pose, bounds_world):
    pose=np.asarray(control_pose,dtype=float);bounds=np.asarray(bounds_world,dtype=float)
    if pose.shape!=(4,4) or bounds.shape!=(2,3) or not np.isfinite(pose).all() or not np.isfinite(bounds).all() or (bounds[1]<bounds[0]).any():
        raise ValueError('Invalid control pose or target bounds')
    return float(np.linalg.norm(np.maximum(np.maximum(bounds[0]-pose[:3,3],pose[:3,3]-bounds[1]),0)))
