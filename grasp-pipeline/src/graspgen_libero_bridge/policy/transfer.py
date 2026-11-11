"""Bounded alternatives for an explicit rigid-object placement goal.

Every alternative preserves the destination XY. Yaw and preplace clearance are
searched without relaxing collision, limit, or controller checks. The selected
object orientation must also be used by the subsequent placement phase.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def transfer_candidates(object_goal, eef_to_object, heights=(.12, .08, .04, .16, .20),
                        yaw_degrees=(0., 45., -45., 90., -90., 180.)):
    object_goal = np.asarray(object_goal, dtype=float)
    relation = np.asarray(eef_to_object, dtype=float)
    for value in (object_goal, relation):
        if value.shape != (4, 4) or not np.isfinite(value).all() or not np.allclose(value[3], [0,0,0,1]):
            raise ValueError('Expected finite homogeneous placement transforms')
        if not np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(value[:3,:3])<.99999:
            raise ValueError('Invalid placement rotation')
    inverse = np.linalg.inv(relation)
    for height in heights:
        if not np.isfinite(height) or not .02 <= height <= .30:
            raise ValueError('Preplace height must be between 2 and 30 cm')
        for yaw in yaw_degrees:
            if not np.isfinite(yaw):raise ValueError('Nonfinite yaw')
            placed = object_goal.copy()
            placed[:3,:3] = Rotation.from_euler('z',float(yaw),degrees=True).as_matrix()@object_goal[:3,:3]
            raised = placed.copy();raised[2,3] += float(height)
            yield dict(object_goal=placed, eef_goal=raised@inverse,
                       height_m=float(height), yaw_degrees=float(yaw))
