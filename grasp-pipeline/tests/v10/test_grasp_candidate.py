import numpy as np
import pytest
from graspgen_libero_bridge.planning.grasp_candidate import control_target_distance

def test_empty_grasp_is_not_mistaken_for_collision_free_pick():
 bounds=[[-.15,-.275,0],[-.087,-.212,.076]]
 empty=np.eye(4);empty[:3,3]=[.023,-.198,.086]
 assert control_target_distance(empty,bounds)>.1
 valid=np.eye(4);valid[:3,3]=[-.12,-.24,.04]
 assert control_target_distance(valid,bounds)==0

def test_corner_distance_and_bad_bounds():
 p=np.eye(4);p[:3,3]=[1.03,1.04,.5]
 assert control_target_distance(p,[[0,0,0],[1,1,1]])==pytest.approx(.05)
 with pytest.raises(ValueError):control_target_distance(p,[[1,0,0],[0,1,1]])
