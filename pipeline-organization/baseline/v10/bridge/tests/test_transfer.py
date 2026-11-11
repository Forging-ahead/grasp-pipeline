import numpy as np
import pytest
from graspgen_libero_bridge.transfer import transfer_candidates

def test_object_and_eef_goals_preserve_grasp_and_destination():
 goal=np.eye(4);goal[:3,3]=[.2,.1,.9]
 relation=np.eye(4);relation[:3,3]=[.03,.04,.02]
 for option in transfer_candidates(goal,relation):
  raised=option['object_goal'].copy();raised[2,3]+=option['height_m']
  np.testing.assert_allclose(option['eef_goal']@relation,raised,atol=1e-12)
  np.testing.assert_allclose(option['object_goal'][:3,3],goal[:3,3])

def test_rejects_invalid_pose_and_unbounded_clearance():
 with pytest.raises(ValueError):list(transfer_candidates(np.zeros((4,4)),np.eye(4)))
 with pytest.raises(ValueError):list(transfer_candidates(np.eye(4),np.eye(4),heights=[-.1]))

def test_search_passes_selected_orientation_to_placement(tmp_path):
 from types import SimpleNamespace
 from graspgen_libero_bridge.evaluate import PickPlace
 from graspgen_libero_bridge.errors import ExecutionError
 calls=[]
 def plan(goal,phase):
  calls.append(goal)
  if len(calls)==1:raise ExecutionError('no IK',code='motion_plan_failed',category='planning',phase=phase)
  return {'success':True,'path_file':'verified_path'}
 fake=SimpleNamespace(config={},adapter=SimpleNamespace(step_count=200),output=tmp_path,result={},plan_and_execute=plan)
 placed,_=PickPlace.transfer_with_alternatives(fake,np.eye(4),np.eye(4))
 np.testing.assert_allclose(placed[:3,:3],calls[-1][:3,:3]);assert fake.result['transfer_search']['selected']['yaw_degrees']==45

def test_search_does_not_retry_after_partial_execution(tmp_path):
 from types import SimpleNamespace
 from graspgen_libero_bridge.evaluate import PickPlace
 from graspgen_libero_bridge.errors import ExecutionError
 fake=SimpleNamespace(config={},adapter=SimpleNamespace(step_count=200),output=tmp_path,result={})
 calls=[]
 def plan(goal,phase):
  calls.append(goal);fake.adapter.step_count+=1
  raise ExecutionError('stopped',code='motion_plan_failed',category='planning',phase=phase)
 fake.plan_and_execute=plan
 with pytest.raises(ExecutionError):PickPlace.transfer_with_alternatives(fake,np.eye(4),np.eye(4))
 assert len(calls)==1

def test_approach_distance_search_keeps_independent_candidate_pools(tmp_path):
 from types import SimpleNamespace
 from graspgen_libero_bridge.evaluate import PickPlace
 from graspgen_libero_bridge.errors import ExecutionError
 calls=[]
 def request(r,**kw):
  calls.append((r['approach_distance_m'],[x['generation_index'] for x in r['candidates']]))
  return dict(success=False,reason='no_executable_grasp',attempted_indices=[0],attempts=[])
 fake=SimpleNamespace(config={'selection_rounds':3,'max_candidates_per_round':1,'native_descent_fallback':False},output=tmp_path,result={},planner=SimpleNamespace(request=request),scene=lambda:{})
 with pytest.raises(ExecutionError):PickPlace.select_candidate(fake,[{'generation_index':0},{'generation_index':1}])
 assert calls==[(.10,[0,1]),(.05,[0,1]),(.10,[1])]

def test_approach_deviation_replans_from_feedback_without_closing_gripper(tmp_path,monkeypatch):
 from types import SimpleNamespace
 from graspgen_libero_bridge.evaluate import PickPlace
 from graspgen_libero_bridge.errors import ExecutionError
 import graspgen_libero_bridge.evaluate as module
 path=tmp_path/'old.npz';np.savez(path,world_T_eef=np.eye(4)[None])
 calls=[]
 def follow(path,phase,**kw):
  calls.append(phase)
  if phase=='approach':raise ExecutionError('branch',code='joint_branch_deviation',category='control',phase=phase)
 def replan(goal,phase,**kw):
  assert phase=='approach_replan' and kw['gripper']==-1 and kw['allow_contact'] is False
  calls.append(phase)
 monkeypatch.setattr(module,'plan_cartesian',lambda *a,**kw:dict(success=True,state_restored=True,path_file='new_descent'))
 fake=SimpleNamespace(controller=SimpleNamespace(follow_path=follow,monitor=object()),adapter=object(),result={},output=tmp_path,plan_and_execute=replan)
 PickPlace.execute_grasp_paths(fake,dict(approach_path_file=str(path),descend_path_file='old',selected={'world_T_eef':np.eye(4).tolist()}))
 assert calls==['approach','approach_replan','descend']

def test_real_execution_contact_is_not_retried_as_branch_deviation(tmp_path):
 from types import SimpleNamespace
 from graspgen_libero_bridge.evaluate import PickPlace
 from graspgen_libero_bridge.errors import ExecutionError
 def follow(*a,**kw):raise ExecutionError('contact',code='forbidden_contact',category='collision',phase='approach')
 fake=SimpleNamespace(controller=SimpleNamespace(follow_path=follow),result={},output=tmp_path)
 with pytest.raises(ExecutionError,match='contact'):PickPlace.execute_grasp_paths(fake,{'approach_path_file':'unused'})
 assert 'approach_replanning' not in fake.result
