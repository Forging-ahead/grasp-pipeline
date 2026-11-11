from types import SimpleNamespace
import numpy as np
import pytest
from graspgen_libero_bridge.control.controller import Controller,ExecutionError
from graspgen_libero_bridge.planning.grasp_contact import preview_grasp_contacts
import test_cartesian as fixtures

def servo():
 c=Controller.__new__(Controller);pose=np.eye(4);pose[0,3]=.0045
 c.adapter=SimpleNamespace(config={},state=lambda:{'world_T_eef':pose.copy()});c.recorder=SimpleNamespace(phase='initial');c.events=[];c.position_scale=.05;c.rotation_scale=.5
 return c,pose

def test_near_goal_progress_extends_time_but_not_precision():
 c,p=servo();steps=[]
 def step(*a):steps.append(1);p[0,3]-=.00005
 c.step=step;c.move_to(np.eye(4),phase='place',max_steps=20,position_tolerance=.0025)
 assert 20<len(steps)<=140 and abs(p[0,3])<=.0025
 assert c.events[-1]['unchanged_position_tolerance_m']==.0025

def test_stalled_place_is_not_given_extra_budget():
 c,p=servo();p[0,3]=.0035;steps=[];c.step=lambda *a:steps.append(1)
 with pytest.raises(ExecutionError):c.move_to(np.eye(4),phase='place',max_steps=20,position_tolerance=.0025)
 assert len(steps)==20 and c.events[-1]['enabled'] is False

def test_progress_above_old_five_mm_gate_can_finish_at_original_precision():
 def run(config):
  c,p=servo();p[0,3]=.0105;c.adapter.config=config;steps=[]
  def step(*a):steps.append(1);p[0,3]-=.00004
  c.step=step
  return c,p,steps
 old,p,steps=run({})
 with pytest.raises(ExecutionError):old.move_to(np.eye(4),phase='place',max_steps=120,position_tolerance=.0025)
 assert len(steps)==120 and p[0,3]>.005
 new,p,steps=run(dict(place_tracking_extra_steps=240,place_tracking_extension_radius_m=.015))
 new.move_to(np.eye(4),phase='place',max_steps=120,position_tolerance=.0025)
 assert 120<len(steps)<=360 and abs(p[0,3])<=.0025
 assert new.events[-1]['unchanged_position_tolerance_m']==.0025

def test_wider_extension_radius_still_rejects_stall():
 c,p=servo();p[0,3]=.0062;c.adapter.config=dict(place_tracking_extra_steps=240,place_tracking_extension_radius_m=.015)
 steps=[];c.step=lambda *a:steps.append(1)
 with pytest.raises(ExecutionError):c.move_to(np.eye(4),phase='place',max_steps=120,position_tolerance=.0025)
 assert len(steps)==120 and c.events[-1]['enabled'] is False

def test_improving_place_remains_bounded_if_it_never_arrives():
 c,p=servo();p[0,3]=.0149;c.adapter.config=dict(place_tracking_extra_steps=240,place_tracking_extension_radius_m=.015)
 steps=[]
 def step(*a):steps.append(1);p[0,3]-=.000005
 c.step=step
 with pytest.raises(ExecutionError):c.move_to(np.eye(4),phase='place',max_steps=120,position_tolerance=.0025)
 assert len(steps)==360 and p[0,3]>.0025

@pytest.mark.parametrize('radius',[.02,float('nan')])
def test_invalid_extension_radius_is_rejected(radius):
 c,p=servo();c.adapter.config=dict(place_tracking_extension_radius_m=radius)
 with pytest.raises(ValueError):c.move_to(np.eye(4),phase='place')

@pytest.mark.skipif(not fixtures.AVAILABLE,reason='requires MuJoCo')
@pytest.mark.parametrize('right_y,expected',[(0.,True),(.05,False)])
def test_bilateral_contact_screen_restores_full_state(right_y,expected):
 import mujoco
 xml=f'''<mujoco><worldbody>
 <body name="arm"><joint type="slide" axis="0 0 1"/><geom type="sphere" size=".001" contype="0" conaffinity="0"/></body>
 <body name="leftfinger"><joint type="slide" axis="-1 0 0"/><geom type="box" size=".004 .01 .01"/></body>
 <body name="rightfinger" pos="0 {right_y} 0"><joint type="slide" axis="-1 0 0"/><geom type="box" size=".004 .01 .01"/></body>
 <body name="target"><geom type="box" size=".02 .01 .01"/></body></worldbody></mujoco>'''
 m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);sim=fixtures._Sim(m,d);sim.model.body_id2name=lambda i:m.body(i).name
 d.qpos[:]=[.01,.034,-.032];d.qvel[:]=[.1,-.2,.3];d.qacc_warmstart[:]=[.3,.2,.1];d.qfrc_applied[:]=[1,2,3];d.time=7.;sim.forward()
 before={n:np.array(getattr(d,n),copy=True) for n in ['qpos','qvel','qacc_warmstart','qfrc_applied']};time=d.time
 robot=SimpleNamespace(_ref_joint_pos_indexes=[0],_ref_gripper_joint_pos_indexes=[1,2]);adapter=SimpleNamespace(env=SimpleNamespace(sim=sim),task_env=SimpleNamespace(robots=[robot]));monitor=SimpleNamespace(target_bodies={m.body('target').id},check=lambda *a:[])
 r=preview_grasp_contacts(adapter,monitor,[0.]);assert r['passed'] is expected and r['state_restored']
 for n,v in before.items():np.testing.assert_array_equal(getattr(d,n),v)
 assert d.time==time and sim.steps==0
