"""Kinematic bilateral-jaw check using actual MuJoCo target geometry.

Only the preview fingers move; full physics state is restored. This is a
necessary engagement screen, not a dynamic grasp or force-closure certificate.
"""
import numpy as np
from .cartesian import _snapshot,_restore
from .env_adapter import body_subtree_ids

def preview_grasp_contacts(adapter,monitor,q_arm):
 sim=adapter.env.sim;model,data=sim.model,sim.data;robot=adapter.task_env.robots[0]
 saved=_snapshot(sim);onsets={};normals={};world_failures=[]
 try:
  data.qpos[robot._ref_joint_pos_indexes]=np.asarray(q_arm,dtype=float)
  for aperture in np.linspace(.04,0,41):
   data.qpos[robot._ref_gripper_joint_pos_indexes]=[aperture,-aperture];sim.forward()
   bad=monitor.check(True)
   if bad:
    world_failures=bad;break
   for i in range(data.ncon):
    con=data.contact[i];b1,b2=[int(model.geom_bodyid[int(g)]) for g in [con.geom1,con.geom2]]
    if (b1 in monitor.target_bodies)==(b2 in monitor.target_bodies):continue
    other=b2 if b1 in monitor.target_bodies else b1;name=model.body_id2name(other) or ''
    side='left' if ('leftfinger' in name or 'finger_joint1_tip' in name) else ('right' if ('rightfinger' in name or 'finger_joint2_tip' in name) else None)
    if side and con.dist<=.0001 and side not in onsets:
     onsets[side]=float(aperture)
     axis=np.asarray(data.site_xmat[robot.eef_site_id]).reshape(3,3)[:,0] if hasattr(robot,'eef_site_id') else np.array([1.,0,0])
     normal=np.asarray(con.frame).reshape(3,3)[0]
     if b2 in monitor.target_bodies:normal=-normal
     normals[side]=float(np.dot(normal,axis))
   if len(onsets)==2:break
 finally:_restore(sim,saved)
 both=len(onsets)==2;balanced=both and abs(onsets['left']-onsets['right'])<=.015
 passed=both and balanced and not world_failures
 return dict(passed=bool(passed),onset_half_apertures_m=onsets,contact_normal_along_closing_axis=normals,world_contacts=world_failures,
  state_restored=True,collision_exemptions_added=[],limitation='Static bilateral contact screen, not dynamic grasp success')
