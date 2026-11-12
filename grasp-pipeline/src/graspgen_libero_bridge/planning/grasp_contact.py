"""Kinematic bilateral-jaw check using actual MuJoCo target geometry.

Only the preview fingers move; full physics state is restored. This is a
necessary engagement screen, not a dynamic grasp or force-closure certificate.
"""
import numpy as np
from graspgen_libero_bridge.planning.cartesian import _snapshot,_restore
from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids

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


def preview_handle_contacts(adapter, monitor, q_arm):
    """Close from the measured aperture, then return a supported preview grip.

    Starting at the maximum opening can falsely reject a reachable handle near
    a wall. This screen never opens beyond the actual prepared width and never
    changes the live episode. Normal directions and onset balance are necessary
    engagement conditions; dynamics and friction still need physical execution.
    """
    sim = adapter.env.sim
    model, data = sim.model, sim.data
    robot = adapter.task_env.robots[0]
    finger_ids = np.asarray(robot._ref_gripper_joint_pos_indexes, dtype=int)
    opening = np.asarray(data.qpos[finger_ids], dtype=float).copy()
    if opening.shape != (2,) or opening[0] < 0 or opening[1] > 0:
        raise ValueError('Handle preview requires the Panda parallel fingers')
    snapshot = _snapshot(sim)
    onsets, normals, bad = {}, {}, []
    closed = None
    try:
        data.qpos[robot._ref_joint_pos_indexes] = np.asarray(q_arm, dtype=float)
        steps = max(2, int(np.ceil(np.max(np.abs(opening))/.0005))+1)
        for u in np.linspace(0., 1., steps):
            data.qpos[finger_ids] = opening*(1.-u)
            sim.forward()
            bad = monitor.check(True)
            if bad:
                break
            axis = np.asarray(data.site_xmat[robot.eef_site_id]).reshape(3,3)[:,0]
            for i in range(data.ncon):
                contact = data.contact[i]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
                if (b1 in monitor.target_bodies)==(b2 in monitor.target_bodies):
                    continue
                other = b2 if b1 in monitor.target_bodies else b1
                name = model.body_id2name(other) or ''
                side = 'left' if ('leftfinger' in name or 'finger_joint1_tip' in name) else (
                    'right' if ('rightfinger' in name or 'finger_joint2_tip' in name) else None)
                if side is None or contact.dist > .0001:
                    continue
                normal = np.asarray(contact.frame).reshape(3,3)[0]
                if b2 in monitor.target_bodies:
                    normal = -normal
                component = float(normal@axis)
                if abs(component)<.5:
                    continue
                if side not in onsets:
                    index = 0 if side=='left' else 1
                    onsets[side] = float(abs(data.qpos[finger_ids[index]]))
                    normals[side] = component
            if len(onsets)==2:
                candidate = np.asarray(data.qpos[finger_ids]).copy()
                candidate[0] = max(0., candidate[0]-.0002)
                candidate[1] = min(0., candidate[1]+.0002)
                data.qpos[finger_ids] = candidate
                sim.forward()
                bad = monitor.check(True)
                if not bad:
                    closed = candidate.tolist()
                break
    finally:
        _restore(sim, snapshot)
    balanced = len(onsets)==2 and abs(onsets['left']-onsets['right'])<=.003
    opposed = len(normals)==2 and normals['left']*normals['right']<-.25
    passed = bool(closed is not None and balanced and opposed and not bad)
    return dict(success=passed, passed=passed, q_fingers=closed if passed else None,
        measured_open_q_fingers=opening.tolist(), onset_half_apertures_m=onsets,
        contact_normal_along_closing_axis=normals, world_contacts=bad,
        state_restored=True, collision_exemptions_added=[],
        limitation='Static handle engagement only; physical compression and joint progress remain required')
