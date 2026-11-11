"""Geometry-derived articulation skills; discovery currently uses simulator truth.

No task names or world coordinates are encoded. Only fingers may contact the
identified handle. The drawer joint is moved by physical contact, never qpos edits.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
from graspgen_libero_bridge.adapters.libero.scene import _geometry
from graspgen_libero_bridge.control.controller import ContactMonitor


def constrained_pose(pose, axis, anchor, delta, joint_type):
    """Transport a grip pose with a slide (2) or hinge (3) joint in world frame."""
    pose = np.asarray(pose, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    out = pose.copy()
    if joint_type == 2:
        out[:3, 3] += axis * float(delta)
    elif joint_type == 3:
        turn = Rotation.from_rotvec(axis * float(delta)).as_matrix()
        anchor = np.asarray(anchor, dtype=float)
        out[:3, 3] = anchor + turn @ (pose[:3, 3] - anchor)
        out[:3, :3] = turn @ pose[:3, :3]
    else:
        raise ValueError('Only slide and hinge paths are defined')
    return out


def discover(adapter, region, output):
    """Find the goal joint and a marked mesh or elongated capsule handle."""
    model, data = adapter.env.sim.model, adapter.env.sim.data
    if region in adapter.task_env.obj_body_id:
        eligible = body_subtree_ids(model, adapter.task_env.obj_body_id[region])
        joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in eligible
                  and int(model.jnt_type[j]) in (2,3)]
    else:
        site = int(model.site_name2id(region))
        body = int(model.site_bodyid[site])
        joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == body
                  and int(model.jnt_type[j]) in (2,3)]
    if len(joints) != 1:
        raise ValueError('Goal must identify exactly one slide or hinge joint')
    joint = joints[0]
    body = int(model.jnt_bodyid[joint])
    bodies = body_subtree_ids(model,body)
    handles = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in bodies and int(model.geom_type[g]) == 7:
            if 'handle' in (model.mesh_id2name(int(model.geom_dataid[g])) or '').lower():
                handles.append(g)
    folder = Path(output)/'handle_geometry'
    folder.mkdir(parents=True,exist_ok=True)
    capsule = None
    if len(handles) == 1:
        _,points = _geometry(model,data,handles[0],folder,20.)
        padding = .004
    elif not handles:
        bars = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
                and int(model.geom_type[g]) == 3 and (model.geom_contype[g] or model.geom_conaffinity[g])
                and model.geom_size[g,1] > 2*model.geom_size[g,0]]
        if not bars:
            # A rotary knob has no handle bar. Identify it from its semantic
            # visual mesh, then derive the grip from its physical joint body.
            knobs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
                     and int(model.geom_type[g]) == 7 and any(tag in
                     (model.mesh_id2name(int(model.geom_dataid[g])) or '').lower() for tag in ('knob', 'button'))]
            if int(model.jnt_type[joint]) != 3 or not knobs:
                raise ValueError('No semantic mesh, elongated capsule handle or rotary knob')
            physical = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
                        and (model.geom_contype[g] or model.geom_conaffinity[g])]
            points = np.concatenate([_geometry(model,data,g,folder,20.)[1] for g in physical])
            lo, hi = points.min(axis=0), points.max(axis=0)
            axis = np.asarray(data.xaxis[joint]).copy()
            cov = np.cov(points.T)
            _, vectors = np.linalg.eigh(cov)
            along = vectors[:, -1] - axis*np.dot(axis, vectors[:, -1])
            along /= np.linalg.norm(along)
            center = (lo+hi)/2
            center[2] = hi[2]-.012
            return dict(region=region,joint_id=joint,body_id=body,joint_name=model.joint_id2name(joint),
                joint_type=3,axis=axis,anchor=np.asarray(data.xanchor[joint]).copy(),
                q_address=int(model.jnt_qposadr[joint]),limits=np.asarray(model.jnt_range[joint]).copy(),
                handle_center=center,handle_axis=along,handle_half_sizes=(hi-lo)/2,
                handle_rotation=np.eye(3),handle_geoms=physical,
                handle_geom_names=[model.geom_id2name(g) for g in physical],
                rotary_knob=True,uses_ground_truth_articulation=True,uses_semantic_handle_mesh=False)
        capsule = max(bars,key=lambda g:float(model.geom_size[g,1]))
        _,points = _geometry(model,data,capsule,folder,20.)
        padding = .025
    else:raise ValueError('Ambiguous handle geometry')
    lo,hi = points.min(axis=0),points.max(axis=0)
    collisions=[]
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in bodies or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
        _,points = _geometry(model,data,g,folder,20.)
        if np.all(points.min(axis=0)>=lo-padding) and np.all(points.max(axis=0)<=hi+padding):collisions.append(g)
    if capsule is None:
        bars=[g for g in collisions if int(model.geom_type[g])==6]
        if not bars:raise ValueError('No physical handle bar inside visual bounds')
        bar=max(bars,key=lambda g:float(np.max(model.geom_size[g])))
        half=np.asarray(model.geom_size[bar]).copy()
        long_axis=np.asarray(data.geom_xmat[bar]).reshape(3,3)[:,int(np.argmax(half))]
    else:
        bar=capsule;radius,length=model.geom_size[bar,:2]
        half=np.array([radius,radius,length+radius])
        long_axis=np.asarray(data.geom_xmat[bar]).reshape(3,3)[:,2]
    return dict(region=region,joint_id=joint,body_id=body,joint_name=model.joint_id2name(joint),
                joint_type=int(model.jnt_type[joint]),axis=np.asarray(data.xaxis[joint]).copy(),
                anchor=np.asarray(data.xanchor[joint]).copy(),q_address=int(model.jnt_qposadr[joint]),
                limits=np.asarray(model.jnt_range[joint]).copy(),handle_center=np.asarray(data.geom_xpos[bar]).copy(),
                handle_axis=long_axis,handle_half_sizes=half,handle_rotation=np.asarray(data.geom_xmat[bar]).reshape(3,3).copy(),
                handle_geoms=collisions,handle_geom_names=[model.geom_id2name(g) for g in collisions],
                uses_ground_truth_articulation=True,uses_semantic_handle_mesh=bool(handles))


def grip_candidates(info, desired_q, measured_q, distance=.07):
    """Side approach to a handle, parameterized by the opening direction."""
    if info['joint_type'] == 2:
        outward = info['axis'] * np.sign(desired_q - measured_q)
    elif info['joint_type'] == 3:
        outward = np.cross(info['axis'],info['handle_center']-info['anchor'])*np.sign(desired_q-measured_q)
        outward /= np.linalg.norm(outward)
    else:raise ValueError('Unsupported joint type')
    # Shared side/oblique approaches keep the palm above nearby table objects.
    up = np.array([0., 0., 1.])
    for tilt in (0., 35., 55.):
        angle = np.deg2rad(tilt)
        approach = -outward*np.cos(angle)-up*np.sin(angle)
        approach /= np.linalg.norm(approach)
        closing = np.cross(info['handle_axis'], approach)
        closing /= np.linalg.norm(closing)
        for flip in (1., -1.):
            x = closing * flip
            rotation = np.column_stack((x, np.cross(approach, x), approach))
            grip = np.eye(4)
            grip[:3, :3] = rotation
            grip[:3, 3] = info['handle_center']
            pregrip = grip.copy()
            pregrip[:3, 3] -= distance * approach
            yield pregrip, grip


class ArticulationPreviewMonitor:
    """Synchronize the mechanism with each robot sample in a static preview.

    The caller must hold a full simulation snapshot. This wrapper is never used
    for physical control. It preserves the original contact checks/permissions.
    """
    def __init__(self, adapter, monitor, info, reference_pose, initial_q, goal_q):
        self.adapter, self.monitor, self.info = adapter, monitor, info
        self.reference = np.asarray(reference_pose).copy()
        self.initial_q, self.goal_q = float(initial_q), float(goal_q)

    def __getattr__(self, name):
        return getattr(self.monitor, name)

    def check(self, allow_finger_target=False):
        sim = self.adapter.env.sim
        site = self.adapter.task_env.robots[0].eef_site_id
        axis = np.asarray(self.info['axis'], dtype=float)
        axis = axis / np.linalg.norm(axis)
        if self.info['joint_type'] == 2:
            delta = float(np.dot(np.asarray(sim.data.site_xpos[site])-self.reference[:3, 3], axis))
        elif self.info['joint_type'] == 3:
            rotation = np.asarray(sim.data.site_xmat[site]).reshape(3, 3) @ self.reference[:3, :3].T
            delta = float(np.dot(Rotation.from_matrix(rotation).as_rotvec(), axis))
        else:
            raise ValueError('Preview requires a slide or hinge')
        q = np.clip(self.initial_q+delta, min(self.initial_q,self.goal_q), max(self.initial_q,self.goal_q))
        sim.data.qpos[self.info['q_address']] = q
        sim.forward()
        return self.monitor.check(allow_finger_target)


class HandleMonitor(ContactMonitor):
    """Keep all old collision checks, exempting only finger/handle contact."""
    def __init__(self, adapter, target, allowed_geoms):
        super().__init__(adapter, target)
        self.allowed_handle_geoms = set(allowed_geoms)

    def check(self, allow_finger_target=False):
        bad = super().check(False)
        if not allow_finger_target:
            return bad
        kept = []
        for item in bad:
            names = (item.get('geom1'), item.get('geom2'))
            permitted = any(name in self.allowed_handle_geoms for name in names) and any(
                'finger' in (name or '') for name in names)
            if not permitted or item.get('self_collision'):
                kept.append(item)
        return kept
