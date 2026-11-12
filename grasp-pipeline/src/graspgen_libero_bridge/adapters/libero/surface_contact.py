"""Geometry-derived closed-finger pushing contacts on a moving mechanism."""
from pathlib import Path
import numpy as np
from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
from graspgen_libero_bridge.adapters.libero.scene import _geometry


def closing_push_candidates(adapter, info, qgoal, output):
    """Choose exposed box faces facing the required joint motion.

    Uses live joint axes, moving-body collision geometry and measured closed
    finger-pad offsets. Returns candidates and the moving geoms fingers may
    touch; the cabinet housing and every non-finger link remain obstacles.
    """
    model, data = adapter.env.sim.model, adapter.env.sim.data
    folder = Path(output)/'surface_geometry';folder.mkdir(parents=True,exist_ok=True)
    bodies = body_subtree_ids(model,info['body_id'])
    moving = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
              and (model.geom_contype[g] or model.geom_conaffinity[g])]
    finger = [g for g in range(model.ngeom) if
              'finger' in (model.body_id2name(int(model.geom_bodyid[g])) or '')
              and (model.geom_contype[g] or model.geom_conaffinity[g])]
    pads = [g for g in finger if 'pad' in (model.geom_id2name(g) or '')]
    pose = np.asarray(adapter.state()['world_T_eef'])
    local = np.concatenate([(_geometry(model,data,g,folder,20.)[1]-pose[:3,3])@pose[:3,:3]
                            for g in (pads or finger)])
    sign = np.sign(qgoal-float(data.qpos[info['q_address']]))
    faces = []
    for geom in moving:
        if int(model.geom_type[geom]) != 6:
            continue
        sizes = np.asarray(model.geom_size[geom]);rotation=np.asarray(data.geom_xmat[geom]).reshape(3,3)
        center=np.asarray(data.geom_xpos[geom]).copy();thin=int(np.argmin(sizes))
        direction = np.asarray(info['axis'])*sign
        if info['joint_type']==3:
            direction=np.cross(info['axis'],center-info['anchor'])*sign
        norm=np.linalg.norm(direction)
        if norm<1e-8:continue
        direction/=norm
        normal=rotation[:,thin].copy();normal*=np.sign(normal@direction)
        alignment=float(normal@direction)
        if alignment<.7:continue
        sides=[i for i in range(3) if i!=thin]
        area=float(4*np.prod(sizes[sides]))
        if area<.003:continue
        faces.append((area*alignment,geom,center-normal*sizes[thin],normal,rotation,sizes,sides))
    candidates=[]
    for _,geom,center,normal,rotation,sizes,sides in sorted(faces,key=lambda x:-x[0])[:2]:
        for u,v in ((0.,0.),(.55,0.),(-.55,0.),(0.,.45),(0.,-.45),(.55,.45),(-.55,.45)):
            point=center+rotation[:,sides[0]]*sizes[sides[0]]*u+rotation[:,sides[1]]*sizes[sides[1]]*v
            for tilt in (70.,45.,20.):
                z=normal*np.cos(np.deg2rad(tilt))+np.array([0.,0.,-1.])*np.sin(np.deg2rad(tilt))
                z/=np.linalg.norm(z)
                x=np.cross([0.,0.,1.],normal)
                if np.linalg.norm(x)<1e-6:continue
                x/=np.linalg.norm(x)
                for flip in (1.,-1.):
                    frame=np.column_stack((x*flip,np.cross(z,x*flip),z))
                    transformed=local@frame.T;projection=transformed@normal
                    support=transformed[projection>=projection.max()-.001].mean(axis=0)
                    grip=np.eye(4);grip[:3,:3]=frame;grip[:3,3]=point-support
                    pre=grip.copy();pre[:3,3]-=normal*.07;pre[2,3]+=.03
                    geom_pose=np.eye(4);geom_pose[:3,:3]=rotation
                    geom_pose[:3,3]=data.geom_xpos[geom]
                    candidates.append((pre,grip,dict(group='closing_surface',geom=model.geom_id2name(geom),
                        face_uv=[u,v],tilt_degrees=tilt,normal=normal.tolist(),
                        geom_T_eef=(np.linalg.inv(geom_pose)@grip).tolist(),free_handle_roll=False)))
    return candidates,moving
