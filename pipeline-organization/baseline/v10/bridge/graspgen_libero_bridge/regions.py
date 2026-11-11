"""Resolve a virtual task region to its actual arena support geometry."""
import numpy as np
from .env_adapter import body_subtree_ids


def support_surface(adapter):
    model,data=adapter.env.sim.model,adapter.env.sim.data
    region=adapter.config['goal_destination'];site=int(model.site_name2id(region))
    center=np.asarray(data.site_xpos[site]);excluded=set()
    for body in adapter.task_env.obj_body_id.values():excluded.update(body_subtree_ids(model,body))
    excluded.update(body_subtree_ids(model,model.body_name2id(adapter.robot.robot_model.root_body)))
    candidates=[]
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) in excluded or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):continue
        kind=int(model.geom_type[geom]);pos=np.asarray(data.geom_xpos[geom]);rot=np.asarray(data.geom_xmat[geom]).reshape(3,3);size=np.asarray(model.geom_size[geom])
        if kind==6:
            if np.max(np.abs(rot[2]))<.995:continue
            extent=np.abs(rot)@size
            if np.any(np.abs(center[:2]-pos[:2])>extent[:2]):continue
            top=float(pos[2]+extent[2])
        elif kind==0:
            if abs(rot[2,2])<.995:continue
            top=float(pos[2])
        else:continue
        if abs(top-center[2])<=.05:
            candidates.append((top,model.geom_id2name(geom) or f'geom_{geom}'))
    if not candidates:raise ValueError('No horizontal arena support near the goal region')
    top=max(x[0] for x in candidates)
    names=[name for height,name in candidates if abs(height-top)<.002]
    return dict(region=region,marker_z_m=float(center[2]),physical_surface_z_m=top,geom_names=names,
                method='contact-enabled arena plane/box below region; movable objects excluded',uses_ground_truth_geometry=True)
