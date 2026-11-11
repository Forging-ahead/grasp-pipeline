"""Allocate nonoverlapping footprints for multiple On goals on one support."""
import numpy as np


def pack_rectangles(bounds, support_low, support_high, goal_half, gap=.006, margin=.003):
    """Translate body-relative 2D footprints within physical and goal bounds.

    Only body origins must satisfy the virtual goal region, while the complete
    conservative footprint must remain on the real physical support. Return
    None when this bounded axis-packing method finds no arrangement.
    """
    bounds=np.asarray(bounds,float);lo=np.asarray(support_low,float);hi=np.asarray(support_high,float)
    goal_half=np.asarray(goal_half,float)
    if bounds.ndim!=3 or bounds.shape[1:]!=(2,2) or np.any(bounds[:,1]<=bounds[:,0]):raise ValueError('Invalid object footprints')
    choices=[]
    for axis in (0,1):
        other=1-axis;widths=bounds[:,1,axis]-bounds[:,0,axis]
        total=float(widths.sum()+gap*(len(bounds)-1))
        if total>hi[axis]-lo[axis]-2*margin:continue
        cursor=(lo[axis]+hi[axis]-total)/2
        shifts=[]
        for box,width in zip(bounds,widths):
            shift=np.zeros(2);shift[axis]=cursor-box[0,axis]
            shift[other]=(lo[other]+hi[other]-box[0,other]-box[1,other])/2
            cursor+=width+gap
            placed=box+shift
            if np.any(placed[0]<lo+margin) or np.any(placed[1]>hi-margin) or np.any(np.abs(shift)>goal_half-margin):break
            shifts.append(shift)
        else:choices.append((float(hi[axis]-lo[axis]-total),shifts))
    return max(choices,key=lambda x:x[0])[1] if choices else None


def surface_slots(adapter, stages, directory):
    """Infer shared support surfaces from actual collision geometry and goals."""
    from pathlib import Path
    from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
    from graspgen_libero_bridge.adapters.libero.scene import _geometry,_contact_geom_ids
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    m,d=adapter.env.sim.model,adapter.env.sim.data
    groups={}
    for i,stage in enumerate(stages):
        if stage['skill']=='pick_place' and stage['relation']=='on':groups.setdefault(stage['region'],[]).append((i,stage))
    result={};audit=[]
    for region,items in groups.items():
        if len(items)<2 or region not in adapter.task_env.object_sites_dict:continue
        owner=items[0][1]['destination']
        if owner not in adapter.task_env.obj_body_id:continue
        site=int(m.site_name2id(region));rotation=np.asarray(d.site_xmat[site]).reshape(3,3);center=np.asarray(d.site_xpos[site])
        if abs(rotation[2,2])<.995:continue
        bodies=body_subtree_ids(m,adapter.task_env.obj_body_id[owner]);surfaces=[]
        for g in _contact_geom_ids(m):
            if int(m.geom_bodyid[g]) not in bodies or int(m.geom_type[g])!=6:continue
            _,world=_geometry(m,d,g,directory,20.)
            local=(world-center)@rotation;lo,hi=local.min(axis=0),local.max(axis=0)
            if hi[2]-lo[2]>.05 or abs(hi[2])>.05 or np.any(lo[:2]>0) or np.any(hi[:2]<0):continue
            surfaces.append((hi[2],lo[:2],hi[:2],m.geom_id2name(g)))
        if not surfaces:continue
        _,low,high,name=max(surfaces,key=lambda x:x[0])
        footprints=[]
        for _,stage in items:
            target=stage['target'];bodies=body_subtree_ids(m,adapter.task_env.obj_body_id[target]);points=[]
            origin=adapter.target_pose(target)[:3,3]
            for g in _contact_geom_ids(m):
                if int(m.geom_bodyid[g]) in bodies:points.append(_geometry(m,d,g,directory,20.)[1])
            local=(np.concatenate(points)-origin)@rotation
            footprints.append(np.stack((local[:,:2].min(axis=0),local[:,:2].max(axis=0))))
        size=np.asarray(adapter.task_env.object_sites_dict[region].size)
        offsets=pack_rectangles(footprints,low,high,size[:2])
        entry=dict(region=region,physical_support=name,footprints=np.asarray(footprints).tolist(),
            support_low=low.tolist(),support_high=high.tolist(),arrangement_found=offsets is not None)
        if offsets is not None:
            entry['body_origins_world']=[]
            for (index,stage),xy in zip(items,offsets):
                world=center+rotation[:,:2]@xy
                result[index]=world[:2].tolist();entry['body_origins_world'].append(dict(target=stage['target'],xy=world[:2].tolist()))
        audit.append(entry)
    return result,audit
