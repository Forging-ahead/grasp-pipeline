"""Gravity-aligned opening geometry and bounded object orientation candidates."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def gravity_frame(region, size):
    region, size = np.asarray(region, float), np.asarray(size, float)
    rotation = region[:3, :3]
    vertical = int(np.argmax(np.abs(rotation[2])))
    if abs(rotation[2, vertical]) < np.cos(np.deg2rad(10)):
        raise ValueError('No nearly vertical axis in region box')
    order = [i for i in range(3) if i != vertical] + [vertical]
    result = region.copy()
    result[:3, :3] = rotation[:, order]
    if result[2, 2] < 0:
        result[:3, 2] *= -1
    if np.linalg.det(result[:3, :3]) < 0:
        result[:3, 1] *= -1
    return result, size[order]


def fits_opening(pose, points_local, frame, size, clearance=.002):
    world = np.asarray(points_local) @ pose[:3, :3].T + pose[:3, 3]
    local = (world-frame[:3, 3]) @ frame[:3, :3]
    gravity = frame[:3, :3].T @ np.array([0., 0., -1.])
    for level in (-size[2], size[2]):
        crossing = local + ((level-local[:, 2])/gravity[2])[:, None]*gravity
        if np.any(np.abs(crossing[:, :2]) > size[:2]-clearance):
            return False
    return True


def opening_offsets(pose, points_local, frame, size, clearance=.002):
    """Interior shifts allowed by the entire object footprint, center first."""
    world=np.asarray(points_local)@pose[:3,:3].T+pose[:3,3]
    local=(world-frame[:3,3])@frame[:3,:3]
    lower=-size[:2]+clearance-local[:,:2].min(axis=0)
    upper=size[:2]-clearance-local[:,:2].max(axis=0)
    if np.any(lower>upper):return []
    offsets=[np.zeros(2)]
    for fraction in (.6,.95):
        lo,hi=lower*fraction,upper*fraction
        offsets.extend(np.array([x,y]) for x,y in [(lo[0],0),(hi[0],0),(0,lo[1]),(0,hi[1]),
            (lo[0],lo[1]),(lo[0],hi[1]),(hi[0],lo[1]),(hi[0],hi[1])])
    unique=[]
    for xy in offsets:
        if np.all(xy>=lower) and np.all(xy<=upper) and not any(np.allclose(xy,x) for x in unique):unique.append(xy)
    return [frame[:3,:2]@xy for xy in unique]


def orientation_candidates(pose, points_local, frame, size, floor_z, clearance=.015, *, require_origin_containment=False):
    """Try the current orientation and cube rotations, keeping gravity clearance.

    No asset or task identifiers are used. These bounded alternatives are tested
    for geometric opening fit, then still require actual robot collision checks.
    """
    rotations = [np.asarray(pose)[:3,:3]]
    _, basis = np.linalg.eigh(np.cov(np.asarray(points_local).T))
    if np.linalg.det(basis)<0:
        basis[:,0]*=-1
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            axes = np.eye(3)[:, order] @ np.diag(signs)
            r = frame[:3,:3] @ axes @ basis.T
            if np.linalg.det(r) > .99 and not any(np.allclose(r, x) for x in rotations):
                rotations.append(r)
    rotations.sort(key=lambda r: float(Rotation.from_matrix(r@pose[:3,:3].T).magnitude()))
    candidates = []
    for expanded in (False,True):
        if expanded:
            base=list(rotations)
            rotations=[]
            for matrix in base:
                for axis in (0,1):
                    for angle in (-35.,-25.,-15.,15.,25.,35.):
                        tilt=Rotation.from_rotvec(frame[:3,axis]*np.deg2rad(angle)).as_matrix()
                        rotations.append(tilt@matrix)
            rotations.sort(key=lambda r: float(Rotation.from_matrix(r@pose[:3,:3].T).magnitude()))
        for rotation in rotations:
            candidate = np.asarray(pose).copy()
            candidate[:3, :3] = rotation
            rotated = np.asarray(points_local) @ candidate[:3, :3].T
            opening_coordinates = rotated @ frame[:3,:3]
            midpoint = (opening_coordinates[:,:2].min(axis=0)+opening_coordinates[:,:2].max(axis=0))/2
            candidate[:3,3] = frame[:3,3]-frame[:3,:2]@midpoint
            candidate[2, 3] = floor_z - float(rotated[:, 2].min()) + clearance
            # LIBERO's region containment uses the object's body origin. A
            # visually inserted object may still fail if that origin settles
            # below the region. Match the unmodified predicate as well as the
            # physical footprint; the final physics result is checked again.
            rest_origin=(candidate[:3,3]-np.array([0.,0.,clearance])-frame[:3,3])@frame[:3,:3]
            contained=not require_origin_containment or bool(np.all(np.abs(rest_origin)<=size-.001))
            if contained and fits_opening(candidate, points_local, frame, size):
                candidates.append(candidate)
        if candidates:break
    return candidates


def region_support_height(adapter, expected_z):
    """Use a real thin support box near the region's nominal support height."""
    m, d = adapter.env.sim.model, adapter.env.sim.data
    name = adapter.config['goal_destination']
    site = int(m.site_name2id(name))
    center = np.asarray(d.site_xpos[site])
    owner = adapter.body_id(adapter.destination)
    from .env_adapter import body_subtree_ids
    bodies = body_subtree_ids(m, owner)
    options = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in bodies or int(m.geom_type[g]) != 6 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        r = np.asarray(d.geom_xmat[g]).reshape(3, 3)
        extent = np.abs(r) @ np.asarray(m.geom_size[g])
        pos = np.asarray(d.geom_xpos[g])
        if extent[2] > .025 or np.any(np.abs(center[:2]-pos[:2]) > extent[:2]+.001):
            continue
        top = float(pos[2]+extent[2])
        if abs(top-expected_z) <= .04:
            options.append((abs(top-expected_z), top, m.geom_id2name(g) or f'geom_{g}'))
    if not options:
        return float(expected_z), None
    _, top, geom = min(options)
    return top, geom
