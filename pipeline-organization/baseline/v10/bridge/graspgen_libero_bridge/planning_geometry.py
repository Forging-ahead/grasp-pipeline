"""CPU geometry used by the LIBERO planner (no CUDA imports).

The generated URDF is derived from the installed robosuite MJCF, including its
gripper mount and control site.  It deliberately does not reuse the Isaac URDF.
Collision spheres cover each collision geom's AABB; this is conservative and
can reject otherwise valid motions.  Link0 is a fixed support fixture.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ARM_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
MAX_ROBOT_SPHERES = 300
SELF_COLLISION_SPHERE_LIMIT = 1024  # CUDA kernel requires strictly fewer.
DEFAULT_ATTACHMENT_SLOTS = 128


def transform(value):
    a = np.asarray(value, dtype=float)
    if a.shape == (7,):
        t = np.eye(4)
        t[:3, 3] = a[:3]
        t[:3, :3] = Rotation.from_quat(a[[4, 5, 6, 3]]).as_matrix()
        a = t
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError("Expected finite 4x4 transform or [xyz,wxyz]")
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid homogeneous transform")
    r = a[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or np.linalg.det(r) < .99999:
        raise ValueError("Invalid rotation matrix")
    return a


def pose_list(t):
    t = transform(t)
    q = Rotation.from_matrix(t[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return [*t[:3, 3].tolist(), *q.tolist()]


def xml_transform(element):
    t = np.eye(4)
    t[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    if element.get("quat"):
        q = np.fromstring(element.get("quat"), sep=" ")
        t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    elif element.get("euler"):
        t[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("euler"), sep=" ")).as_matrix()
    for unsupported in ("axisangle", "xyaxes", "zaxis", "fromto"):
        if element.get(unsupported):
            raise ValueError(f"Unsupported MJCF orientation: {unsupported}")
    return t


def box_spheres(bounds, cell_size=.025, max_cells=192):
    """A complete AABB cover: every grid cell fits inside its sphere."""
    b = np.asarray(bounds, dtype=float)
    if b.shape != (2, 3) or not np.isfinite(b).all() or (b[1] < b[0]).any():
        raise ValueError("Invalid bounds")
    dims = np.maximum(b[1] - b[0], 1e-6)
    size = float(cell_size)
    count = np.maximum(1, np.ceil(dims / size).astype(int))
    while int(np.prod(count)) > max_cells:
        size *= 1.1
        count = np.maximum(1, np.ceil(dims / size).astype(int))
    axes = [np.linspace(lo + d/(2*n), lo+d-d/(2*n), n) for lo, d, n in zip(b[0], dims, count)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    radius = float(np.linalg.norm(dims / count) / 2)
    return np.c_[centers, np.full(len(centers), radius)]


def geometry_mesh(geometry):
    import trimesh
    kind = geometry["type"]
    if kind == "mesh":
        mesh = trimesh.load(str(geometry["mesh_path"]), force="mesh", process=False)
        scale = np.asarray(geometry.get("scale", [1, 1, 1]), float)
        mesh = mesh.copy()
        mesh.vertices *= scale
        return mesh
    if kind == "box":
        return trimesh.creation.box(extents=np.asarray(geometry["dims"], float))
    if kind == "sphere":
        return trimesh.creation.icosphere(subdivisions=2, radius=float(geometry["radius"]))
    if kind == "cylinder":
        return trimesh.creation.cylinder(radius=float(geometry["radius"]), height=float(geometry["length"]), sections=64)
    raise ValueError(f"Unsupported geometry type: {kind}")


def target_spheres(target, world_T_eef, slots=DEFAULT_ATTACHMENT_SLOTS):
    """Conservative cover in control-site coordinates, preserving mesh holes only
    to the extent of the input decomposition (single meshes use their AABB)."""
    geometries = target.get("geometries", [target] if "type" in target else [])
    if not geometries:
        raise ValueError("Target has no collision geometry")
    eef_T_world = np.linalg.inv(transform(world_T_eef))
    per_geom = max(1, slots // len(geometries))
    if len(geometries) > slots:
        raise ValueError("Too many target parts for attachment slots")
    result = []
    for geom in geometries:
        mesh = geometry_mesh(geom)
        spheres = box_spheres(mesh.bounds, cell_size=.01, max_cells=per_geom)
        local = eef_T_world @ transform(geom["pose"])
        spheres[:, :3] = spheres[:, :3] @ local[:3, :3].T + local[:3, 3]
        result.append(spheres)
    return np.concatenate(result)


def sphere_world_diagnostics(robot, scene, world_data, attached=None,
                             robot_sphere_buffer=.001, maximum_results=30):
    """CPU signed sphere/world distances, including a carried object's cover.

    Boxes use their exact oriented-box SDF. Meshes use their local AABB to
    identify nearby spheres, then triangle distances with convex-hull sign.
    Far mesh pairs retain an explicitly marked conservative AABB lower bound.
    This diagnoses the planning proxies; it does not assert physical contact.
    """
    import trimesh
    from scipy.spatial import ConvexHull
    fingers = (np.asarray(scene["q_fingers"], float) if "q_fingers" in scene else
               np.asarray([scene["gripper_width"]/2, -scene["gripper_width"]/2], float))
    frames = robot.fk(np.asarray(scene["q_arm"], float), fingers)
    groups = []
    for name, values in robot.spheres.items():
        spheres = np.array([[*v["center"], v["radius"]+robot_sphere_buffer] for v in values], float)
        frame = frames[name]
        spheres[:, :3] = spheres[:, :3] @ frame[:3, :3].T + frame[:3, 3]
        groups.append((name, spheres))
    if attached is not None:
        spheres = np.array(attached, dtype=float, copy=True)
        frame = frames["control_site"]
        spheres[:, :3] = spheres[:, :3] @ frame[:3, :3].T + frame[:3, 3]
        groups.append(("attached_object", spheres))
    result = []
    for category in ("cuboid", "mesh"):
        for obstacle, spec in world_data.get(category, {}).items():
            base_T_geometry = transform(spec["pose"])
            inverse = np.linalg.inv(base_T_geometry)
            if category == "cuboid":
                half = np.asarray(spec["dims"], float)/2
                center = np.zeros(3)
                mesh = None
            else:
                mesh = trimesh.load(spec["file_path"], force="mesh", process=False)
                mesh = mesh.copy()
                mesh.vertices *= np.asarray(spec.get("scale", [1., 1., 1.]))
                bounds = np.asarray(mesh.bounds)
                center, half = bounds.mean(axis=0), (bounds[1]-bounds[0])/2
                hull = ConvexHull(mesh.vertices)
            for link, spheres in groups:
                points = spheres[:, :3] @ inverse[:3, :3].T + inverse[:3, 3]
                q = np.abs(points-center)-half
                signed = np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(axis=1), 0)
                modes = np.repeat("exact_obb_sdf" if mesh is None else "mesh_AABB_lower_bound", len(points)).astype(object)
                if mesh is not None:
                    near = np.flatnonzero(signed-spheres[:, 3] <= .03)
                    if len(near):
                        _, distances, _ = trimesh.proximity.closest_point_naive(mesh, points[near])
                        inside = (points[near] @ hull.equations[:, :3].T+hull.equations[:, 3] <= 1e-9).all(axis=1)
                        signed[near] = np.where(inside, -distances, distances)
                        modes[near] = "mesh_triangle_distance_convex_hull_sign"
                clearance = signed-spheres[:, 3]
                i = int(np.argmin(clearance))
                result.append(dict(link=link, obstacle=obstacle, sphere_index=i,
                    signed_clearance_m=float(clearance[i]), sphere_radius_m=float(spheres[i, 3]),
                    center_base=spheres[i, :3].tolist(), center_geometry=points[i].tolist(),
                    point_signed_distance_m=float(signed[i]), method=str(modes[i])))
    result.sort(key=lambda row: row["signed_clearance_m"])
    return dict(minimum_signed_clearance_m=None if not result else result[0]["signed_clearance_m"],
        blocking_pairs=[row for row in result if row["signed_clearance_m"] <= 0],
        closest_pairs=result[:maximum_results], sphere_count=sum(len(s) for _, s in groups),
        interpretation="planning sphere proxies vs exported world, not MuJoCo physical contacts",
        includes_attachment=attached is not None, robot_sphere_buffer_m=robot_sphere_buffer)


class PandaMJCF:
    """Translate the installed Panda arm and PandaGripper into a planning model."""

    def __init__(self, output_dir, assets_dir=None, max_robot_spheres=MAX_ROBOT_SPHERES):
        import trimesh
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if assets_dir is None:
            spec = importlib.util.find_spec("robosuite")
            if spec is None or spec.origin is None:
                raise RuntimeError("robosuite is required to locate the exact MJCF model")
            assets_dir = Path(spec.origin).parent / "models/assets"
        assets_dir = Path(assets_dir)
        robot_path = assets_dir / "robots/panda/robot.xml"
        gripper_path = assets_dir / "grippers/panda_gripper.xml"
        robot = ET.parse(robot_path).getroot()
        gripper = ET.parse(gripper_path).getroot()
        self.sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (robot_path, gripper_path)}
        self.joints = []
        self.spheres = {}
        self.meshes = []
        self.urdf = ET.Element("robot", name="robosuite_panda_bridge")
        self.arm_limits = []
        self.joint_axes = {}
        self.max_robot_spheres = int(max_robot_spheres)
        if not 16 <= self.max_robot_spheres < SELF_COLLISION_SPHERE_LIMIT:
            raise ValueError("max_robot_spheres must be between 16 and 1023")
        meshes = {}
        for root, folder in ((robot, robot_path.parent), (gripper, gripper_path.parent)):
            for m in root.findall("asset/mesh"):
                meshes[m.get("name")] = (folder / m.get("file")).resolve()
        names = {f"link{i}": f"panda_link{i}" for i in range(8)}
        names.update(right_hand="panda_hand", right_gripper="gripper_mount",
                     leftfinger="panda_leftfinger", rightfinger="panda_rightfinger",
                     eef="bridge_eef", finger_joint1_tip="finger1_tip", finger_joint2_tip="finger2_tip")
        geometry_specs = []

        def origin(parent, t):
            ET.SubElement(parent, "origin", xyz=" ".join(map(str, t[:3, 3])),
                          rpy=" ".join(map(str, Rotation.from_matrix(t[:3, :3]).as_euler("xyz"))))

        def visit(body, parent=None):
            old_name = body.get("name")
            name = names.get(old_name, old_name)
            link = ET.SubElement(self.urdf, "link", name=name)
            js = body.findall("joint")
            if len(js) > 1:
                raise ValueError("Only one MJCF joint per Panda body is supported")
            t = xml_transform(body)
            if parent is not None:
                j = js[0] if js else None
                if j is not None and np.linalg.norm(np.fromstring(j.get("pos", "0 0 0"), sep=" ")) > 1e-12:
                    raise ValueError("Nonzero MJCF joint anchor requires a split transform")
                kind = "fixed" if j is None else ("prismatic" if j.get("type", "hinge") == "slide" else "revolute")
                jname = f"{name}_fixed" if j is None else ("panda_" + j.get("name"))
                elem = ET.SubElement(self.urdf, "joint", name=jname, type=kind)
                ET.SubElement(elem, "parent", link=parent)
                ET.SubElement(elem, "child", link=name)
                origin(elem, t)
                axis = np.array([0., 0., 1.])
                limits = None
                if j is not None:
                    if float(j.get("ref", "0")) != 0:
                        raise ValueError("Nonzero MJCF joint ref is unsupported")
                    axis = np.fromstring(j.get("axis", "0 0 1"), sep=" ")
                    axis /= np.linalg.norm(axis)
                    limits = np.fromstring(j.get("range"), sep=" ")
                    ET.SubElement(elem, "axis", xyz=" ".join(map(str, axis)))
                    ET.SubElement(elem, "limit", lower=str(limits[0]), upper=str(limits[1]), effort="87", velocity="2.175")
                    self.joint_axes[jname] = axis
                    if jname in ARM_NAMES:
                        self.arm_limits.append(limits.tolist())
                self.joints.append(dict(parent=parent, child=name, name=jname, kind=kind, origin=t, axis=axis))
            for geom in body.findall("geom"):
                # robosuite marks collision geometry group=0; visual geoms group=1.
                if geom.get("group", "0") != "0":
                    continue
                g_t = xml_transform(geom)
                geom_type = geom.get("type", "sphere")
                col = ET.SubElement(link, "collision")
                origin(col, g_t)
                shape = ET.SubElement(col, "geometry")
                if geom_type == "mesh":
                    path = meshes[geom.get("mesh")]
                    mesh = trimesh.load(str(path), force="mesh", process=False)
                    self.meshes.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    ET.SubElement(shape, "mesh", filename=str(path))
                    bounds = mesh.bounds
                elif geom_type == "box":
                    half = np.fromstring(geom.get("size"), sep=" ")[:3]
                    bounds = np.stack((-half, half))
                    ET.SubElement(shape, "box", size=" ".join(map(str, 2*half)))
                else:
                    raise ValueError(f"Unsupported Panda collision geometry: {geom_type}")
                if name != "panda_link0":
                    geometry_specs.append(dict(link=name, bounds=np.asarray(bounds, float),
                        transform=g_t, desired_cell_size=.005 if "finger" in name else .025))
            for child in body.findall("body"):
                visit(child, name)

        root = robot.find("worldbody/body/body[@name='link0']")
        if root is None:
            raise ValueError("Unexpected Panda MJCF root hierarchy")
        visit(root)
        visit(gripper.find("worldbody/body"), "panda_hand")
        # Include a site transform if the installed model moves grip_site from eef.
        site = gripper.find(".//site[@name='grip_site']")
        if site is None:
            raise ValueError("PandaGripper has no grip_site")
        site_t = xml_transform(site)
        ET.SubElement(self.urdf, "link", name="control_site")
        site_joint = ET.SubElement(self.urdf, "joint", name="control_site_fixed", type="fixed")
        ET.SubElement(site_joint, "parent", link="bridge_eef")
        ET.SubElement(site_joint, "child", link="control_site")
        origin(site_joint, site_t)
        self.joints.append(dict(parent="bridge_eef", child="control_site", name="control_site_fixed",
                                kind="fixed", origin=site_t, axis=np.array([0., 0., 1.])))
        # The cuRobo self-collision CUDA kernel has a hard nspheres < 1024
        # limit, including reserved attachment slots. Allocate a global robot
        # budget before constructing any CUDA tensors. Every voxel still fits
        # entirely in its covering sphere; coarsening cannot leave gaps.
        if len(geometry_specs) > self.max_robot_spheres:
            raise ValueError("Robot sphere budget cannot cover one cell per collision geometry")

        def grid_count(spec, scale):
            dims = np.maximum(spec["bounds"][1]-spec["bounds"][0], 1e-6)
            return np.maximum(1, np.ceil(dims/(spec["desired_cell_size"]*scale)).astype(int))

        def total_count(scale):
            return sum(int(np.prod(grid_count(spec, scale))) for spec in geometry_specs)

        low, high = 0., 1.
        while total_count(high) > self.max_robot_spheres:
            low, high = high, high*2
        # Find the finest common resolution scale within the budget. The finer
        # requested finger resolution is retained relative to the arm meshes.
        for _ in range(40):
            middle = (low+high)/2
            if total_count(middle) > self.max_robot_spheres:
                low = middle
            else:
                high = middle
        resolution_scale = high*(1+1e-10)
        geom_spheres, resolution_report = {}, []
        for spec in geometry_specs:
            size = spec["desired_cell_size"]*resolution_scale
            cloud = box_spheres(spec["bounds"], cell_size=size, max_cells=self.max_robot_spheres)
            counts = grid_count(spec, resolution_scale)
            dims = np.maximum(spec["bounds"][1]-spec["bounds"][0], 1e-6)
            resolution_report.append(dict(link=spec["link"], grid_counts=counts.tolist(),
                sphere_count=len(cloud), requested_cell_size_m=spec["desired_cell_size"],
                maximum_grid_spacing_m=size, actual_grid_spacing_m=(dims/counts).tolist(),
                covering_sphere_radius_m=float(cloud[0, 3])))
            g_t = spec["transform"]
            cloud[:, :3] = cloud[:, :3] @ g_t[:3, :3].T + g_t[:3, 3]
            geom_spheres.setdefault(spec["link"], []).append(cloud)
        # Fixed finger-tip boxes are represented on the moving finger link.
        fixed_parent = {j["child"]: j for j in self.joints}
        for link_name, clouds in geom_spheres.items():
            if link_name == "panda_link0":
                continue
            target_name = link_name
            cloud = np.concatenate(clouds)
            while target_name in ("finger1_tip", "finger2_tip", "gripper_mount"):
                joint = fixed_parent[target_name]
                cloud[:, :3] = cloud[:, :3] @ joint["origin"][:3, :3].T + joint["origin"][:3, 3]
                target_name = joint["parent"]
            self.spheres.setdefault(target_name, []).extend(
                {"center": row[:3].tolist(), "radius": float(row[3])} for row in cloud)
        self.urdf_path = self.output_dir / "robosuite_panda.urdf"
        ET.indent(self.urdf)
        ET.ElementTree(self.urdf).write(self.urdf_path, encoding="unicode", xml_declaration=True)
        (self.output_dir / "ROBOT_SOURCE.json").write_text(json.dumps(dict(
            source_xml=self.sources, collision_meshes=self.meshes, collision_representation="complete collision-geom AABB grid sphere cover",
            fixed_base_link0_omitted=True, sphere_counts={k: len(v) for k, v in self.spheres.items()},
            total_robot_spheres=sum(map(len, self.spheres.values())), max_robot_spheres=self.max_robot_spheres,
            reserved_attachment_spheres=DEFAULT_ATTACHMENT_SLOTS,
            self_collision_kernel_sphere_limit_exclusive=SELF_COLLISION_SPHERE_LIMIT,
            resolution_scale=resolution_scale, geometry_resolutions=resolution_report), indent=2) + "\n")

    def fk(self, q_arm, q_fingers=(.04, -.04)):
        values = dict(zip(ARM_NAMES + FINGER_NAMES, [*q_arm, *q_fingers]))
        frames = {"panda_link0": np.eye(4)}
        pending = list(self.joints)
        while pending:
            changed = False
            for j in pending[:]:
                if j["parent"] not in frames:
                    continue
                move = np.eye(4)
                q = values.get(j["name"], 0.)
                if j["kind"] == "revolute":
                    move[:3, :3] = Rotation.from_rotvec(j["axis"] * q).as_matrix()
                elif j["kind"] == "prismatic":
                    move[:3, 3] = j["axis"] * q
                frames[j["child"]] = frames[j["parent"]] @ j["origin"] @ move
                pending.remove(j)
                changed = True
            if not changed:
                raise ValueError("Disconnected generated Panda URDF")
        return frames

    def robot_config(self, fingers=(.04, -.04), attachment_slots=DEFAULT_ATTACHMENT_SLOTS):
        robot_count = sum(map(len, self.spheres.values()))
        if not 1 <= int(attachment_slots) or robot_count+int(attachment_slots) >= SELF_COLLISION_SPHERE_LIMIT:
            raise ValueError(f"cuRobo self-collision requires <{SELF_COLLISION_SPHERE_LIMIT} total spheres; "
                             f"robot={robot_count}, attachment_slots={attachment_slots}")
        # Native Franka self-collision exclusions are kept explicit; adjacent
        # collision meshes naturally intersect at their mechanical connections.
        ignore = {
            "panda_link1": ["panda_link2", "panda_link3", "panda_link4"],
            "panda_link2": ["panda_link3", "panda_link4"],
            "panda_link3": ["panda_link4", "panda_link6"],
            "panda_link4": ["panda_link5", "panda_link6", "panda_link7"],
            "panda_link5": ["panda_link6", "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger"],
            "panda_link6": ["panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger", "attached_object"],
            "panda_link7": ["panda_hand", "panda_leftfinger", "panda_rightfinger", "attached_object"],
            "panda_hand": ["panda_leftfinger", "panda_rightfinger", "attached_object"],
            "panda_leftfinger": ["panda_rightfinger", "attached_object"],
            "panda_rightfinger": ["attached_object"],
        }
        return {"kinematics": dict(
            use_usd_kinematics=False, urdf_path=str(self.urdf_path), asset_root_path=str(self.output_dir),
            base_link="panda_link0", ee_link="control_site", collision_link_names=[*self.spheres, "attached_object"],
            collision_spheres=self.spheres, collision_sphere_buffer=.001, use_global_cumul=True,
            extra_collision_spheres={"attached_object": attachment_slots},
            self_collision_ignore=ignore, self_collision_buffer={name: 0. for name in [*self.spheres, "attached_object"]},
            link_names=[*self.spheres, "control_site"], mesh_link_names=[],
            lock_joints=dict(zip(FINGER_NAMES, map(float, fingers))),
            extra_links={"attached_object": dict(parent_link_name="control_site", link_name="attached_object",
                fixed_transform=[0, 0, 0, 1, 0, 0, 0], joint_type="FIXED", joint_name="attach_joint")},
            cspace=dict(joint_names=ARM_NAMES + FINGER_NAMES,
                retract_config=[0., -.5, 0., -2., 0., 1.5, .785, *fingers],
                null_space_weight=[1.] * 9, cspace_distance_weight=[1.] * 9,
                max_acceleration=10., max_jerk=200.))}
