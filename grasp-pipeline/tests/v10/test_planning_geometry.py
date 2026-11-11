"""CPU checks for the planner model; these tests never create a CUDA context."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.planning.planning_geometry import (PandaMJCF, box_spheres, pose_list, transform,
    MAX_ROBOT_SPHERES, SELF_COLLISION_SPHERE_LIMIT, DEFAULT_ATTACHMENT_SLOTS)
from graspgen_libero_bridge.planning.planner_worker import densify, finger_values


class GeometryTests(unittest.TestCase):
    def test_pose_round_trip_and_invalid_rotation(self):
        t = np.eye(4)
        t[:3, :3] = Rotation.from_euler("xyz", [.4, -.2, .8]).as_matrix()
        t[:3, 3] = [.1, .2, .3]
        np.testing.assert_allclose(transform(pose_list(t)), t, atol=1e-12)
        t[0, 0] = 5
        with self.assertRaises(ValueError):
            transform(t)

    def test_grid_covers_box(self):
        lo, hi = np.array([-.02, -.03, -.05]), np.array([.05, .08, .03])
        spheres = box_spheres(np.stack((lo, hi)), max_cells=35)
        points = np.random.default_rng(123).uniform(lo, hi, size=(5000, 3))
        distance = np.linalg.norm(points[:, None, :]-spheres[None, :, :3], axis=-1)
        self.assertTrue((distance <= spheres[None, :, 3]+1e-12).any(axis=1).all())
        self.assertLessEqual(len(spheres), 35)

    def test_joint_path_densification_and_finger_signs(self):
        q = densify(np.array([[0.] * 7, [.19] * 7]), maximum_joint_step=.025)
        self.assertLessEqual(np.max(np.abs(np.diff(q, axis=0))), .025001)
        np.testing.assert_allclose(q[-1], [.19] * 7)
        np.testing.assert_allclose(finger_values({"gripper_width": .06}), [.03, -.03])
        with self.assertRaises(ValueError):
            finger_values({"q_fingers": [.03, .03]})

    def test_generated_urdf_fk_matches_mujoco_control_site(self):
        import mujoco
        override = os.environ.get("ROBOSUITE_ASSETS")
        spec = importlib.util.find_spec("robosuite") if override is None else None
        if override is None and (spec is None or spec.origin is None):
            self.fail("Set ROBOSUITE_ASSETS to the sim environment's robosuite/models/assets directory")
        assets = Path(override) if override else Path(spec.origin).parent / "models/assets"
        # Compile an independent minimal MJCF with the actual installed arm and
        # gripper; MuJoCo handles its own quaternion and joint conventions.
        compiled = ET.Element("mujoco", model="panda_fk_reference")
        ET.SubElement(compiled, "compiler", angle="radian")
        asset = ET.SubElement(compiled, "asset")
        trees = []
        for rel in ("robots/panda/robot.xml", "grippers/panda_gripper.xml"):
            source = assets / rel
            tree = ET.parse(source).getroot()
            trees.append(tree)
            for elem in tree.find("asset"):
                elem = copy.deepcopy(elem)
                if elem.tag == "mesh":
                    elem.set("file", str((source.parent / elem.get("file")).resolve()))
                asset.append(elem)
        world = copy.deepcopy(trees[0].find("worldbody"))
        world.find(".//body[@name='right_hand']").append(copy.deepcopy(trees[1].find("worldbody/body")))
        compiled.append(world)
        mj_model = mujoco.MjModel.from_xml_string(ET.tostring(compiled, encoding="unicode"))
        data = mujoco.MjData(mj_model)
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        samples = [[0, -.5, 0, -2, 0, 1.5, .785], [.2, -.4, -.1, -1.8, .3, 1.6, .6],
                   [-.3, -.2, .2, -2.2, -.2, 1.3, .9], [.1, -.7, .4, -1.6, .1, 1.9, .4]]
        with tempfile.TemporaryDirectory(prefix="libero_bridge_fk_") as tmp:
            robot = PandaMJCF(tmp, assets_dir=assets)
            sphere_count = sum(map(len, robot.spheres.values()))
            self.assertLessEqual(sphere_count, MAX_ROBOT_SPHERES)
            self.assertLess(sphere_count+DEFAULT_ATTACHMENT_SLOTS, SELF_COLLISION_SPHERE_LIMIT)
            robot.robot_config()  # Also enforces the limit before CUDA startup.
            with self.assertRaises(ValueError):
                robot.robot_config(attachment_slots=SELF_COLLISION_SPHERE_LIMIT-sphere_count)
            for q in samples:
                for name, val in zip([f"joint{i}" for i in range(1, 8)], q):
                    joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    data.qpos[mj_model.jnt_qposadr[joint_id]] = val
                mujoco.mj_forward(mj_model, data)
                actual = robot.fk(q)["control_site"]
                np.testing.assert_allclose(actual[:3, 3], data.site_xpos[site_id], atol=1e-8)
                np.testing.assert_allclose(actual[:3, :3], data.site_xmat[site_id].reshape(3, 3), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
