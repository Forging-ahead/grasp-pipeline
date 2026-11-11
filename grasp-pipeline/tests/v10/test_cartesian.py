"""CPU-only MuJoCo tests for contact-phase IK and preview state restoration.

No LIBERO environment, renderer, CUDA device, or model weights are created.
The small model has seven joints, an actual site Jacobian, a free-joint target,
and an optional real collision wall. It tests the planner's integration contract.
"""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from graspgen_libero_bridge.planning.cartesian import plan_cartesian, preview_path_endpoint


AVAILABLE = (importlib.util.find_spec("mujoco") is not None and
             importlib.util.find_spec("scipy") is not None)


class _Model:
    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def joint_name2id(self, name):
        return self._model.joint(name).id


class _Data:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        return getattr(self._data, {"body_xpos": "xpos", "body_xmat": "xmat"}.get(name, name))


class _Sim:
    def __init__(self, model, data):
        self.model, self.data = _Model(model), _Data(data)
        self.steps = 0

    def get_state(self):
        data = self.data._data
        return SimpleNamespace(time=data.time, qpos=data.qpos.copy(), qvel=data.qvel.copy())

    def set_state(self, state):
        data = self.data._data
        data.time = state.time
        data.qpos[:] = state.qpos
        data.qvel[:] = state.qvel

    def forward(self):
        import mujoco
        mujoco.mj_forward(self.model._model, self.data._data)

    def step(self):
        self.steps += 1
        raise AssertionError("Cartesian planning must not step dynamics")


class _Monitor:
    def __init__(self, sim, held=False):
        self.sim, self.held, self.target = sim, held, "target"
        self.allowed_support_geoms = {"existing_support_permission"}
        self.calls = []
        self.collision_count = 0

    def check(self, allow_finger_target=False):
        self.calls.append(bool(allow_finger_target))
        failures = []
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            if float(contact.dist) < -1e-7:
                failures.append({"penetration_m": -float(contact.dist)})
        self.collision_count += len(failures)
        return failures


@unittest.skipUnless(AVAILABLE, "CPU MuJoCo and SciPy required")
class CartesianPlannerTests(unittest.TestCase):
    def make_adapter(self, wall=False, moving_wall=False):
        import mujoco
        # Three translations plus four rotations give a nonsingular site task
        # and one redundant joint without introducing renderer/model assets.
        axes = ["1 0 0", "0 1 0", "0 0 1", "1 0 0", "0 1 0", "0 0 1", "0 0 1"]
        xml = ['<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>']
        if wall:
            xml.append('<geom name="wall" type="box" pos="0 0 .025" size=".025 .025 .002"/>')
        for i, axis in enumerate(axes):
            kind = "slide" if i < 3 else "hinge"
            xml.append(f'<body name="link{i}"><joint name="joint{i}" type="{kind}" axis="{axis}" range="-1 1"/><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>')
        xml.append('<site name="control"/><geom name="probe" type="sphere" size=".004"/>')
        xml.extend(['</body>'] * 7)
        xml.append('<body name="target" pos=".15 0 0"><freejoint/><geom type="sphere" size=".006" mass=".1"/></body>')
        if moving_wall:
            xml.append('<body name="fixture" pos="0 0 .025"><joint name="fixture_slide" type="slide" axis="0 0 1" range="-.1 .1"/><geom type="box" size=".025 .025 .002" mass="1"/></body>')
        xml.append('</worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_string("".join(xml))
        data = mujoco.MjData(model)
        data.time = 1.25
        data.qvel[:] = np.linspace(.0001, .0002, model.nv)
        data.qfrc_applied[:] = np.linspace(.01, .02, model.nv)
        sim = _Sim(model, data)
        sim.forward()
        data.qacc_warmstart[:] = np.linspace(.2, .3, model.nv)
        site = model.site("control").id
        target = model.body("target").id
        robot = SimpleNamespace(_ref_joint_pos_indexes=np.arange(7),
                                _ref_joint_vel_indexes=np.arange(7), eef_site_id=site,
                                robot_model=SimpleNamespace(joints=[f"joint{i}" for i in range(7)]))

        def target_pose(name):
            result = np.eye(4)
            result[:3, :3] = data.xmat[target].reshape(3, 3)
            result[:3, 3] = data.xpos[target]
            return result

        adapter = SimpleNamespace(env=SimpleNamespace(sim=sim), target_object="target",
                                  task_env=SimpleNamespace(robots=[robot], obj_body_id={"target": target}),
                                  target_pose=target_pose)
        return adapter, sim

    def before(self, sim):
        data = sim.data._data
        return {name: np.array(getattr(data, name), copy=True)
                for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "qfrc_applied")}

    def assert_restored(self, sim, values):
        for name, value in values.items():
            np.testing.assert_array_equal(getattr(sim.data._data, name), value, err_msg=name)
        self.assertEqual(sim.data.time, 1.25)
        self.assertEqual(sim.steps, 0)

    def test_ik_pose_and_integrator_inputs_restored(self):
        from scipy.spatial.transform import Rotation
        adapter, sim = self.make_adapter()
        monitor = _Monitor(sim)
        values = self.before(sim)
        goal = np.eye(4)
        goal[:3, 3] = [.01, -.005, .04]
        goal[:3, :3] = Rotation.from_rotvec([.02, -.01, .025]).as_matrix()
        with tempfile.TemporaryDirectory() as directory:
            reply = plan_cartesian(adapter, monitor, goal, directory)
            self.assertTrue(reply["success"], reply)
            self.assertTrue(reply["state_restored"])
            with np.load(reply["path_file"], allow_pickle=False) as path:
                self.assertEqual(path["q_arm"].shape[1], 7)
                self.assertEqual(len(path["q_arm"]), len(path["world_T_eef"]))
                self.assertLess(np.linalg.norm(path["world_T_eef"][-1, :3, 3]-goal[:3, 3]), .0004)
                self.assertLess(Rotation.from_matrix(goal[:3, :3] @ path["world_T_eef"][-1, :3, :3].T).magnitude(), .003)
        self.assertTrue(all(monitor.calls))
        self.assertEqual(monitor.allowed_support_geoms, {"existing_support_permission"})
        self.assert_restored(sim, values)

    def test_real_mujoco_collision_blocks_motion_and_restores(self):
        adapter, sim = self.make_adapter(wall=True)
        monitor = _Monitor(sim)
        values = self.before(sim)
        goal = np.eye(4)
        goal[2, 3] = .04
        with tempfile.TemporaryDirectory() as directory:
            reply = plan_cartesian(adapter, monitor, goal, directory)
            self.assertFalse(reply["success"], reply)
            self.assertNotEqual(reply["reason"], "cartesian_planner_error", reply)
            self.assertNotIn("path_file", reply)
        self.assertGreater(monitor.collision_count, 0)
        self.assertTrue(reply["state_restored"])
        self.assert_restored(sim, values)

    def test_held_target_follows_only_in_preview(self):
        adapter, sim = self.make_adapter()
        values = self.before(sim)
        goal = np.eye(4)
        goal[2, 3] = .035
        with tempfile.TemporaryDirectory() as directory:
            reply = plan_cartesian(adapter, _Monitor(sim, held=True), goal, directory)
            self.assertTrue(reply["success"], reply)
            with np.load(reply["path_file"], allow_pickle=False) as path:
                relative = np.linalg.inv(path["world_T_eef"]) @ path["world_T_object"]
                np.testing.assert_allclose(relative, np.broadcast_to(relative[0], relative.shape), atol=1e-10)
                self.assertGreater(path["world_T_object"][-1, 2, 3], .034)
        self.assert_restored(sim, values)

    def test_monitor_exception_restores_physics(self):
        adapter, sim = self.make_adapter()
        values = self.before(sim)

        class BrokenMonitor(_Monitor):
            def check(self, allow_finger_target=False):
                self.calls.append(allow_finger_target)
                if len(self.calls) >= 3:
                    raise RuntimeError("test monitor exception after mutation")
                return []

        goal = np.eye(4)
        goal[2, 3] = .02
        with tempfile.TemporaryDirectory() as directory:
            reply = plan_cartesian(adapter, BrokenMonitor(sim), goal, directory)
            self.assertFalse(reply["success"])
            self.assertEqual(reply["reason"], "cartesian_planner_error")
            self.assertTrue(reply["state_restored"])
            self.assertFalse((Path(directory)/"PATH.npz").exists())
        self.assert_restored(sim, values)

    def test_endpoint_continuation_moves_held_body_only_in_preview(self):
        adapter, sim = self.make_adapter()
        monitor = _Monitor(sim, held=True)
        values = self.before(sim)
        original_target = adapter.target_pose("target").copy()
        goal = np.eye(4)
        goal[2, 3] = .02
        with tempfile.TemporaryDirectory() as directory:
            reply = plan_cartesian(adapter, monitor, goal, Path(directory)/"first")
            self.assertTrue(reply["success"], reply)
            with preview_path_endpoint(adapter, monitor, reply["path_file"]):
                self.assertAlmostEqual(adapter.target_pose("target")[2, 3], .02, delta=.0004)
                np.testing.assert_allclose(adapter.target_pose("target")[:2, 3], original_target[:2, 3], atol=.0004)
                continued = goal.copy()
                continued[0, 3] = .015
                second = plan_cartesian(adapter, monitor, continued, Path(directory)/"second")
                self.assertTrue(second["success"], second)
                self.assertAlmostEqual(sim.data.qpos[2], .02, delta=.0004)
        self.assert_restored(sim, values)
        np.testing.assert_array_equal(adapter.target_pose("target"), original_target)

    def test_endpoint_exception_restores_physics(self):
        adapter, sim = self.make_adapter()
        monitor = _Monitor(sim, held=True)
        values = self.before(sim)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"PATH.npz"
            q = np.zeros((2, 7)); q[-1, 2] = .03
            np.savez(path, q_arm=q)
            with self.assertRaisesRegex(RuntimeError, "preview exception"):
                with preview_path_endpoint(adapter, monitor, path):
                    raise RuntimeError("preview exception")
        self.assert_restored(sim, values)

    def test_coupled_slide_avoids_false_start_collision_and_restores(self):
        from graspgen_libero_bridge.adapters.libero.articulation import ArticulationPreviewMonitor
        from graspgen_libero_bridge.planning.cartesian import _snapshot, _restore
        adapter, sim = self.make_adapter(moving_wall=True)
        values = self.before(sim)
        joint = sim.model.joint_name2id('fixture_slide')
        address = int(sim.model.jnt_qposadr[joint])
        info = dict(joint_type=2, axis=[0.,0.,1.], q_address=address)
        goal = np.eye(4); goal[2,3] = -.02
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _snapshot(sim)
            sim.data.qpos[address] = -.02
            sim.forward()
            old = plan_cartesian(adapter, _Monitor(sim), goal, Path(directory)/'old')
            self.assertFalse(old['success'])
            self.assertEqual(old['reason'], 'cartesian_start_collision')
            _restore(sim, snapshot)
            monitor = ArticulationPreviewMonitor(adapter, _Monitor(sim), info, np.eye(4), 0., -.04)
            goal[2,3] = -.04
            new = plan_cartesian(adapter, monitor, goal, Path(directory)/'coupled')
            self.assertTrue(new['success'], new)
            self.assertTrue(new['state_restored'])
        self.assert_restored(sim, values)

    def test_coupled_hinge_follows_rotation_and_preserves_permissions(self):
        from graspgen_libero_bridge.adapters.libero.articulation import ArticulationPreviewMonitor
        from graspgen_libero_bridge.planning.cartesian import _snapshot, _restore
        adapter, sim = self.make_adapter(moving_wall=True)
        values = self.before(sim)
        address = int(sim.model.jnt_qposadr[sim.model.joint_name2id('fixture_slide')])
        base = _Monitor(sim)
        info = dict(joint_type=3, axis=[0.,0.,-1.], q_address=address)
        monitor = ArticulationPreviewMonitor(adapter, base, info, np.eye(4), .01, -.06)
        snapshot = _snapshot(sim)
        try:
            sim.data.qpos[5] = .03
            sim.forward()
            monitor.check(True)
            self.assertAlmostEqual(sim.data.qpos[address], -.02)
            self.assertEqual(base.calls, [True])
            self.assertEqual(monitor.allowed_support_geoms, {'existing_support_permission'})
        finally:
            _restore(sim, snapshot)
        self.assert_restored(sim, values)


if __name__ == "__main__":
    unittest.main()
