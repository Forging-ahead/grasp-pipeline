"""Regression tests for termination, active references and physical recovery."""
import unittest
from types import SimpleNamespace
import numpy as np
from graspgen_libero_bridge.control.controller import Controller, ExecutionError, ContactMonitor
import test_cartesian as fixtures
import tempfile
from pathlib import Path
from graspgen_libero_bridge.policy.evaluate import PickPlace


class ServoTests(unittest.TestCase):
    def fixture(self, config=None):
        c = Controller.__new__(Controller)
        pose = np.eye(4)
        joint = np.zeros(7)
        active = SimpleNamespace(controller=SimpleNamespace(initial_joint=np.zeros(7)))
        stale = SimpleNamespace(controller=SimpleNamespace(initial_joint=np.zeros(7)))
        c.adapter = SimpleNamespace(config=config or {}, robot=stale,
            task_env=SimpleNamespace(robots=[active]),
            state=lambda: {"world_T_eef": pose.copy(), "q_arm": joint.copy()})
        c.recorder = SimpleNamespace(phase="initial")
        c.position_scale, c.rotation_scale = .05, .5
        c.joint_limits = np.tile([-1., 1.], (7, 1))
        c.reference_margin = .02
        c.events = []
        c.joint_recovery_goal = lambda *a: pose.copy()
        return c, pose, joint

    def test_arrival_on_last_allowed_action_is_success(self):
        c, pose, _ = self.fixture()
        desired = np.eye(4); desired[0, 3] = .01
        c.step = lambda *a: pose.__setitem__((0, 3), .01)
        c.move_to(desired, max_steps=1)

    def test_measured_subgoal_can_stop_a_contact_servo_before_pose_arrival(self):
        c,pose,_=self.fixture();observed=[]
        desired=np.eye(4);desired[0,3]=.01
        c.step=lambda *a:observed.append(1)
        c.move_to(desired,max_steps=3,stop_when=lambda:len(observed)==3)
        self.assertEqual(len(observed),3)
        self.assertEqual(pose[0,3],0.)
        self.assertEqual(c.events[-1]['event'],'observed_subgoal_reached')

    def test_measured_subgoal_does_not_bypass_an_execution_collision(self):
        c,_,_=self.fixture();observed=[]
        def collision(*args):
            observed.append(1)
            raise ExecutionError('contact',code='forbidden_contact')
        c.step=collision;desired=np.eye(4);desired[0,3]=.01
        with self.assertRaises(ExecutionError) as error:
            c.move_to(desired,max_steps=3,stop_when=lambda:bool(observed))
        self.assertEqual(error.exception.code,'forbidden_contact')

    def test_slide_feedback_accepts_small_wrist_load_error_without_stopping_translation(self):
        from scipy.spatial.transform import Rotation
        from graspgen_libero_bridge.adapters.libero.surface_contact import feedback_tolerances
        def run(old):
            c,pose,_=self.fixture();pose[:3,:3]=Rotation.from_euler('z',.002).as_matrix()
            desired=np.eye(4);desired[0,3]=.004
            settings=feedback_tolerances(dict(joint_type=2),pose,desired)
            if old:settings['rotation_tolerance']=.0008
            def step(action,*args):pose[0,3]+=.5*action[0]*.05
            c.step=step;c.move_to(desired,max_steps=20,**settings)
            return abs(pose[0,3]-.004)
        with self.assertRaises(ExecutionError):run(True)
        self.assertLessEqual(run(False),.0008)

    def test_bounded_integral_removes_constant_load_error_without_relaxing_tolerance(self):
        def run(gain):
            c, pose, _ = self.fixture({"position_integral_gain": gain})
            desired = np.eye(4); desired[0, 3] = .01
            def step(action, *a):
                pose[0, 3] += .25 * action[0] * .05 - .0005
            c.step = step
            c.move_to(desired, max_steps=120, position_tolerance=.001)
            return abs(pose[0, 3]-.01)
        with self.assertRaises(ExecutionError) as exc:
            run(0.)
        self.assertEqual(exc.exception.code, "waypoint_timeout")
        self.assertLessEqual(run(.8), .001)

    def test_reference_updates_active_controller_and_preserves_measurement(self):
        c, _, joint = self.fixture()
        joint[0] = 1.000001
        ref = c.set_joint_reference(joint)
        self.assertEqual(ref[0], .98)
        np.testing.assert_array_equal(c.adapter.task_env.robots[0].controller.initial_joint, ref)
        np.testing.assert_array_equal(c.adapter.robot.controller.initial_joint, np.zeros(7))
        self.assertEqual(joint[0], 1.000001)

    def test_limit_recovery_moves_real_state_instead_of_clipping_input(self):
        c, _, joint = self.fixture(); joint[0] = 1.000001
        before = joint.copy(); calls = []
        def step(*a):
            calls.append(1)
            joint[:] += .2*(c.adapter.task_env.robots[0].controller.initial_joint-joint)
        c.step = step
        c.ensure_joint_margin("transfer")
        self.assertGreater(len(calls), 0)
        self.assertLess(joint[0], .9995)
        np.testing.assert_array_equal(c.events[-1]["measured_before"], before)

    def test_recovery_does_not_accept_stationary_or_large_limit_violation(self):
        c, _, joint = self.fixture({"joint_recovery_steps": 3}); joint[0] = 1.000001
        c.step = lambda *a: None
        with self.assertRaises(ExecutionError) as exc:
            c.ensure_joint_margin("transfer")
        self.assertEqual(exc.exception.code, "joint_margin_recovery_failed")
        joint[0] = 1.1
        with self.assertRaises(ExecutionError) as exc:
            c.ensure_joint_margin("transfer")
        self.assertEqual(exc.exception.code, "joint_limit_excursion")

    def test_stalled_servo_reports_control_not_grasp_failure(self):
        c, _, _ = self.fixture(); c.step = lambda *a: None
        desired = np.eye(4); desired[0, 3] = .01
        with self.assertRaises(ExecutionError) as exc:
            c.move_to(desired, max_steps=3, phase="approach")
        self.assertEqual(exc.exception.record()["category"], "control")
        self.assertEqual(exc.exception.record()["phase"], "approach")

    def test_invalid_margin_is_not_accepted_as_recovery(self):
        c, _, joint = self.fixture({"planning_joint_margin_rad": -.01})
        joint[0] = 1.000001
        with self.assertRaises(ValueError):
            c.ensure_joint_margin("transfer")


@unittest.skipUnless(fixtures.AVAILABLE, "requires MuJoCo and SciPy")
class PreflightRestorationTests(unittest.TestCase):
    def test_joint_preview_restores_integrator_inputs(self):
        helper = fixtures.CartesianPlannerTests()
        adapter, sim = helper.make_adapter()
        robot = adapter.task_env.robots[0]
        robot.controller = SimpleNamespace(update=lambda **k: None, reset_goal=lambda: None)
        adapter.task_env._update_observables = lambda **k: None
        adapter.state = lambda: {"world_T_eef": np.eye(4)}
        values = helper.before(sim)
        monitor = ContactMonitor.__new__(ContactMonitor)
        monitor.adapter, monitor.target, monitor.held = adapter, "target", False
        monitor.check = lambda *a: []
        result = monitor.validate_path(np.array([[0.]*7, [.01]*7]))
        self.assertTrue(result["passed"])
        helper.assert_restored(sim, values)

    def test_recovery_fk_goal_is_measured_without_mutating_physics(self):
        helper = fixtures.CartesianPlannerTests()
        adapter, sim = helper.make_adapter()
        robot = adapter.task_env.robots[0]
        robot.controller = SimpleNamespace(update=lambda **k: None, reset_goal=lambda: None)
        adapter.task_env._update_observables = lambda **k: None
        adapter.state = lambda: {"world_T_eef": np.eye(4)}
        monitor = ContactMonitor.__new__(ContactMonitor)
        monitor.adapter, monitor.target, monitor.held = adapter, "target", False
        monitor.check = lambda *a: []
        c = Controller.__new__(Controller); c.adapter, c.monitor = adapter, monitor
        target = np.zeros(7); target[0] = .01
        values = helper.before(sim)
        goal = c.joint_recovery_goal(np.zeros(7), target, True)
        self.assertAlmostEqual(goal[0, 3], .01)
        helper.assert_restored(sim, values)


class OutcomeTests(unittest.TestCase):
    def simulate_abort(self, official):
        with tempfile.TemporaryDirectory() as tmp:
            job = PickPlace({"libero_repo": "/unused"}, Path(tmp)/"run")
            def stop():
                raise ExecutionError("Path rejected", code="selection_budget_exhausted",
                                     category="planning", phase="select_candidate")
            job.adapter = SimpleNamespace(reset=stop, close=lambda: None,
                env=object(), obs={}, step_count=0, success_seen=official,
                check_success=lambda: official)
            return job.run()

    def test_pre_execution_abort_leaves_grasp_unmeasured(self):
        result = self.simulate_abort(False)
        self.assertEqual(result["status"], "aborted")
        self.assertIsNone(result["grasp_success"])
        self.assertFalse(result["grasp_attempt_completed"])
        self.assertEqual(result["grasp_assessment"], "not_attempted")
        self.assertEqual(result["failure"]["category"], "planning")

    def test_official_success_survives_pipeline_abort(self):
        result = self.simulate_abort(True)
        self.assertTrue(result["task_success"])
        self.assertFalse(result["pipeline_success"])
        self.assertEqual(result["status"], "aborted")
