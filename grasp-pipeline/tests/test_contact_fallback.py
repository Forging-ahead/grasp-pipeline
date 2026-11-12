"""A rejected continuation must trigger a different global approach branch."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from graspgen_libero_bridge.policy.skills import ContactSkill
from graspgen_libero_bridge.common.errors import ExecutionError


class ContactFallbackTests(unittest.TestCase):
    def exercise(self, directory, engagement=True):
        local = Path(directory)/'local.npz'
        global_path = Path(directory)/'global.npz'
        np.savez(local,q_arm=np.zeros((1,7)))
        alternate = np.zeros((1,7));alternate[0,0]=.4
        np.savez(global_path,q_arm=alternate)
        sim = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(7)),forward=lambda:None)
        robot = SimpleNamespace(_ref_joint_pos_indexes=np.arange(7),
            controller=SimpleNamespace(update=lambda **kw:None,reset_goal=lambda:None))
        adapter = SimpleNamespace(env=SimpleNamespace(sim=sim),robot=robot,
            task_env=SimpleNamespace(_update_observables=lambda **kw:None))
        followed, calls = [], []
        skill = object.__new__(ContactSkill)
        skill.a = adapter
        skill.c = SimpleNamespace(follow_path=lambda file,**kw:followed.append(file))
        skill.sweep_goal = .1
        def native(pose,phase,contact=False):
            calls.append(phase)
            branch = global_path if sim.data.qpos[0]>.2 else local
            return dict(success=phase=='approach' or engagement,path_file=str(branch))
        def global_plan(pose,phase):
            calls.append(phase)
            return dict(success=True,path_file=str(global_path))
        def sweep(plan):
            calls.append('sweep')
            with np.load(plan['path_file']) as path:
                return bool(path['q_arm'][0,0]>.2)
        skill.native_plan, skill.global_plan, skill.preview_sweep = native,global_plan,sweep
        def restore(sim, state):sim.data.qpos[:]=state
        with patch('graspgen_libero_bridge.policy.skills._snapshot',lambda sim:sim.data.qpos.copy()), \
             patch('graspgen_libero_bridge.policy.skills._restore',restore):
            if engagement:
                self.assertEqual(skill.approach([(np.eye(4),np.eye(4))]),0)
            else:
                with self.assertRaises(ExecutionError):
                    skill.approach([(np.eye(4),np.eye(4))])
        np.testing.assert_array_equal(sim.data.qpos,np.zeros(7))
        return calls,followed

    def test_failed_local_continuation_tries_global_branch_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            calls,followed = self.exercise(directory)
        self.assertEqual(calls,['approach','engage','sweep','global_approach','engage','sweep'])
        self.assertEqual(len(followed),2)
        self.assertTrue(all(Path(p).name=='global.npz' for p in followed))

    def test_global_approach_does_not_bypass_rejected_engagement(self):
        with tempfile.TemporaryDirectory() as directory:
            calls,followed = self.exercise(directory,engagement=False)
        self.assertIn('global_approach',calls)
        self.assertFalse(followed)


if __name__=='__main__':unittest.main()
