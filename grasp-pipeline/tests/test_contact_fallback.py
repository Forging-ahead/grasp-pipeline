"""A rejected continuation must trigger a different global approach branch."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock
import numpy as np

from graspgen_libero_bridge.policy.skills import ContactSkill
from graspgen_libero_bridge.common.errors import ExecutionError


class ContactFallbackTests(unittest.TestCase):
    def exercise(self, directory, engagement=True,free_roll=False):
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
        if free_roll:
            skill.info=dict(joint_type=3,cylindrical_handle=True,axis=np.array([0.,0.,1.]),handle_axis=np.array([0.,0.,1.]))
            skill.candidate_specs=[dict(free_handle_roll=True)]
        def native(pose,phase,contact=False,**kwargs):
            calls.append(phase)
            if free_roll:np.testing.assert_array_equal(kwargs['free_rotation_axis_world'],[0.,0.,1.])
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

    def test_invalid_global_start_is_not_retried_as_a_different_target(self):
        skill=object.__new__(ContactSkill)
        skill.native_plan=Mock(return_value=dict(success=False))
        skill.global_plan=Mock(return_value=dict(success=False,reason='invalid_start_state',status='WORLD_COLLISION'))
        with self.assertRaises(ExecutionError) as error:
            skill.approach([(np.eye(4),np.eye(4))]*2)
        self.assertEqual(error.exception.code,'contact_global_start_invalid')
        self.assertEqual(error.exception.details['status'],'WORLD_COLLISION')
        self.assertEqual(skill.global_plan.call_count,1)

    def test_round_handle_freedom_applies_to_approach_and_engagement(self):
        with tempfile.TemporaryDirectory() as directory:
            calls,followed=self.exercise(directory,free_roll=True)
        self.assertIn('approach',calls);self.assertIn('engage',calls);self.assertEqual(len(followed),2)

    def regrasp(self,directory,success_after,initial_error='contact_approach_exhausted'):
        skill=object.__new__(ContactSkill);skill.job=object();skill.out=Path(directory)
        skill.stage=dict(relation='open',predicate=['open','door']);skill.info=dict(joint_type=3)
        skill.planner=Mock();skill.c=SimpleNamespace(recorder=object())
        progress=[]
        skill.a=SimpleNamespace(task_env=SimpleNamespace(_eval_predicate=lambda goal:len(progress)>=success_after))
        skill._articulate_attempt=Mock(side_effect=ExecutionError('blocked',code=initial_error))
        def child(*args):
            value=Mock()
            def move(**kwargs):
                self.assertEqual(kwargs,dict(max_joint_segment=.45))
                index=len(progress);progress.append(index)
                return dict(initial_q=-index*.4,final_q=-(index+1)*.4,official_subgoal=False)
            value._articulate_attempt.side_effect=move
            return value
        with patch('graspgen_libero_bridge.policy.skills.ContactSkill',side_effect=child) as factory:
            if success_after<=4:
                result=skill.articulate();self.assertTrue(result['official_subgoal'])
            else:
                with self.assertRaises(ExecutionError) as error:skill.articulate()
                self.assertEqual(error.exception.code,'regrasp_budget_exhausted')
            self.assertEqual(factory.call_count,min(success_after,4))
        self.assertEqual(skill.stage['predicate'],['open','door'])

    def test_regrasp_partial_progress_never_counts_as_complete_success(self):
        with tempfile.TemporaryDirectory() as directory:self.regrasp(directory,success_after=5)

    def test_regrasp_stops_only_when_original_predicate_is_true(self):
        with tempfile.TemporaryDirectory() as directory:self.regrasp(directory,success_after=2)

    def test_proxy_blocked_global_start_can_still_try_local_segmented_contact(self):
        with tempfile.TemporaryDirectory() as directory:self.regrasp(directory,success_after=2,initial_error='contact_global_start_invalid')

    def test_access_regrasp_does_not_stop_at_the_earlier_open_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            skill=object.__new__(ContactSkill);skill.job=object();skill.out=Path(directory)
            skill.stage=dict(relation='open',predicate=['open','door'],requires_access=True)
            skill.info=dict(joint_type=3);skill.planner=None;skill.c=SimpleNamespace(recorder=object())
            completed=[]
            skill.a=SimpleNamespace(task_env=SimpleNamespace(_eval_predicate=lambda _:len(completed)>=2))
            skill._articulate_attempt=Mock(side_effect=ExecutionError('blocked',code='contact_approach_exhausted'))
            def child(*args):
                result=Mock()
                def leg(**kwargs):
                    index=len(completed);completed.append(index)
                    return dict(initial_q=-index*.4,final_q=-(index+1)*.4,required_access_reached=len(completed)>=4)
                result._articulate_attempt.side_effect=leg;return result
            with patch('graspgen_libero_bridge.policy.skills.ContactSkill',side_effect=child):
                result=skill.articulate()
            self.assertEqual(len(completed),4);self.assertTrue(result['required_access_reached'])


if __name__=='__main__':unittest.main()
