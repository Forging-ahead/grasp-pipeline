"""Reject stale, partial or wrong-task provenance in a new full sweep."""
import importlib.util
from pathlib import Path

import pytest

P=Path(__file__).resolve().parents[1]/'tools/libero_evaluation/run_all.py'
spec=importlib.util.spec_from_file_location('libero_eval_harness',P)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def data():
    row={'key':'task','suite':'suite','task_name':'native_task'}
    files={'apps/libero/run_task.py':'driver','src/graspgen_libero_bridge/a.py':'a','src/graspgen_libero_bridge/b.py':'b','configs/policy/v10_overrides.json':'policy'}
    result=dict(row,initial_state_id=0,driver_sha256='driver',source_sha256={k:v for k,v in files.items() if k.startswith('src/')},original_goals_unchanged=True)
    return row,{'files':files},result

def test_exact_frozen_runtime_identity_is_accepted():
    row,manifest,result=data();assert module.valid_result(result,row,manifest)

@pytest.mark.parametrize('field',['key','suite','task_name','initial_state_id','driver_sha256','original_goals_unchanged'])
def test_wrong_episode_or_source_must_not_be_accepted(field):
    row,manifest,result=data();result[field]=False if field=='original_goals_unchanged' else 1 if field=='initial_state_id' else 'other'
    assert not module.valid_result(result,row,manifest)

def test_a_partially_matching_source_list_is_rejected():
    row,manifest,result=data();result['source_sha256'].pop('src/graspgen_libero_bridge/b.py')
    assert not module.valid_result(result,row,manifest)

def test_additional_initial_states_are_verified_against_the_requested_episode():
    row,manifest,result=data();result['initial_state_id']=1
    assert module.valid_result(result,row,manifest,initial_state_id=1)
    assert not module.valid_result(result,row,manifest,initial_state_id=0)
    assert not module.valid_result(result,row,manifest,initial_state_id=2)
