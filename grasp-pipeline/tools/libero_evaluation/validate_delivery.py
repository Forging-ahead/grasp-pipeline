"""Validate the delivered one-version evaluation using the Python standard library."""
import argparse, ast, base64, collections, hashlib, json, re, zipfile
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('delivery',type=Path);a=p.parse_args();r=a.delivery
rows=json.loads((r/'RESULTS.json').read_text());s=json.loads((r/'SUMMARY.json').read_text())
groups=json.loads((r/'CATEGORY_SUMMARY.json').read_text());catalog=json.loads((r/'ALL_TASKS.json').read_text())
taxonomy=json.loads((r/'TASK_TAXONOMY.json').read_text());protocol=json.loads((r/'PROTOCOL.json').read_text())
assert len(rows)==len(catalog)==len(taxonomy)==130
assert len({x['key'] for x in rows})==130
assert {x['key'] for x in rows}=={x['key'] for x in catalog}=={x['key'] for x in taxonomy}
tax={x['key']:x for x in taxonomy}
frozen=json.loads((r/'FROZEN_CONFIGS.json').read_text())
good=lambda x:bool(x['result'].get('pipeline_success') and x['result'].get('official_goal_at_end') and x['result'].get('steps',0)>0 and not x['queue_record']['watchdog_reached'] and x['queue_record']['result_identity_valid'])
assert s['full_success']==sum(good(x) for x in rows)
assert s['failed']==130-s['full_success']
assert s['success_rate']==s['full_success']/130
assert s['no_historical_outcomes'] and s['one_frozen_source'] and protocol['branch']=='libero' and protocol['source_commit']=='6d8cd7d7e6a34878bb71a0f5e61007e3f81ac405'
assert len(groups)==16 and sum(x['tasks'] for x in groups)==130
assert len({x['category'] for x in groups})==16
for g in groups:
    selected=[x for x in rows if x['category']==g['category']]
    assert g['tasks']==len(selected) and g['full_success']==sum(good(x) for x in selected),g
    assert g['failure_reasons']==dict(collections.Counter(x['failure']['label'] for x in selected if not good(x)))
    assert g['success_rate']==g['full_success']/g['tasks']

def shape(data):
    assert data[:6]==b'\x93NUMPY'
    width,offset=(2,10) if data[6]==1 else (4,12)
    n=int.from_bytes(data[8:8+width],'little')
    return ast.literal_eval(data[offset:offset+n].decode())['shape']

videos=collections.Counter()
with zipfile.ZipFile(r/'research-source.zip') as source:
    identity=json.loads(source.read('SOURCE_IDENTITY.json'));manifest=identity['files']
    assert identity['source_digest']==protocol['source_digest']
    assert hashlib.sha256(source.read('harness/run_all.py')).hexdigest()==protocol['runner_sha256']
    assert hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()==identity['source_digest']
    for n,h in manifest.items():assert hashlib.sha256(source.read('source/grasp-pipeline/'+n)).hexdigest()==h,n
    assert json.loads(source.read('ALL_TASKS.json'))==catalog
    assert json.loads(source.read('TASK_TAXONOMY.json'))==taxonomy
    for row in rows:
        d=row['result'];q=row['queue_record'];case=r/'cases'/row['key'];n=d['steps']
        assert row['full_success']==good(row)
        assert row['source_verified'] and row['initial_state_id']==d['initial_state_id']==q['initial_state_id']==0
        assert q['seed']==0 and q['evaluation_id']==protocol['evaluation_id']
        actual=json.loads((case/'environment/config.json').read_text());expected=dict(frozen[row['key']])
        for name in ['simulation_gpu','model_gpu','planner_gpu']:expected[name]=q['gpu']
        assert all(actual.get(name)==value for name,value in expected.items()),row['key']
        assert q['source_commit']==protocol['source_commit'] and q['source_digest']==protocol['source_digest'] and q['source_verified_before'] and q['source_verified_after']
        assert d['driver_sha256']==manifest['apps/libero/run_task.py']
        assert d['source_sha256']=={n:h for n,h in manifest.items() if n.startswith('src/graspgen_libero_bridge/') and n.endswith('.py')}
        assert d['original_goals_unchanged'] and d['original_goals']==tax[row['key']]['goals']
        assert row['category']==tax[row['key']]['category']
        assert hashlib.sha256((case/'task.bddl').read_bytes()).hexdigest()==tax[row['key']]['bddl_sha256']
        sha=hashlib.sha256((r/row['video_path']).read_bytes()).hexdigest();assert sha==row['video_sha256']
        videos[sha]+=1
        assert row['frames']==n+41 and row['video_fully_decoded']
        with zipfile.ZipFile(case/'trajectory.npz') as z:
            assert shape(z.read('actions.npy'))==(n,7)
            assert shape(z.read('world_T_eef.npy'))==(n+1,4,4)
            assert shape(z.read('phases.npy'))==(n,)
        states=json.loads((case/'states.json').read_text())
        assert len(states)==n+1 and [x['step_count'] for x in states]==list(range(n+1))
        assert all(abs(x['sim_time']-states[0]['sim_time']-i/20)<1e-7 for i,x in enumerate(states))
        assert all(x['label_source']=='handwritten code policy; not a VLM decision' for x in json.loads((case/'KEYPOINTS.json').read_text()))
page=(r/'LIBERO130_libero分支复测_20261113_含全部视频.html').read_text()
embedded=collections.Counter(hashlib.sha256(base64.b64decode(x,validate=True)).hexdigest()
    for x in re.findall(r'src="data:video/mp4;base64,([^"]+)"',page))
assert embedded==videos and sum(videos.values())==130
assert len(re.findall(r'<details class="case"',page))==130
assert not re.search(r'(?:src|href)\s*=\s*[\"\x27]https?://',page,re.I)
assert json.loads((r/'DEPLOYMENT.json').read_text())['passed']
assert json.loads((r/'DEPLOYMENT.json').read_text())['weight_hashes_checked']
audit=json.loads((r/'SOURCE_AUDIT.json').read_text())
assert audit['all_actor_hashes_match'] and not audit['external_changes'] and not audit['historical_outcomes_imported']
qa=dict(passed=True,records=130,categories=16,embedded_videos=130,embedded_videos_match_files=True,
    one_frozen_source=True,all_initial_states0=True,goals_unchanged=True,continuous_episodes=True,
    success=s['full_success'],failed=s['failed'],same_runtime_versions=True)
(r/'LOCAL_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print(json.dumps(qa,ensure_ascii=False))
