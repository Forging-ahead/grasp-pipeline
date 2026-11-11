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
good=lambda x:bool(x['result'].get('pipeline_success') and x['result'].get('official_goal_at_end') and x['result'].get('steps',0)>0)
assert s['full_success']==sum(good(x) for x in rows)
assert s['failed']==130-s['full_success']
assert s['success_rate']==s['full_success']/130
assert s['no_historical_outcomes'] and s['one_frozen_source'] and protocol['source_version']=='v10'
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
    manifest=json.loads(source.read('versions/v10/FROZEN_SOURCE.json'))
    assert len(manifest)==39
    for n,h in manifest.items():assert hashlib.sha256(source.read('versions/v10/'+n)).hexdigest()==h,n
    assert manifest['run_task.py']==protocol['driver_sha256']
    assert hashlib.sha256(source.read('ALL_TASKS.json')).hexdigest()==protocol['catalog_sha256']
    assert hashlib.sha256(source.read('TASK_TAXONOMY.json')).hexdigest()==protocol['taxonomy_sha256']
    for row in rows:
        d=row['result'];q=row['queue_record'];case=r/'cases'/row['key'];n=d['steps']
        assert row['full_success']==good(row)
        assert row['source_verified'] and row['initial_state_id']==d['initial_state_id']==q['initial_state_id']==0
        assert q['seed']==0 and q['evaluation_id']==protocol['evaluation_id']
        assert q['source_version']=='v10' and q['source_verified_before'] and q['source_verified_after']
        assert d['driver_sha256']==manifest['run_task.py']
        assert all(manifest[k]==v for k,v in d['source_sha256'].items())
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
page=(r/'LIBERO130任务最终v10复测报告_含全部视频.html').read_text()
embedded=collections.Counter(hashlib.sha256(base64.b64decode(x,validate=True)).hexdigest()
    for x in re.findall(r'src="data:video/mp4;base64,([^"]+)"',page))
assert embedded==videos and sum(videos.values())==130
assert len(re.findall(r'<details class="case"',page))==130
assert not re.search(r'(?:src|href)\s*=\s*[\"\x27]https?://',page,re.I)
assert json.loads((r/'RUNTIME_VERSIONS.json').read_text())['all_compared_versions_and_module_hashes_match']
audit=json.loads((r/'SOURCE_AUDIT.json').read_text())
assert audit['all_actor_hashes_match'] and not audit['external_changes'] and not audit['historical_outcomes_imported']
qa=dict(passed=True,records=130,categories=16,embedded_videos=130,embedded_videos_match_files=True,
    one_frozen_source=True,all_initial_states0=True,goals_unchanged=True,continuous_episodes=True,
    success=s['full_success'],failed=s['failed'],same_runtime_versions=True)
(r/'LOCAL_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print(json.dumps(qa,ensure_ascii=False))
