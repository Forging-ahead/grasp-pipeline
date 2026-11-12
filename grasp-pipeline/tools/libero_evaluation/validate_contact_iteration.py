#!/usr/bin/env python3
"""Independent stdlib check of the contact-iteration portable delivery."""
import argparse
import collections
import hashlib
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from validate_subset import array_shape, Media, read, same, sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('delivery',type=Path);args=p.parse_args()
    r=args.delivery.resolve();rows=read(r/'RESULTS.json');summary=read(r/'SUMMARY.json')
    manifest=read(r/'SOURCE_IDENTITY.json');protocols=read(r/'PROTOCOLS.json');configs=read(r/'FROZEN_CONFIGS.json')
    old={x['key']:x for x in read(r/'BASELINE_RESULTS.json')}
    assert len(rows)==summary['episodes']==sum(x['trials'] for x in protocols)
    assert len({(x['key'],x['initial_state_id']) for x in rows})==len(rows)
    assert {x['revision'] for x in protocols}=={summary['source_revision']}
    assert {x['source_digest'] for x in protocols}=={summary['source_digest']}=={manifest['source_digest']}
    import json
    assert hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==manifest['source_digest']
    with tarfile.open(r/'frozen-source.tar.gz') as archive:
        assert archive.pax_headers['comment']==summary['source_revision']
        for name,expected in manifest['files'].items():
            assert hashlib.sha256(archive.extractfile('grasp-pipeline/'+name).read()).hexdigest()==expected
        if (r/'MATRIX_PROTOCOL.json').exists():
            scheduling=read(r/'MATRIX_PROTOCOL.json');complete=read(r/'MATRIX_COMPLETE.json')
            assert scheduling['revision']==complete['revision']==summary['source_revision']
            assert scheduling['source_digest']==complete['source_digest']==manifest['source_digest']
            assert complete['episodes']==len(rows) and complete['source_unchanged']
            for name,key in [('run_matrix.py','runner_sha256'),('run_subset.py','single_episode_runner_sha256')]:
                assert hashlib.sha256(archive.extractfile('grasp-pipeline/tools/libero_evaluation/'+name).read()).hexdigest()==scheduling[key]
            cohorts=read(r/'COHORTS.json')
            assert {(g['init_state_id'],key) for g in cohorts for key in g['keys']}=={(x['initial_state_id'],x['key']) for x in rows}
    suites=list(ET.parse(r/'REGRESSION.xml').getroot().iter('testsuite'))
    assert sum(int(x.get('tests',0)) for x in suites)==summary['cpu_tests']
    assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
    actor={k:v for k,v in manifest['files'].items() if k.startswith('src/graspgen_libero_bridge/') and k.endswith('.py')}
    videos=[]
    for row in rows:
        key,init=row['key'],row['initial_state_id'];case=r/'cases'/('init_'+str(init))/key
        result,queue=read(case/'RESULT.json'),read(case/'QUEUE_RECORD.json')
        assert result==row['result'] and queue==row['queue']
        assert result['key']==key and result['initial_state_id']==queue['initial_state_id']==init
        assert result['source_sha256']==actor and result['driver_sha256']==manifest['files']['apps/libero/run_task.py']
        assert queue['source_digest']==manifest['source_digest'] and queue['source_unchanged'] and queue['result_identity_valid']
        assert result['original_goals_unchanged']
        assert result['original_goals']==old[key]['result']['original_goals']
        if init==0:
            assert result['original_goals']==old[key]['result']['original_goals']
            assert row['before_success']==old[key]['after_success']
        assert sha(case/'task.bddl')==row['original_bddl_sha256']
        config=read(case/'environment/config.json');prior=read(case/'INITIAL_CONFIG_BEFORE.json')
        ignore={'simulation_gpu','model_gpu','planner_gpu','init_state_id'}
        assert {k:v for k,v in config.items() if k not in ignore}=={k:v for k,v in prior.items() if k not in ignore}
        expected=dict(configs[str(init)][key])
        for name in ['simulation_gpu','model_gpu','planner_gpu']:expected[name]=queue['gpu']
        assert all(config.get(k)==v for k,v in expected.items())
        assert config['seed']==0 and config['init_state_id']==init and config['max_steps']==5000
        states=read(case/'states.json');n=result['steps'];assert len(states)==n+1
        if init==0:
            initial=read(case/'INITIAL_STATE_BEFORE.json')
            for name in ['q_arm','q_fingers','qvel_arm','qvel_fingers','world_T_eef','world_T_base','object_poses']:same(states[0][name],initial[name])
        assert all(abs(b['sim_time']-a['sim_time']-.05)<1e-9 for a,b in zip(states,states[1:]))
        with zipfile.ZipFile(case/'trajectory.npz') as trace:
            for name,shape in [('actions',(n,7)),('q_arm',(n+1,7)),('q_fingers',(n+1,2)),('world_T_eef',(n+1,4,4)),('phases',(n,))]:
                assert array_shape(trace.read(name+'.npy'))==shape
        passed=bool(not queue['watchdog_reached'] and result.get('pipeline_success') and result.get('official_goal_at_end') and n>0)
        assert passed==row['after_success']==queue['pipeline_success']
        assert row['decoded_frames']==n+41
        model=read(case/'environment/MODEL_SNAPSHOT.json')
        assert not model['live_physics_modified'] and model['phase']=='after_official_reset_and_settle'
        for name,info in model['files'].items():assert sha(case/'environment'/name)==info['sha256']
        for prefix in ['before','after']:
            if not row.get(prefix+'_video'):continue
            actual=sha(r/row[prefix+'_video']);assert actual==row[prefix+'_video_sha256'];videos.append(actual)
    paired=[x for x in rows if x['initial_state_id']==0];ps=summary['paired_init0']
    assert ps['tasks']==len(paired) and ps['before_success']==sum(x['before_success'] for x in paired)
    assert ps['after_success']==sum(x['after_success'] for x in paired)
    assert ps['improved']==sum(x['change']=='improved' for x in paired) and ps['regressed']==sum(x['change']=='regressed' for x in paired)
    assert summary['failure_counts']==dict(collections.Counter(x['failure']['code'] for x in rows if not x['after_success']))
    assert not summary['development_runs_counted'] and not summary['historical_successes_substituted']
    media=Media();media.feed(next(r.glob('*.html')).read_text());assert media.cases==len(rows)
    assert collections.Counter(media.videos)==collections.Counter(videos)
    files=read(r/'FILES_SHA256.json')
    assert set(files)=={str(x.relative_to(r)) for x in r.rglob('*') if x.is_file()}-{'FILES_SHA256.json','DELIVERY_QA.json'}
    for name,expected in files.items():assert sha(r/name)==expected
    qa=dict(passed=True,episodes=len(rows),paired_init0=ps,embedded_videos=len(videos),source_revision=summary['source_revision'],
        source_files_checked=len(manifest['files']),artifact_hashes_checked=len(files),same_frozen_source=True,continuous_trajectories=True)
    (r/'DELIVERY_QA.json').write_text(json.dumps(qa,indent=2)+'\n');print(json.dumps(qa))


if __name__=='__main__':main()
