#!/usr/bin/env python3
"""Balance several initial-state cohorts using the frozen single-case runner.

Each episode runs exactly once through run_subset.py. Cohort aggregates only
link to those original outputs; no outcomes, videos or versions are substituted.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from run_all import read, dump, identity, valid_result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--cohorts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--revision',required=True);p.add_argument('--gpu-slots',type=int,nargs='+',default=[0,1,2,3,0,1,2,3])
    p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    source,out=a.source.resolve(),a.output.resolve()
    if out.exists() or not 1<=len(a.gpu_slots)<=16 or min(a.gpu_slots)<0:p.error('Use a new output and 1..16 nonnegative GPU slots')
    sys.path.insert(0,str(source/'src'))
    from graspgen_libero_bridge.runtime.config import load_task,task_catalog
    catalog={row['key']:row for row in task_catalog()};groups=read(a.cohorts)
    assert groups and len({g['init_state_id'] for g in groups})==len(groups)
    for group in groups:
        assert type(group['init_state_id']) is int and group['init_state_id']>=0
        assert group['keys'] and len(set(group['keys']))==len(group['keys'])
        assert set(group['keys'])<=catalog.keys()
    manifest=identity(source)
    configurations={g['init_state_id']:{key:load_task(key,gpu=0,init=g['init_state_id'],paths_file=a.paths)[1]
        for key in g['keys']} for g in groups}
    assert all(cfg['seed']==0 and cfg['max_steps']==5000 for configs in configurations.values() for cfg in configs.values())
    ordered=list(dict.fromkeys(key for g in groups for key in g['keys']))
    episodes=[dict(key=key,initial_state_id=g['init_state_id']) for key in ordered for g in groups if key in g['keys']]
    if a.validate_only:
        print(json.dumps(dict(episodes=len(episodes),cohort_sizes={g['init_state_id']:len(g['keys']) for g in groups},source_digest=manifest['source_digest'],simulation_started=False)));return
    out.mkdir(parents=True);(out/'episodes').mkdir();(out/'logs').mkdir()
    dump(out/'COHORTS.json',groups);dump(out/'SOURCE_IDENTITY.json',manifest)
    dump(out/'MATRIX_PROTOCOL.json',dict(revision=a.revision,source=str(source),source_digest=manifest['source_digest'],
        episodes=len(episodes),gpu_slots=a.gpu_slots,outer_retries=False,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        single_episode_runner_sha256=hashlib.sha256((source/'tools/libero_evaluation/run_subset.py').read_bytes()).hexdigest(),
        execution='One frozen run_subset.py invocation per episode; original outputs retained'))
    for group in groups:
        init=group['init_state_id'];directory=out/('init_'+str(init));directory.mkdir()
        (directory/'runs/subset').mkdir(parents=True);(directory/'logs').mkdir()
        dump(directory/'SOURCE_IDENTITY.json',manifest)
        dump(directory/'ALL_TASKS.json',[catalog[key] for key in group['keys']])
        dump(directory/'FROZEN_CONFIGS.json',configurations[init])
        dump(directory/'PROTOCOL.json',dict(revision=a.revision,source=str(source),source_digest=manifest['source_digest'],
            trials=len(group['keys']),trials_per_task=1,initial_state_id=init,seed=0,max_steps=5000,
            gpu_slots=a.gpu_slots,outer_retries=False,historical_outcomes_used=False,fresh_graspgen_inference=True,
            scheduling_protocol=str(out/'MATRIX_PROTOCOL.json'),scope='Contact-policy cohort; not a full-130 score'))
    def verify():assert identity(source)==manifest,'Frozen actor changed'
    def run(episode,gpu):
        verify();key,init=episode['key'],episode['initial_state_id']
        directory=out/'episodes'/('init_'+str(init)+'__'+key)
        command=[sys.executable,str(source/'tools/libero_evaluation/run_subset.py'),'--source',str(source),
            '--paths',str(a.paths.resolve()),'--output',str(directory),'--revision',a.revision,'--init',str(init),
            '--gpu-slots',str(gpu),'--keys',key]
        dump(out/'logs'/('init_'+str(init)+'__'+key+'.command.json'),dict(command=command,gpu=gpu,initial_state_id=init))
        with (out/'logs'/('init_'+str(init)+'__'+key+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            dump(out/'logs'/('init_'+str(init)+'__'+key+'.runner.json'),dict(pid=child.pid,command=command))
            assert child.wait()==0,('Single-episode runner failed',key,init)
        complete=read(directory/'SUBSET_COMPLETE.json');assert complete['total']==1 and complete['source_unchanged']
        original=directory/'runs/subset'/key
        result,record=read(original/'RESULT.json'),read(original/'QUEUE_RECORD.json')
        assert valid_result(result,catalog[key],manifest,initial_state_id=init) and record['result_identity_valid']
        assert record['source_digest']==manifest['source_digest'] and record['initial_state_id']==init
        cohort=out/('init_'+str(init))
        (cohort/'runs/subset'/key).symlink_to(original,target_is_directory=True)
        (cohort/'logs'/(key+'.log')).symlink_to(directory/'logs'/(key+'.log'))
        verify();return record
    pending,active,finished=list(episodes),{},[];started=time.time()
    def progress():
        state=dict(total=len(episodes),finished=len(finished),pending=len(pending),running=list(active.values()),
            results=finished,elapsed_seconds=time.time()-started,complete=len(finished)==len(episodes))
        dump(out/'PROGRESS.json',state)
        for group in groups:
            init=group['init_state_id'];results=[x for x in finished if x['initial_state_id']==init]
            dump(out/('init_'+str(init))/'PROGRESS.json',dict(total=len(group['keys']),finished=len(results),
                pending=sum(x['initial_state_id']==init for x in pending),running=[x for x in active.values() if x['initial_state_id']==init],
                results=results,elapsed_seconds=time.time()-started,complete=len(results)==len(group['keys'])))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpu_slots)) as pool:
        while pending or active:
            slots=list(a.gpu_slots)
            for entry in active.values():slots.remove(entry['gpu'])
            for gpu in slots:
                if not pending:break
                episode=pending.pop(0)
                active[pool.submit(run,episode,gpu)]=dict(episode,gpu=gpu,started_unix=time.time())
            for future in list(active):
                if future.done():
                    result=future.result();del active[future];finished.append(result)
                    print(json.dumps(result,ensure_ascii=False),flush=True)
            progress()
            if pending or active:time.sleep(3)
    verify()
    for group in groups:
        results=[x for x in finished if x['initial_state_id']==group['init_state_id']]
        assert len(results)==len(group['keys'])
        dump(out/('init_'+str(group['init_state_id']))/'SUBSET_COMPLETE.json',dict(total=len(results),source_unchanged=True,
            pipeline_success=sum(x['pipeline_success'] for x in results),results=results))
    dump(out/'MATRIX_COMPLETE.json',dict(episodes=len(finished),source_unchanged=True,source_digest=manifest['source_digest'],
        revision=a.revision,wall_seconds=time.time()-started))


if __name__=='__main__':main()
