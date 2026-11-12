#!/usr/bin/env python3
"""Resume a quota-interrupted matrix without changing the frozen actor.

Retain every complete original episode, including policy failures. Re-run only
episodes with missing/incomplete artifacts or a recorded storage error. Original
attempts remain untouched and the exception to no-outer-retry is explicit.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_all import read, dump, identity, valid_result


def main():
    import numpy as np
    import imageio_ffmpeg
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-matrix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--paths',type=Path,required=True)
    a=p.parse_args();old=a.old_matrix.resolve();out=a.output.resolve()
    assert not out.exists();out.mkdir(parents=True)
    protocol=read(old/'MATRIX_PROTOCOL.json');source=Path(protocol['source'])
    manifest=read(old/'SOURCE_IDENTITY.json');groups=read(old/'COHORTS.json')
    assert identity(source)==manifest
    sys.path.insert(0,str(source/'src'))
    from graspgen_libero_bridge.runtime.config import task_catalog
    catalog={x['key']:x for x in task_catalog()}
    ordered=list(dict.fromkeys(k for g in groups for k in g['keys']))
    episodes=[dict(key=k,initial_state_id=g['init_state_id']) for k in ordered for g in groups if k in g['keys']]
    for name in ['SOURCE_IDENTITY.json','COHORTS.json','MATRIX_PROTOCOL.json']:
        shutil.copy2(old/name,out/name)
    (out/'episodes').mkdir();(out/'logs').mkdir()
    for g in groups:
        init=g['init_state_id'];d=out/('init_'+str(init));d.mkdir();(d/'runs/subset').mkdir(parents=True);(d/'logs').mkdir()
        for name in ['SOURCE_IDENTITY.json','ALL_TASKS.json','FROZEN_CONFIGS.json','PROTOCOL.json']:
            shutil.copy2(old/('init_'+str(init))/name,d/name)
    def link(episode,directory):
        key,init=episode['key'],episode['initial_state_id'];cohort=out/('init_'+str(init))
        (cohort/'runs/subset'/key).symlink_to(directory/'runs/subset'/key,target_is_directory=True)
        (cohort/'logs'/(key+'.log')).symlink_to(directory/'logs'/(key+'.log'))
    def complete_record(episode,directory):
        key,init=episode['key'],episode['initial_state_id'];case=directory/'runs/subset'/key
        result,record=read(case/'RESULT.json'),read(case/'QUEUE_RECORD.json')
        assert valid_result(result,catalog[key],manifest,initial_state_id=init) and record['result_identity_valid']
        assert record['source_digest']==manifest['source_digest'] and record['source_unchanged']
        assert not any(term in (result.get('reason') or '').lower() for term in ['disk quota','no space left'])
        states=read(case/'states.json');n=result['steps'];assert len(states)==n+1
        with np.load(case/'trajectory.npz',allow_pickle=False) as trace:
            assert trace['actions'].shape==(n,7) and np.isfinite(trace['actions']).all()
        np.testing.assert_allclose(np.diff([x['sim_time'] for x in states]),.05,atol=1e-9,rtol=0)
        assert (case/'FINAL_PHYSICS_STATE.npz').stat().st_size>0
        snapshot=read(case/'environment/MODEL_SNAPSHOT.json')
        for name,info in snapshot['files'].items():
            assert hashlib.sha256((case/'environment'/name).read_bytes()).hexdigest()==info['sha256']
        decoded=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-i',str(case/'rollout.mp4'),'-f','framehash','-'],capture_output=True,text=True,check=True)
        assert len([x for x in decoded.stdout.splitlines() if x and not x.startswith('#')])==n+1
        return record
    retained=[];pending=[];interruptions=[];finished=[]
    for episode in episodes:
        name='init_'+str(episode['initial_state_id'])+'__'+episode['key'];directory=old/'episodes'/name
        if not directory.exists():pending.append(episode);continue
        try:record=complete_record(episode,directory)
        except (OSError,AssertionError,ValueError,KeyError,subprocess.CalledProcessError) as error:
            interruptions.append(dict(**episode,original_directory=str(directory),reason=str(error) or type(error).__name__))
            pending.append(episode);continue
        link(episode,directory);finished.append(record)
        retained.append(dict(**episode,directory=str(directory),original_subset_finalized=(directory/'SUBSET_COMPLETE.json').exists()))
    recovery=dict(reason='Original scheduler and some episode writers aborted on shared-volume Errno 122 Disk quota exceeded',
        frozen_actor_unchanged=True,original_matrix=str(old),original_files_retained=True,
        actor_revision=protocol['revision'],source_digest=manifest['source_digest'],
        retained=retained,infrastructure_reruns=interruptions,previously_unstarted=len(pending)-len(interruptions),
        no_policy_failure_retries=True,outer_retry_exception='Only quota-interrupted or artifact-incomplete episodes; all complete policy failures retained',
        recovery_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),started_unix=time.time())
    dump(out/'INFRASTRUCTURE_RECOVERY.json',recovery);shutil.copy2(__file__,out/'recover_matrix.py')
    print('RECOVERY',json.dumps({k:len(recovery[k]) for k in ['retained','infrastructure_reruns']}),flush=True)
    def verify():assert identity(source)==manifest
    def run(episode,gpu):
        verify();key,init=episode['key'],episode['initial_state_id'];directory=out/'episodes'/('init_'+str(init)+'__'+key)
        command=[sys.executable,str(source/'tools/libero_evaluation/run_subset.py'),'--source',str(source),
            '--paths',str(a.paths.resolve()),'--output',str(directory),'--revision',protocol['revision'],'--init',str(init),'--gpu-slots',str(gpu),'--keys',key]
        with (out/'logs'/('init_'+str(init)+'__'+key+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            dump(out/'logs'/('init_'+str(init)+'__'+key+'.runner.json'),dict(pid=process.pid,command=command))
            assert process.wait()==0,('Recovery episode runner failed',key,init)
        record=complete_record(episode,directory);assert read(directory/'SUBSET_COMPLETE.json')['source_unchanged']
        link(episode,directory);verify();return record
    active={};slots=protocol['gpu_slots'];started=time.time()
    def progress():
        dump(out/'PROGRESS.json',dict(total=len(episodes),finished=len(finished),pending=len(pending),running=list(active.values()),
            results=finished,complete=len(finished)==len(episodes),elapsed_seconds=time.time()-started))
        for g in groups:
            init=g['init_state_id'];rows=[x for x in finished if x['initial_state_id']==init]
            dump(out/('init_'+str(init))/'PROGRESS.json',dict(total=len(g['keys']),finished=len(rows),results=rows,complete=len(rows)==len(g['keys'])))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        while pending or active:
            available=list(slots)
            for item in active.values():available.remove(item['gpu'])
            for gpu in available:
                if not pending:break
                episode=pending.pop(0);active[pool.submit(run,episode,gpu)]=dict(**episode,gpu=gpu,started_unix=time.time())
            for f in list(active):
                if f.done():record=f.result();del active[f];finished.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
            progress()
            if pending or active:time.sleep(3)
    verify()
    for g in groups:
        rows=[x for x in finished if x['initial_state_id']==g['init_state_id']]
        assert len(rows)==len(g['keys'])
        dump(out/('init_'+str(g['init_state_id']))/'SUBSET_COMPLETE.json',dict(total=len(rows),source_unchanged=True,pipeline_success=sum(x['pipeline_success'] for x in rows),results=rows))
    recovery['complete']=True;recovery['finished_unix']=time.time();dump(out/'INFRASTRUCTURE_RECOVERY.json',recovery)
    dump(out/'MATRIX_COMPLETE.json',dict(episodes=len(finished),source_unchanged=True,source_digest=manifest['source_digest'],revision=protocol['revision'],
        infrastructure_recovery='INFRASTRUCTURE_RECOVERY.json',wall_seconds=time.time()-started))


if __name__=='__main__':main()
