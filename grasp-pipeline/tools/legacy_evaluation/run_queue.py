"""Same-snapshot evaluation queue. Scheduling can add truly idle GPUs without changing the actor."""
import concurrent.futures, datetime, hashlib, json, os, signal, subprocess, time
from pathlib import Path

root=Path(__file__).resolve().parent
version=root/'versions/v10'
simulation_python=str(root/'runtime/openpi/bin/python')
tag='full130'
rows=json.loads((root/'ALL_TASKS.json').read_text())
manifest=json.loads((version/'FROZEN_SOURCE.json').read_text())
progress_path=root/'PROGRESS.json'
logs=root/'queue_logs';logs.mkdir(exist_ok=True)
(root/'runtime_tmp').mkdir(exist_ok=True)
(root/'runtime_cache').mkdir(exist_ok=True)
started=time.time();completed=[];pending=[];active={};adopted={}
drain_path=root/'CONCURRENCY_DRAIN.json'
draining={x['key']:x for x in json.loads(drain_path.read_text())['running_at_drain']} if drain_path.exists() else {}

def verify_source():
    for name,sha in manifest.items():
        assert hashlib.sha256((version/name).read_bytes()).hexdigest()==sha, name
    assert (root/'ALL_TASKS.json').read_bytes()==(version/'ALL_TASKS.json').read_bytes()

verify_source()
for row in rows:
    out=version/'runs'/tag/row['key']
    if (out/'QUEUE_RECORD.json').exists():
        record=json.loads((out/'QUEUE_RECORD.json').read_text())
        assert record['evaluation_id']==root.name and record['source_verified_before'] and record['source_verified_after']
        completed.append(record)
    elif out.exists():
        # The earlier dispatcher has received SIGINT and is waiting for its
        # owned workers to finish. Count those actors against the same GPU
        # capacity while filling only the newly available slots.
        assert row['key'] in draining, 'Unexpected unfinished task: '+str(out)
        pidinfo=json.loads((logs/(row['key']+'.pid.json')).read_text())
        info=dict(draining[row['key']],adopted=True,pid=pidinfo['pid'])
        cmd=Path('/proc')/str(pidinfo['pid'])/'cmdline'
        if cmd.exists():
            argv=cmd.read_bytes().split(b'\0')
            assert str(version/'run_task.py').encode() in argv and row['key'].encode() in argv
        adopted[row['key']]=info
    else:
        assert not out.exists(), 'Uncommitted task output requires explicit audit: '+str(out)
        pending.append(row)
if progress_path.exists():
    started=json.loads(progress_path.read_text())['started_unix']

def run(row,gpu):
    verify_source();key=row['key'];out=version/'runs'/tag/key
    wall_budget=int(row['wall_budget_s'])
    cmd=[simulation_python,str(version/'run_task.py'),key,'--tag',tag,'--gpu',str(gpu),'--init','0']
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
        TMPDIR=str(root/'runtime_tmp'),XDG_CACHE_HOME=str(root/'runtime_cache'),
        MPLCONFIGDIR=str(root/'runtime_cache/matplotlib'),CUDA_CACHE_PATH=str(root/'runtime_cache/cuda'))
    launch=time.time();expired=False
    with (logs/(key+'.log')).open('x') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        (logs/(key+'.pid.json')).write_text(json.dumps(dict(pid=proc.pid,gpu=gpu,started=launch,timeout_s=wall_budget,command=cmd)))
        try:
            proc.wait(timeout=wall_budget)
        except subprocess.TimeoutExpired:
            expired=True
            os.kill(proc.pid,signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    # No retry or outcome substitution, including infrastructure failures.
    record=dict(key=key,evaluation_id=root.name,source_version='v10',gpu=gpu,initial_state_id=0,
        seed=0,returncode=proc.returncode,watchdog_reached=expired,wall_budget_s=wall_budget,
        started_unix=launch,finished_unix=time.time(),wall_seconds=time.time()-launch,
        source_verified_before=True,source_verified_after=False,command=cmd)
    if (out/'RESULT.json').exists():
        d=json.loads((out/'RESULT.json').read_text())
        record.update({k:d.get(k) for k in ['status','pipeline_success','official_goal_at_end','reason','steps']})
        assert d.get('driver_sha256')==manifest['run_task.py'],key
        assert all(manifest.get(n)==h for n,h in d.get('source_sha256',{}).items()),key
    else:
        out.mkdir(parents=True,exist_ok=True)
        record.update(status='driver_exit_without_result',pipeline_success=False,official_goal_at_end=False,
            reason='Driver exited without a final result; preserved stdout/stderr is authoritative.')
        (out/'RESULT.json').write_text(json.dumps(record,indent=2))
    verify_source();record['source_verified_after']=True
    record['raw_video_exists']=(out/'rollout.mp4').exists()
    (out/'QUEUE_RECORD.json').write_text(json.dumps(record,indent=2))
    return record

def save_state():
    slots=json.loads((root/'GPU_SLOTS.json').read_text())
    state=dict(evaluation_id=root.name,source_version='v10',total=len(rows),finished=len(completed),
        pending=len(pending),started_unix=started,updated_unix=time.time(),elapsed_seconds=time.time()-started,
        worker_slots=slots,running=[v for v in active.values()]+list(adopted.values()),results=completed,
        complete=len(completed)==len(rows))
    temp=progress_path.with_suffix('.tmp');temp.write_text(json.dumps(state,ensure_ascii=False,indent=2));temp.replace(progress_path)

save_state()
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    while pending or active or adopted:
        slots={int(gpu):int(count) for gpu,count in json.loads((root/'GPU_SLOTS.json').read_text()).items()}
        assert 0<sum(slots.values())<=16 and all(0<count<=4 for count in slots.values())
        for key in list(adopted):
            path=version/'runs'/tag/key/'QUEUE_RECORD.json'
            if path.exists():
                record=json.loads(path.read_text())
                assert record['evaluation_id']==root.name and record['source_verified_before'] and record['source_verified_after']
                completed.append(record);del adopted[key]
                print(json.dumps(dict(finished=len(completed),total=len(rows),adopted_result=True,**record),ensure_ascii=False),flush=True)
        counts={gpu:sum(x['gpu']==gpu for x in list(active.values())+list(adopted.values())) for gpu in slots}
        for gpu,capacity in slots.items():
            while pending and counts[gpu]<capacity:
                row=pending.pop(0);f=pool.submit(run,row,gpu)
                active[f]=dict(key=row['key'],gpu=gpu,started_unix=time.time(),wall_budget_s=row['wall_budget_s'])
                counts[gpu]+=1
        for future in list(active):
            if not future.done():continue
            record=future.result()  # Fail loudly on a harness error; never silently drop a task.
            del active[future];completed.append(record)
            print(json.dumps(dict(finished=len(completed),total=len(rows),**record),ensure_ascii=False),flush=True)
        save_state()
        if pending or active:time.sleep(3)
verify_source();save_state()
print('QUEUE_COMPLETE',flush=True)
