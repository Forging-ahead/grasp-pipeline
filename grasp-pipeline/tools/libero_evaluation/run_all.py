#!/usr/bin/env python3
"""Run all 130 organized LIBERO tasks once from a frozen checkout."""
import argparse,concurrent.futures,datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def dump(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def identity(source):
    files=[p for d in ['src','apps','configs'] for p in (source/d).rglob('*') if p.is_file() and p.suffix in ['.py','.json'] and '__pycache__' not in p.parts and not p.name.endswith('.local.json')]
    files.append(source/'run.py')
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    return dict(source_digest=digest(hashes),files=hashes)
def valid_result(result,row,manifest,initial_state_id=0):
    expected={n:h for n,h in manifest['files'].items() if n.startswith('src/graspgen_libero_bridge/') and n.endswith('.py')}
    return bool(result.get('key')==row['key'] and result.get('initial_state_id')==initial_state_id
        and result.get('suite')==row['suite'] and result.get('task_name')==row['task_name']
        and result.get('driver_sha256')==manifest['files']['apps/libero/run_task.py']
        and result.get('source_sha256')==expected and result.get('original_goals_unchanged'))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--taxonomy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    p.add_argument('--sessions-per-gpu',type=int,choices=[1,2,3,4],default=2)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args();source=args.source.resolve();out=args.output.resolve()
    if out.exists() or len(set(args.gpus))!=len(args.gpus) or min(args.gpus)<0:p.error('Use a new output and distinct nonnegative GPUs')
    sys.path.insert(0,str(source/'src'))
    from graspgen_libero_bridge.runtime.config import task_catalog,load_task,load_paths
    rows=task_catalog();taxonomy=read(args.taxonomy)
    assert len(rows)==len(taxonomy)==len({r['key'] for r in rows})==130
    assert {r['key'] for r in rows}=={r['key'] for r in taxonomy} and len({r['category'] for r in taxonomy})==16
    paths=load_paths(args.paths)
    for row in rows:
        t=next(t for t in taxonomy if t['key']==row['key'])
        assert row['suite']==t['suite'] and row['task_name']==t['task_name']
        assert row.get('bddl_sha256',t['bddl_sha256'])==t['bddl_sha256']
        assert hashlib.sha256((Path(paths['libero_repo'])/t['bddl_path']).read_bytes()).hexdigest()==t['bddl_sha256']
    paths=load_paths(args.paths);manifest=identity(source)
    if args.validate_only:
        configs=[load_task(r['key'],gpu=0,init=0,paths_file=args.paths)[1] for r in rows]
        assert all(c['seed']==0 and c['max_steps']==5000 and c['settle_steps']==10 for c in configs)
        print(json.dumps(dict(tasks=130,categories=16,bddl_hashes_verified=130,source_digest=manifest['source_digest'],simulation_started=False)));return
    out.mkdir(parents=True);(out/'logs').mkdir();(out/'tmp').mkdir();(out/'cache').mkdir();(out/'runs').mkdir()
    slots={str(g):args.sessions_per_gpu for g in args.gpus};dump(out/'GPU_SLOTS.json',slots)
    dump(out/'SCHEDULING_HISTORY.json',[dict(time_unix=time.time(),slots=slots,reason='Initial capacity; source and task budgets are fixed')])
    configs={r['key']:load_task(r['key'],gpu=0,init=0,paths_file=args.paths)[1] for r in rows}
    assert all(c['seed']==0 and c['max_steps']==5000 and c['input_mode']=='rgbd' and c['settle_steps']==10 for c in configs.values())
    protocol=dict(evaluation_id=out.parent.name,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),branch='libero',source_commit=args.commit,source=str(source),source_digest=manifest['source_digest'],trials=130,trials_per_task=1,init_state_id=0,seed=0,task_order_index=0,max_control_steps=5000,control_hz=20,wall_budget_counts={str(b):sum(r['wall_budget_s']==b for r in rows) for b in sorted({r['wall_budget_s'] for r in rows})},watchdog_grace_s=60,success_definition='pipeline_success AND official_goal_at_end AND steps > 0 with matching source, task and original goals',teacher_sources=['BDDL goals','simulator segmentation','simulator poses','simulator collision geometry'],no_historical_outcomes=True,no_external_retries=True,fresh_graspgen_inference=True,one_frozen_source=True,language_order_review='Both stack_transport tasks reviewed separately; final predicates do not certify instruction order',gpus=args.gpus,initial_sessions_per_gpu=args.sessions_per_gpu,concurrency_limit=16,paths=paths,driver_unchanged=True,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,value in [('PROTOCOL',protocol),('SOURCE_IDENTITY',manifest),('ALL_TASKS',rows),('TASK_TAXONOMY',taxonomy),('FROZEN_CONFIGS',configs)]:dump(out/(name+'.json'),value)
    def verify():assert identity(source)==manifest,'Frozen actor changed'
    def run(row,gpu):
        verify();key=row['key'];case=out/'runs/full130'/key
        command=[paths['simulation_python'],str(source/'apps/libero/run_task.py'),key,'--gpu',str(gpu),'--init','0','--tag','full130','--paths',str(args.paths.resolve()),'--output-root',str(out/'runs')]
        env=dict(os.environ,PYTHONPATH=str(source/'src'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(out/'tmp'),XDG_CACHE_HOME=str(out/'cache'),PYTHONPYCACHEPREFIX=str(out/'cache/pycache'),MPLCONFIGDIR=str(out/'cache/matplotlib'),CUDA_CACHE_PATH=str(out/'cache/cuda'))
        started=time.time();expired=False;terminated_group=False
        with (out/'logs'/(key+'.log')).open('x') as log:
            proc=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            dump(out/'logs'/(key+'.pid.json'),dict(pid=proc.pid,gpu=gpu,command=command,started_unix=started,wall_budget_s=row['wall_budget_s']))
            try:proc.wait(timeout=row['wall_budget_s'])
            except subprocess.TimeoutExpired:
                expired=True;os.kill(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            try:os.killpg(proc.pid,signal.SIGTERM);terminated_group=True
            except ProcessLookupError:pass
        verify();case.mkdir(parents=True,exist_ok=True)
        result=read(case/'RESULT.json') if (case/'RESULT.json').exists() else {}
        valid=valid_result(result,row,manifest)
        record=dict(key=key,evaluation_id=out.parent.name,source_commit=args.commit,source_digest=manifest['source_digest'],source_verified_before=True,source_verified_after=True,result_identity_valid=valid,gpu=gpu,initial_state_id=0,seed=0,returncode=proc.returncode,watchdog_reached=expired,wall_budget_s=row['wall_budget_s'],started_unix=started,finished_unix=time.time(),wall_seconds=time.time()-started,command=command,orphan_group_cleanup=terminated_group,status=result.get('status','driver_exit_without_result'),steps=result.get('steps',0),pipeline_success=bool(valid and not expired and result.get('pipeline_success') and result.get('official_goal_at_end') and result.get('steps',0)>0),official_goal_at_end=bool(valid and result.get('official_goal_at_end')),reason=result.get('reason'),raw_video_exists=(case/'rollout.mp4').is_file())
        dump(case/'QUEUE_RECORD.json',record);return record
    pending=list(rows);active={};done=[];started=time.time()
    def state():
        dump(out/'PROGRESS.json',dict(evaluation_id=out.parent.name,source_digest=manifest['source_digest'],total=130,finished=len(done),pending=len(pending),running=list(active.values()),results=done,worker_slots=read(out/'GPU_SLOTS.json'),started_unix=started,updated_unix=time.time(),elapsed_seconds=time.time()-started,complete=len(done)==130))
    state()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        while pending or active:
            slots={int(g):int(n) for g,n in read(out/'GPU_SLOTS.json').items()}
            assert set(slots)==set(args.gpus) and 0<sum(slots.values())<=16 and all(1<=n<=4 for n in slots.values())
            counts={g:sum(x['gpu']==g for x in active.values()) for g in slots}
            for g,capacity in slots.items():
                while pending and counts[g]<capacity:
                    row=pending.pop(0);f=pool.submit(run,row,g);active[f]=dict(key=row['key'],gpu=g,started_unix=time.time(),wall_budget_s=row['wall_budget_s']);counts[g]+=1
            for f in list(active):
                if f.done():
                    record=f.result();del active[f];done.append(record)
                    print(json.dumps(dict(finished=len(done),total=130,**record),ensure_ascii=False),flush=True)
            state()
            if active or pending:time.sleep(3)
    verify();dump(out/'SWEEP_COMPLETE.json',dict(completed=130,source_unchanged=True,source_digest=manifest['source_digest'],finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('QUEUE_COMPLETE',flush=True)
if __name__=='__main__':main()
