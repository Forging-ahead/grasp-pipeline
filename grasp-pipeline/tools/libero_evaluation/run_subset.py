#!/usr/bin/env python3
"""Run an explicit task subset once, preserving failures and frozen provenance."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_all import dump, identity, read, valid_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keys', nargs='+', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--init', type=int, default=0)
    parser.add_argument('--gpu-slots', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out.exists() or args.init < 0 or len(set(args.keys)) != len(args.keys) or not 1 <= len(args.gpu_slots) <= 16:
        parser.error('Use a new output, unique keys and at most 16 worker slots')
    sys.path.insert(0, str(source/'src'))
    from graspgen_libero_bridge.runtime.config import load_paths, load_task, task_catalog
    catalog = {row['key']: row for row in task_catalog()}
    rows = [catalog[key] for key in args.keys]
    paths = load_paths(args.paths)
    manifest = identity(source)
    configs = {row['key']: load_task(row['key'], gpu=0, init=args.init, paths_file=args.paths)[1] for row in rows}
    assert all(cfg['init_state_id']==args.init and cfg['seed']==0 and cfg['max_steps']==5000 for cfg in configs.values())
    out.mkdir(parents=True)
    for name in ['logs', 'runs', 'tmp', 'cache']:
        (out/name).mkdir()
    dump(out/'SOURCE_IDENTITY.json', manifest)
    dump(out/'FROZEN_CONFIGS.json', configs)
    dump(out/'ALL_TASKS.json', rows)
    dump(out/'PROTOCOL.json', dict(revision=args.revision, source=str(source),
        source_digest=manifest['source_digest'], trials=len(rows), trials_per_task=1,
        initial_state_id=args.init, seed=0, max_steps=5000, gpu_slots=args.gpu_slots,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        historical_outcomes_used=False, outer_retries=False, fresh_graspgen_inference=True,
        scope='Targeted iteration subset; not a new full-130 success rate'))

    def verify():
        assert identity(source)==manifest, 'Frozen actor changed during subset run'

    def run(row, gpu):
        verify()
        key = row['key']
        command = [paths['simulation_python'], str(source/'apps/libero/run_task.py'), key,
            '--gpu', str(gpu), '--init', str(args.init), '--tag', 'subset', '--paths', str(args.paths.resolve()),
            '--output-root', str(out/'runs')]
        environment = dict(os.environ, PYTHONPATH=str(source/'src'), PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
            TMPDIR=str(out/'tmp'), XDG_CACHE_HOME=str(out/'cache'),
            PYTHONPYCACHEPREFIX=str(out/'cache/pycache'), MPLCONFIGDIR=str(out/'cache/matplotlib'),
            CUDA_CACHE_PATH=str(out/'cache/cuda'))
        started = time.time()
        timed_out = False
        with (out/'logs'/(key+'.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=source, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            dump(out/'logs'/(key+'.pid.json'), dict(pid=process.pid, gpu=gpu, command=command,
                started_unix=started, wall_budget_s=row['wall_budget_s']))
            try:
                process.wait(timeout=row['wall_budget_s'])
            except subprocess.TimeoutExpired:
                timed_out = True
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        verify()
        case = out/'runs/subset'/key
        result = read(case/'RESULT.json') if (case/'RESULT.json').exists() else {}
        valid = valid_result(result, row, manifest,initial_state_id=args.init)
        record = dict(key=key, revision=args.revision, source_digest=manifest['source_digest'],
            result_identity_valid=valid, source_unchanged=True, gpu=gpu, initial_state_id=args.init,
            returncode=process.returncode, watchdog_reached=timed_out, wall_seconds=time.time()-started,
            pipeline_success=bool(valid and not timed_out and result.get('pipeline_success') and
                result.get('official_goal_at_end') and result.get('steps', 0)>0),
            official_goal_at_end=bool(result.get('official_goal_at_end')), steps=result.get('steps',0),
            status=result.get('status','missing_result'), failure=result.get('failure'), reason=result.get('reason'),
            raw_video_exists=(case/'rollout.mp4').is_file())
        case.mkdir(parents=True, exist_ok=True)
        dump(case/'QUEUE_RECORD.json', record)
        return record

    pending, active, finished = list(rows), {}, []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpu_slots)) as pool:
        while pending or active:
            available = list(args.gpu_slots)
            for entry in active.values():
                available.remove(entry['gpu'])
            for gpu in available:
                if not pending:
                    break
                row = pending.pop(0)
                active[pool.submit(run,row,gpu)] = dict(key=row['key'],gpu=gpu,started_unix=time.time())
            for future in list(active):
                if future.done():
                    record = future.result()
                    del active[future]
                    finished.append(record)
                    print(json.dumps(record,ensure_ascii=False),flush=True)
            dump(out/'PROGRESS.json', dict(total=len(rows),finished=len(finished),pending=len(pending),
                running=list(active.values()),results=finished,elapsed_seconds=time.time()-started,
                complete=len(finished)==len(rows)))
            if pending or active:
                time.sleep(3)
    verify()
    dump(out/'SUBSET_COMPLETE.json', dict(total=len(rows), source_unchanged=True,
        pipeline_success=sum(x['pipeline_success'] for x in finished), results=finished,
        wall_seconds=time.time()-started))


if __name__=='__main__':
    main()
