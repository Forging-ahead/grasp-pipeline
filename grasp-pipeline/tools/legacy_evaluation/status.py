"""Compact live progress; avoid copying videos or historical results."""
import argparse,collections,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--watch',type=int,default=0);p.add_argument('--details',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parent;end=time.monotonic()+a.watch;previous=None
while True:
    f=r/'PROGRESS.json'
    if f.exists():
        try:d=json.loads(f.read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            # The distributed mount can briefly expose a replacement gap.
            if time.monotonic()>=end:break
            time.sleep(.5);continue
        completed=d['results']
        good=lambda x:bool(x.get('pipeline_success') and x.get('official_goal_at_end') and x.get('steps',0)>0)
        snap=dict(finished=d['finished'],total=d['total'],success=sum(good(x) for x in completed),
            failed=sum(not good(x) for x in completed),elapsed_minutes=round((time.time()-d['started_unix'])/60,1),
            missing_video=[x['key'] for x in completed if not x.get('raw_video_exists')],
            wall_budget_exhausted=sum(x.get('watchdog_reached',False) for x in completed),
            running=[])
        for task in d['running']:
            log=r/'queue_logs'/(task['key']+'.log');lines=log.read_text(errors='replace').splitlines() if log.exists() else []
            phase=next((line for line in reversed(lines) if line.startswith('{"stage":')),lines[-1] if lines else '')
            snap['running'].append(dict(key=task['key'],gpu=task['gpu'],minutes=round((time.time()-task['started_unix'])/60,1),phase=phase[:500]))
        stamp=(d['finished'],tuple(x['phase'] for x in snap['running'])) if a.details else (d['finished'],)
        if not a.details:
            snap['running_gpu_counts']=dict(collections.Counter(x['gpu'] for x in snap['running']))
            snap['oldest_running_minutes']=max((x['minutes'] for x in snap['running']),default=0)
            snap['running']=len(snap['running'])
        if stamp!=previous or time.monotonic()>=end:
            print(json.dumps(snap,ensure_ascii=False),flush=True);previous=stamp
        if d['complete']:break
    else:print('Waiting for queue progress',flush=True)
    if time.monotonic()>=end:break
    time.sleep(min(5,max(0,end-time.monotonic())))
