#!/usr/bin/env python3
"""Read-only progress for the organized 130-task run."""
import argparse,collections,datetime,json,time
from pathlib import Path

def snapshot(root):
    s=json.loads((root/'PROGRESS.json').read_text())
    results=s['results']
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=s['complete'] and (root/'SWEEP_COMPLETE.json').exists(),finished=s['finished'],total=s['total'],success=sum(bool(r['pipeline_success']) for r in results),official_final=sum(bool(r['official_goal_at_end']) for r in results),failed=sum(not r['pipeline_success'] for r in results),invalid_result_identities=sum(not r['result_identity_valid'] for r in results),watchdog_reached=sum(r['watchdog_reached'] for r in results),running=len(s['running']),pending=s['pending'],elapsed_minutes=round(s['elapsed_seconds']/60,1),worker_slots=s['worker_slots'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        value=snapshot(a.root);print(json.dumps(value,ensure_ascii=False),flush=True)
        if value['complete'] or not a.watch:break
        time.sleep(45)
