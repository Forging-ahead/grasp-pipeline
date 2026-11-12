#!/usr/bin/env python3
"""Read-only current-turn monitor; no scheduling or policy changes."""
import argparse
import datetime
import json
from pathlib import Path
import time

from report_atomic65 import summarize

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
seen = set()
while True:
    summary = summarize(args.root)
    total = summary["groups"]["all"]
    new = []
    for row in summary["rows"]:
        if row["completed"] and row["task_id"] not in seen:
            seen.add(row["task_id"])
            new.append({"task": row["task_id"], "status": row["status"], "steps": row["physical_steps"],
                        "failure": row["failure_code"]})
    print(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "completed": total["completed"],
        "success": total["success"], "native_budget_success": total["native_budget_success"],
        "unsupported": total["unsupported"], "technical": total["technical_failure"], "new": new}, ensure_ascii=False), flush=True)
    if summary["complete"]:
        break
    time.sleep(45)
