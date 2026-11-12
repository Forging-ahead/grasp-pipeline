#!/usr/bin/env python3
"""Freeze current code and replay selected baseline pools from identical inputs."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.adapters.robocasa.config import load_config
from graspgen_libero_bridge.runtime.robocasa_provenance import source_identity, digest_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True, help="TaskName:seed")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-profile", choices=["native", "libero-v10-teacher"], default="native")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to reuse an output directory")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("GPU indices must be distinct nonnegative integers")
    baseline = json.loads((args.baseline_root / "MATRIX.json").read_text())
    lookup = {(c["task_id"], c["seed"]): c for c in baseline["cells"]}
    cells = []
    for item in args.cases:
        task, seed = item.rsplit(":", 1)
        cell = lookup[(task, int(seed))]
        if cell in cells:
            parser.error("Duplicate case")
        cells.append(cell)
    identity = source_identity()
    policy = load_config(cells[0]["task_id"], budget_profile=args.budget_profile)["policy"]
    args.output.mkdir(parents=True)
    (args.output / "cases").mkdir(); (args.output / "launcher_logs").mkdir()
    matrix = {"experiment": "paired_v10_migration_diagnostics", "planned_episodes": len(cells), "cells": cells,
              "source_digest": identity["source_digest"], "policy_digest": digest_json(policy),
              "paired_diagnostic": True, "baseline_root": str(args.baseline_root.resolve()),
              "budget_profile": args.budget_profile,
              "baseline_source_digest": baseline["source_digest"], "gpus": args.gpus,
              "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (args.output / "MATRIX.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (args.output / "POLICY.json").write_text(json.dumps(policy, indent=2) + "\n")
    (args.output / "PROTOCOL.json").write_text((ROOT / "docs/robocasa/EXPERIMENT_PROTOCOL.json").read_text())
    (args.output / "SOURCE_IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
    with tarfile.open(args.output / "SOURCE.tar.gz", "w:gz") as archive:
        for name in identity["files"]:
            archive.add(ROOT / name, arcname=name, recursive=False)
    def shard(item):
        position, gpu = item
        for cell in cells[position::len(args.gpus)]:
            if source_identity()["source_digest"] != identity["source_digest"]:
                raise RuntimeError("Actor code changed during the paired experiment")
            name = cell["task_id"] + "__seed" + str(cell["seed"]).zfill(6)
            command = [sys.executable, str(ROOT / "robocasa.py"), "run", cell["task_id"],
                       "--seed", str(cell["seed"]), "--split", cell["split"], "--gpu", str(gpu),
                       "--budget-profile", args.budget_profile,
                       "--paths", str(args.paths.resolve()), "--replay-grasps-from", str((args.baseline_root / "cases" / name).resolve()),
                       "--output", str((args.output / "cases" / name).resolve())]
            with (args.output / "launcher_logs" / (name + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps({"case": name, "gpu": gpu, "return_code": result.returncode}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(shard, enumerate(args.gpus)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
