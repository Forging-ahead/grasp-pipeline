#!/usr/bin/env python3
"""Freeze a complete task matrix, then optionally execute it without best-of retries."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.adapters.robocasa.config import catalog, load_config
from graspgen_libero_bridge.runtime.robocasa_provenance import source_identity, digest_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["E1_minimal_pick_place_port", "E1_pick_place_development", "E1_development_snapshot", "E2_main"], required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/robocasa/EXPERIMENT_PROTOCOL.json")
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--readiness", type=Path, help="Required for E2 execution, generated from real development evidence")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error("Use distinct nonnegative GPU IDs; one session per GPU in the first implementation")
    if args.output.exists():
        parser.error("Refusing to reuse a matrix output directory")
    design = json.loads(args.protocol.read_text())
    exp = design["experiments"][args.experiment]
    selected = [t for t in catalog() if (t["task_id"] in exp["tasks"] if "tasks" in exp else
                                        args.experiment == "E2_main" or t["role"] == "development")]
    cells = [{"task_id": task["task_id"], "category": task["category"], "role": task["role"],
              "seed": seed, "split": exp["split"]}
             for task in selected for seed in range(exp["seed_start_inclusive"], exp["seed_stop_exclusive"])]
    expected = exp.get("planned_episodes", exp.get("planned_episodes_per_snapshot", exp.get("planned_episodes_per_version")))
    if len(cells) != expected:
        parser.error("Task matrix does not match the predeclared experiment count")
    identity = source_identity()
    config = load_config(cells[0]["task_id"], seed=cells[0]["seed"], split=cells[0]["split"],
                         paths_file=args.paths, policy_file=args.policy)
    policy_digest = digest_json(config["policy"])
    if args.execute:
        if exp.get("execution_available") is False:
            parser.error("The unchanged-policy baseline was not preserved before debugging; use E1_pick_place_development")
        if args.paths is None:
            parser.error("Execution requires --paths")
        if args.experiment == "E2_main":
            if args.readiness is None:
                parser.error("E2 execution requires --readiness from real native development/control evidence")
            readiness = json.loads(args.readiness.read_text())
            if readiness.get("passed") is not True or readiness.get("source_digest") != identity["source_digest"] or readiness.get("policy_digest") != policy_digest:
                parser.error("Readiness evidence is missing, failed, or belongs to another source/policy version")
    args.output.mkdir(parents=True)
    manifest = {"schema_version": "robocasa-matrix-1", "experiment": args.experiment,
                "source_digest": identity["source_digest"], "policy_digest": policy_digest,
                "protocol_digest": digest_json(design), "source_files": identity["files"],
                "planned_episodes": len(cells), "cells": cells, "gpus": args.gpus,
                "sessions_per_gpu": 1, "execute_requested": args.execute}
    (args.output / "MATRIX.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "PROTOCOL.json").write_text(json.dumps(design, indent=2) + "\n")
    (args.output / "POLICY.json").write_text(json.dumps(config["policy"], indent=2) + "\n")
    with tarfile.open(args.output / "SOURCE.tar.gz", "w:gz") as archive:
        for name in identity["files"]:
            archive.add(ROOT / name, arcname="grasp-pipeline/" + name)
    if not args.execute:
        print(json.dumps({"planned_episodes": len(cells), "simulation_started": False, "manifest": str(args.output / 'MATRIX.json')}))
        return 0

    def run_shard(shard, gpu):
        for cell in shard:
            if source_identity()["source_digest"] != identity["source_digest"]:
                raise RuntimeError("Source changed; stopped scheduling remaining cases")
            name = cell["task_id"] + f"__seed{cell['seed']:06d}"
            directory = args.output / "cases" / name
            command = [sys.executable, str(ROOT / "robocasa.py"), "run", cell["task_id"],
                       "--seed", str(cell["seed"]), "--split", cell["split"], "--gpu", str(gpu),
                       "--paths", str(args.paths.resolve()), "--output", str(directory.resolve())]
            if args.policy is not None:
                command.extend(["--policy", str(args.policy.resolve())])
            seconds = config["policy"]["episode_wall_seconds_composite" if cell["category"] == "composite" else "episode_wall_seconds_atomic"]
            started = time.monotonic()
            supervisor = {"cell": cell, "gpu": gpu, "timed_out": False}
            log_dir = args.output / "launcher_logs"; log_dir.mkdir(exist_ok=True)
            with (log_dir / (name + ".log")).open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    supervisor["return_code"] = process.wait(timeout=seconds + 10)
                except subprocess.TimeoutExpired:
                    supervisor["timed_out"] = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    supervisor["return_code"] = process.returncode
            supervisor["wall_s"] = time.monotonic() - started
            directory.mkdir(parents=True, exist_ok=True)
            child_path = directory / "SUPERVISOR.json"
            if child_path.is_file():
                supervisor["child_supervisor"] = json.loads(child_path.read_text())
                supervisor["timed_out"] |= supervisor["child_supervisor"].get("timed_out", False)
            (directory / "SUPERVISOR.json").write_text(json.dumps(supervisor, indent=2) + "\n")
            print(json.dumps({"case": name, "return_code": supervisor["return_code"], "timed_out": supervisor["timed_out"]}), flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(run_shard, cells[index::len(args.gpus)], gpu) for index, gpu in enumerate(args.gpus)]
        for future in futures:
            future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
