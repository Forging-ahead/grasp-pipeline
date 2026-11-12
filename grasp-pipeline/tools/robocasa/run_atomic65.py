#!/usr/bin/env python3
"""One frozen all-atomic coverage sweep; unsupported scenes are explicitly probes."""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.adapters.robocasa.config import load_config
from graspgen_libero_bridge.runtime.robocasa_provenance import source_identity, digest_json
from atomic65 import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--sessions-per-gpu", type=int, choices=[1, 2], default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("Use a new output directory and distinct nonnegative GPUs")
    base = load_config("PickPlaceCounterToStove", seed=args.seed, paths_file=args.paths, budget_profile="libero-v10-teacher")
    binding = inventory(base["paths"]["robocasa_repo"])
    identity = source_identity()
    expected = "387084f9bee3cb24489e3cfa7c33815aab7a37dfb6408af09002bde3c0687a67"
    if identity["source_digest"] != expected:
        parser.error("Actor differs from the tested r10 version")
    cells = []
    for task in binding["tasks"]:
        cells.append({k: task[k] for k in ("task_id", "category", "role", "execution_mode", "binding_status", "unsupported_reason")})
        cells[-1].update(seed=args.seed, split="pretrain")
    args.output.mkdir(parents=True)
    (args.output / "cases").mkdir(); (args.output / "launcher_logs").mkdir()
    tools = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*atomic65*.py")}
    manifest = {"schema_version": "robocasa-atomic65-matrix-1", "experiment": "r10_all65_atomic_single_initial_state",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source_digest": identity["source_digest"],
        "policy_digest": digest_json(base["policy"]), "task_bindings_digest": digest_json(binding),
        "planned_episodes": 65, "planned_policy_attempts": 22, "planned_unsupported_scene_probes": 43,
        "cells": cells, "gpus": args.gpus, "sessions_per_gpu": args.sessions_per_gpu,
        "budget_profile": "libero-v10-teacher", "native_horizons_kept_separately": True,
        "fresh_graspgen_inference": True, "tools_sha256": tools, "execute_requested": args.execute,
        "episode_wall_cap_s": 900, "work_deadline_s": 890, "recording_cleanup_reserved_s": 10,
        "no_external_retries": True, "no_historical_result_reuse": True}
    for name, content in (("MATRIX", manifest), ("TASK_BINDINGS", binding), ("POLICY", base["policy"]), ("SOURCE_IDENTITY", identity)):
        (args.output / (name + ".json")).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    with tarfile.open(args.output / "SOURCE.tar.gz", "w:gz") as archive:
        for name in identity["files"]:
            archive.add(ROOT / name, arcname="grasp-pipeline/" + name, recursive=False)
        for name in tools:
            archive.add(Path(__file__).parent / name, arcname="tools/" + name, recursive=False)
    if not args.execute:
        print(json.dumps({"planned": 65, "policy_attempts": 22, "unsupported_probes": 43, "simulation_started": False})); return 0
    tasks = {task["task_id"]: task for task in binding["tasks"]}
    pending = queue.Queue()
    # Run actual policy attempts first, keeping queued probes visible in the denominator.
    for cell in sorted(cells, key=lambda c: (c["execution_mode"] != "run", c["task_id"])):
        pending.put(cell)
    def worker(slot):
        gpu = args.gpus[slot % len(args.gpus)]
        while True:
            try:
                cell = pending.get_nowait()
            except queue.Empty:
                return
            if source_identity()["source_digest"] != identity["source_digest"]:
                raise RuntimeError("Actor source changed; remaining tasks not scheduled")
            name = cell["task_id"] + "__seed" + str(cell["seed"]).zfill(6)
            directory = args.output / "cases" / name; directory.mkdir()
            config = copy.deepcopy(base); config["gpu"] = gpu; config["task"] = tasks[cell["task_id"]]
            (directory / "CONFIG.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
            command = [sys.executable, str(Path(__file__).parent / "atomic65_case.py"), "--config", str((directory / "CONFIG.json").resolve()),
                       "--mode", cell["execution_mode"], "--output", str(directory.resolve())]
            started = time.monotonic()
            with (args.output / "launcher_logs" / (name + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            (directory / "LAUNCHER.json").write_text(json.dumps({"return_code": result.returncode, "gpu": gpu,
                "session_slot": slot, "wall_s": time.monotonic() - started, "execution_mode": cell["execution_mode"]}, indent=2) + "\n")
            print(json.dumps({"case": name, "gpu": gpu, "mode": cell["execution_mode"], "return_code": result.returncode}), flush=True)
            pending.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus) * args.sessions_per_gpu) as pool:
        list(pool.map(worker, range(len(args.gpus) * args.sessions_per_gpu)))
    (args.output / "SWEEP_COMPLETE.json").write_text(json.dumps({"planned": 65, "completed_launches": len(list((args.output / "cases").glob("*/LAUNCHER.json"))),
        "source_unchanged": source_identity()["source_digest"] == identity["source_digest"]}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
