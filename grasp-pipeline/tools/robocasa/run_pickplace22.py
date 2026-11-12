#!/usr/bin/env python3
"""Frozen r11 rerun of the exact 22 r10 task identities and initial states."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime
import hashlib
import json
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


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--sessions-per-gpu", type=int, choices=[1, 2], default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        parser.error("Use a new output directory and distinct nonnegative GPUs")
    baseline_root = args.baseline_root.resolve()
    old = read(baseline_root / "MATRIX.json")
    if old.get("source_digest") != "387084f9bee3cb24489e3cfa7c33815aab7a37dfb6408af09002bde3c0687a67":
        parser.error("Expected the frozen r10 all-65 baseline")
    base = load_config("PickPlaceCounterToStove", paths_file=args.paths, budget_profile="libero-v10-teacher")
    if base["policy"]["version"] != "robocasa-port-r11-v10":
        parser.error("This rerun requires the r11 migration fix")
    identity = source_identity()
    if identity["source_digest"] != args.expected_source_digest:
        parser.error("Source does not match the explicitly frozen r11 actor")
    cells, configs = [], {}
    for previous in old["cells"]:
        if previous["execution_mode"] != "run":
            continue
        name = previous["task_id"] + "__seed" + str(previous["seed"]).zfill(6)
        source = baseline_root / "cases" / name
        if not (source / "LAUNCHER.json").exists():
            parser.error("Baseline is incomplete: " + name)
        result = read(source / "RESULT.json")
        config = read(source / "CONFIG.json")
        if result.get("source_digest") != old["source_digest"] or result.get("paired_diagnostic"):
            parser.error("Baseline identity mismatch: " + name)
        if config["policy"]["budget_profile"] != "libero-v10-teacher":
            parser.error("Baseline was not a collection-budget run")
        # Carry task/seed/split literally from the old case; only policy,
        # deployment, GPU scheduling and the initial-state assertion change.
        config = copy.deepcopy(config)
        config.update(policy=copy.deepcopy(base["policy"]), paths=copy.deepcopy(base["paths"]), baseline_case=str(source))
        config.pop("candidate_replay_case", None)
        configs[name] = config
        cells.append({"case": name, "task_id": previous["task_id"], "seed": previous["seed"],
            "split": previous["split"], "category": previous["category"], "execution_mode": "run",
            "baseline_case": str(source), "baseline_success": bool(result.get("pipeline_success")),
            "task_binding_digest": digest_json(config["task"])})
    if len(cells) != 22 or len({c["task_id"] for c in cells}) != 22:
        parser.error("The denominator must be the original 22 actual policy attempts")
    cells.sort(key=lambda c: c["task_id"])
    args.output.mkdir(parents=True)
    for name in ("cases", "launcher_logs"):
        (args.output / name).mkdir()
    tool_names = ("run_pickplace22.py", "atomic65_case.py", "atomic65_driver.py")
    hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in tool_names}
    manifest = {"schema_version": "robocasa-pickplace22-rerun-1", "planned_episodes": 22,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_digest": identity["source_digest"], "policy_digest": digest_json(base["policy"]),
        "baseline_root": str(baseline_root), "baseline_source_digest": old["source_digest"],
        "cells": cells, "gpus": args.gpus, "sessions_per_gpu": args.sessions_per_gpu,
        "budget_profile": "libero-v10-teacher", "episode_wall_cap_s": 1200,
        "work_deadline_s": 1190, "recording_cleanup_reserved_s": 10,
        "baseline_wall_cap_s": old["episode_wall_cap_s"], "tools_sha256": hashes,
        "fresh_graspgen_inference": True, "no_external_retries": True,
        "no_historical_success_replacement": True, "initial_state_check_before_action": True,
        "comparison_limit": "same verified initial physics; fresh stochastic GraspGen pools; wall budget also changed; not a single-component ablation"}
    for name, value in (("MATRIX", manifest), ("SOURCE_IDENTITY", identity), ("POLICY", base["policy"]), ("FROZEN_CONFIGS", configs)):
        (args.output / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    with tarfile.open(args.output / "SOURCE.tar.gz", "w:gz") as archive:
        for name in identity["files"]:
            archive.add(ROOT / name, arcname="grasp-pipeline/" + name, recursive=False)
        for name in tool_names:
            archive.add(Path(__file__).parent / name, arcname="tools/" + name, recursive=False)
    if not args.execute:
        print(json.dumps({"planned": 22, "simulation_started": False, "source_digest": identity["source_digest"]})); return 0
    pending = queue.Queue()
    for cell in cells:
        pending.put(cell)
    def worker(slot):
        gpu = args.gpus[slot % len(args.gpus)]
        while True:
            try:
                cell = pending.get_nowait()
            except queue.Empty:
                return
            if source_identity()["source_digest"] != identity["source_digest"]:
                raise RuntimeError("Actor changed; remaining cases were not launched")
            name = cell["case"]
            directory = args.output / "cases" / name; directory.mkdir()
            config = copy.deepcopy(configs[name]); config["gpu"] = gpu
            (directory / "CONFIG.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
            command = [sys.executable, str(Path(__file__).parent / "atomic65_case.py"), "--config", str((directory / "CONFIG.json").resolve()),
                       "--mode", "run", "--output", str(directory.resolve())]
            started = time.monotonic()
            with (args.output / "launcher_logs" / (name + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            (directory / "LAUNCHER.json").write_text(json.dumps({"return_code": result.returncode, "gpu": gpu,
                "session_slot": slot, "wall_s": time.monotonic() - started}, indent=2) + "\n")
            print(json.dumps({"case": name, "gpu": gpu, "return_code": result.returncode}), flush=True)
            pending.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus) * args.sessions_per_gpu) as pool:
        list(pool.map(worker, range(len(args.gpus) * args.sessions_per_gpu)))
    (args.output / "SWEEP_COMPLETE.json").write_text(json.dumps({"planned": 22,
        "completed_launches": len(list((args.output / "cases").glob("*/LAUNCHER.json"))),
        "source_unchanged": source_identity()["source_digest"] == identity["source_digest"]}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
