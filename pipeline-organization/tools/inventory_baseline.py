#!/usr/bin/env python3
"""Inventory the delivered v10 baseline without importing or running the policy.

Default: copy the frozen actor snapshot once and write inventories in ../audit.
--check: validate existing copies and original artifacts; write nothing.
Only Python's standard library is required. No network or GPU is used.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "pipeline-organization"
DELIVERY = ROOT / "outputs/libero-final130-four4090-20261111/delivery"
DEV = ROOT / "work/libero-policy-broad-20261111"
SNAPSHOT = OUT / "baseline/v10"
FINAL_ARCHIVE_SHA256 = "28aeb7efc75805390e54cbfc109276676994141254bdbfed361e32622387e6c8"

GROUPS = {
    "entry": ["run_task", "__init__", "__main__"],
    "policy": ["evaluate", "task_policy", "skills", "articulation", "placement", "transfer", "surface_packing"],
    "perception_model": ["pointcloud", "graspgen_worker", "grasp_candidate", "grasp_contact"],
    "planning_execution": ["planner_worker", "planning_geometry", "cartesian", "controller"],
    "environment_geometry": ["env_adapter", "scene", "regions", "placement_geometry"],
    "shared_utilities": ["frames", "rpc", "errors"],
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(path.read_text())


def write_json(name, value):
    folder = OUT / "audit"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def literal_or_expression(node):
    try:
        return {"literal": ast.literal_eval(node)}
    except (ValueError, TypeError):
        return {"expression": ast.unparse(node)}


def inventory_python(name, data):
    source = data.decode("utf-8")
    tree = ast.parse(source, filename=name)
    group = "tests" if name.startswith("bridge/tests/") else next(
        group for group, stems in GROUPS.items() if Path(name).stem in stems)
    imports, settings = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend({"module": item.name, "line": node.lineno, "relative": False} for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append({"module": node.module, "line": node.lineno, "relative": node.level > 0})
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            owner = ast.unparse(node.func.value)
            if (owner == "config" or owner == "cfg" or owner.endswith(".config")) and node.args:
                key = literal_or_expression(node.args[0])
                if "literal" in key and isinstance(key["literal"], str):
                    settings.append({"key": key["literal"], "line": node.lineno,
                                     "default": literal_or_expression(node.args[1]) if len(node.args) > 1 else {"literal": None}})
    patterns = {
        "simulator_api": r"\b(mujoco|robosuite)\b|\.sim\.(model|data)|\bsim\.(model|data)",
        "libero_task_api": r"\blibero\b|parsed_problem|obj_body_id|object_sites_dict|fixtures_dict|_eval_predicate",
        "robot_assumption": r"Panda|franka_panda|robot0_|gripper0_|ARM_NAMES|FINGER_NAMES|OSC_POSE|_ref_joint",
    }
    evidence = {kind: [{"line": i, "text": line.strip()[:240]} for i, line in enumerate(source.splitlines(), 1)
                       if re.search(pattern, line)] for kind, pattern in patterns.items()}
    return {"path": name, "sha256": sha(data), "bytes": len(data), "lines": len(source.splitlines()),
            "group": group, "imports": imports, "config_reads": settings, "coupling_evidence": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only verification of the snapshot and original delivery")
    args = parser.parse_args()
    manifest = read_json(DELIVERY / "DELIVERY_MANIFEST.json")["files"]
    failures = []
    delivery_aliases = {}
    for name, record in manifest.items():
        path = DELIVERY / name
        # A user may rename the standalone report after delivery. Accept only
        # one byte-identical HTML sibling; preserve both names in the inventory.
        if not path.exists() and name.endswith(".html"):
            matches = [p for p in DELIVERY.glob("*.html")
                       if p.stat().st_size == record["bytes"] and file_sha(p) == record["sha256"]]
            if len(matches) == 1:
                path = matches[0]
                delivery_aliases[name] = path.name
        if not path.is_file() or path.stat().st_size != record["bytes"] or file_sha(path) != record["sha256"]:
            failures.append("delivery hash mismatch: " + name)
    final_archive = DELIVERY.parent / "libero-final130-v10-results.zip"
    if file_sha(final_archive) != FINAL_ARCHIVE_SHA256:
        failures.append("final results archive hash mismatch")
    if failures:
        raise RuntimeError("\n".join(failures))

    zip_path = DELIVERY / "research-source.zip"
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Research source ZIP CRC failed")
        prefix = "versions/v10/"
        frozen = json.loads(archive.read(prefix + "FROZEN_SOURCE.json"))
        names = [n for n in archive.namelist() if n.startswith(prefix) and not n.endswith("/")]
        source_inventory, snapshot_files, differences = [], {}, []
        for name in names:
            short = name[len(prefix):]
            if PurePosixPath(short).is_absolute() or ".." in PurePosixPath(short).parts:
                raise ValueError("Unsafe archive path")
            data = archive.read(name)
            target = SNAPSHOT / short
            if not args.check and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            if not target.is_file() or target.read_bytes() != data:
                failures.append("snapshot mismatch: " + short)
            snapshot_files[short] = {"sha256": sha(data), "bytes": len(data)}
            local = DEV / short
            if not local.is_file() or local.read_bytes() != data:
                differences.append({"path": short, "status": "different" if local.exists() else "missing"})
            if short.endswith(".py"):
                if frozen.get(short) != sha(data):
                    failures.append("frozen Python hash mismatch: " + short)
                if not local.is_file() or local.read_bytes() != data:
                    failures.append("development Python mismatch: " + short)
                source_inventory.append(inventory_python(short, data))
        if set(frozen) != {f["path"] for f in source_inventory} or len(frozen) != 39:
            failures.append("Frozen Python coverage differs from expected 39 files")

        catalog = json.loads(archive.read(prefix + "ALL_TASKS.json"))
        if archive.read(prefix + "ALL_TASKS.json") != (DELIVERY / "ALL_TASKS.json").read_bytes():
            failures.append("Catalog differs between actor snapshot and delivery")
        driver_tree = ast.parse(archive.read(prefix + "run_task.py"))
        driver_overrides = next({kw.arg: {**literal_or_expression(kw.value), "line": kw.value.lineno}
                                 for kw in node.value.keywords}
                                for node in driver_tree.body
                                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cfg" for t in node.targets))
        helpers = {n: {"sha256": sha(archive.read(n)), "bytes": len(archive.read(n))}
                   for n in archive.namelist() if not n.startswith(prefix) and n.endswith(".py")}

    summaries = {name: read_json(DELIVERY / name) for name in
                 ["SUMMARY.json", "PROTOCOL.json", "RUNTIME_VERSIONS.json", "SOURCE_AUDIT.json", "LOCAL_QA.json"]}
    taxonomy = read_json(DELIVERY / "TASK_TAXONOMY.json")
    results = read_json(DELIVERY / "RESULTS.json")
    keys = {row["key"] for row in catalog}
    if len(keys) != 130 or keys != {r["key"] for r in taxonomy} or keys != {r["key"] for r in results}:
        failures.append("130 task identities do not match across catalog/taxonomy/results")
    if len({r["category"] for r in taxonomy}) != 16:
        failures.append("Expected 16 task categories")
    if (summaries["SUMMARY.json"]["full_success"], summaries["SUMMARY.json"]["official_goal_at_end"],
        summaries["SUMMARY.json"]["success_excluding_known_order_violations"]) != (86, 88, 85):
        failures.append("Baseline result definitions/counts changed")
    for r in results:
        result = r["result"]
        if result["initial_state_id"] != 0 or result["driver_sha256"] != frozen["run_task.py"]:
            failures.append("Task init/driver mismatch: " + r["key"])
    full = sum(bool(r["result"].get("pipeline_success")) and bool(r["result"].get("official_goal_at_end"))
               and r["result"].get("steps", 0) > 0 for r in results)
    if full != 86:
        failures.append("Full success count differs from raw results")

    config_keys = sorted(set().union(*(row["config"].keys() for row in catalog)))
    common, varying = {}, {}
    for key in config_keys:
        values = Counter(json.dumps(row["config"].get(key), sort_keys=True, ensure_ascii=False) for row in catalog)
        if len(values) == 1:
            common[key] = json.loads(next(iter(values)))
        else:
            varying[key] = [{"value": json.loads(v), "tasks": count} for v, count in values.items()]
    runtime = summaries["RUNTIME_VERSIONS.json"]["four_gpu_server"]
    external = read_json(DELIVERY / "EXTERNAL_SOURCE_BEFORE.json")
    selected_external = {k: v for k, v in external.items()
                         if k.endswith((".pt", ".pth", ".ckpt")) or k.endswith("graspgen_franka_panda.yml")
                         or k.endswith("grasp_server.py") or k.endswith("curobo/__init__.py")}
    inventories = {
        "BASELINE.json": {"policy_version": "v10", "source_archive": rel(zip_path), "source_archive_sha256": file_sha(zip_path),
                          "snapshot": rel(SNAPSHOT), "snapshot_files": snapshot_files, "frozen_python_files": len(frozen),
                          "development_root": rel(DEV), "development_differences_from_full_snapshot": differences,
                          "driver_sha256": frozen["run_task.py"], "source_basis": "delivered final130 research-source.zip, versions/v10 only",
                          "old_bridge_README_is_historical": True, "protocol": summaries["PROTOCOL.json"]},
        "SOURCE_INVENTORY.json": {"description": "Static AST inventory and coupling search evidence; comments may match. Not a portable interface implementation.",
                                  "groups": dict(Counter(f["group"] for f in source_inventory)), "files": source_inventory,
                                  "evaluation_helpers_in_source_archive": helpers},
        "CONFIG_INVENTORY.json": {"task_catalog": "baseline/v10/ALL_TASKS.json", "task_count": len(catalog),
                                  "catalog_common": common, "catalog_varying": varying, "driver_overrides": driver_overrides,
                                  "precedence": ["catalog row.config", "run_task.py dict overrides including --gpu and --init",
                                                 "destination bootstrapped to target_object", "per-stage target/destination/region/relation and optional surface packing"],
                                  "caveat": "server.json is an older single-task example. Config reads/defaults are not an exhaustive inventory of numeric constants in source."},
        "DEPENDENCIES.json": {"recorded_runtime_not_a_fresh_install_lockfile": True, "roles": runtime,
                              "external_path_settings": {k: common[k] for k in ["libero_repo", "grasp_repo", "graspgen_python", "graspgen_config", "planner_python"]},
                              "external_source_and_weight_fingerprints": selected_external,
                              "full_external_manifest": rel(DELIVERY / "EXTERNAL_SOURCE_BEFORE.json"),
                              "external_sources_unchanged_during_final130": summaries["SOURCE_AUDIT.json"]["external_changes"] == [],
                              "pyproject_does_not_install_runtime_dependencies": True,
                              "external_files_and_weights_not_bundled_in_snapshot": True},
        "ARTIFACT_INDEX.json": {"delivery": rel(DELIVERY), "manifest": rel(DELIVERY / "DELIVERY_MANIFEST.json"),
                                "byte_identical_local_renames": delivery_aliases,
                                "result_archive": rel(final_archive), "result_archive_sha256": FINAL_ARCHIVE_SHA256,
                                "summary": summaries["SUMMARY.json"],
                                "files": {n: record for n, record in manifest.items() if "/" not in n},
                                "per_task_results": rel(DELIVERY / "RESULTS.json"),
                                "video_files": [n for n in manifest if n.endswith(".mp4")],
                                "sequence_caveat": "One final-predicate-successful stack_transport task violates instruction order; exclude it from candidate successful trajectories."},
        "WORKSPACE_INDEX.json": {"work_entries": sorted(p.name for p in (ROOT / "work").iterdir()),
                                 "output_entries": sorted(p.name for p in (ROOT / "outputs").iterdir()),
                                 "classification": {"canonical_policy": rel(SNAPSHOT), "verified_development_python": rel(DEV),
                                                    "canonical_evaluation": rel(DELIVERY),
                                                    "evaluation_helper_worktree": "work/libero-final130-four4090-20261111",
                                                    "separate_upstream_project_not_this_pipeline": "RPent",
                                                    "other_work_and_outputs": "historical/debug/scheduling/analysis; not combined with final130 baseline"}},
    }
    validation = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": not failures,
                  "delivery_files_verified": len(manifest), "frozen_python_verified": len(frozen),
                  "byte_identical_local_renames": delivery_aliases,
                  "snapshot_files_verified": len(snapshot_files), "development_python_match": True,
                  "tasks_verified": len(keys), "categories_verified": 16, "full_success_recomputed": full,
                  "final_results_archive_sha256_match": True, "research_source_zip_crc": True,
                  "policy_or_config_changed": False, "gpu_or_network_used": False,
                  "validation_scope": "Local source/config/artifact identity and static inventory, not fresh trajectory reproduction",
                  "failures": failures}
    if failures:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 1
    if not args.check:
        for name, value in inventories.items():
            write_json(name, value)
        write_json("VALIDATION.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
