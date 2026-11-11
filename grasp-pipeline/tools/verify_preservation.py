#!/usr/bin/env python3
"""Verify mechanical relocation and all 130 effective configs against v10.

Uses the standard library only. Does not import policy/model/simulator modules.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.runtime.config import load_task


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(baseline):
    relocation = json.loads((ROOT / "migration/V10_RELOCATION.json").read_text())
    frozen = json.loads((baseline / "FROZEN_SOURCE.json").read_text())
    checked, differences = [], []
    categories = {"import_relocation", "module_relocation", "runtime_import", "source_provenance", "worker_pythonpath",
                  "checkout_path", "config_location", "test_pythonpath", "test_checkout_path", "launcher_options",
                  "source_pythonpath", "config_loading", "output_location", "legacy_deployment_paths"}
    for record in relocation["files"]:
        name = record["original"]
        data = (baseline / name).read_bytes()
        if sha(data) != frozen[name] or sha(data) != record["original_sha256"]:
            raise ValueError("Original frozen source hash mismatch: " + name)
        expected = data.decode()
        for edit in record["edits"]:
            if edit["category"] not in categories or expected.count(edit["old"]) != edit["count"]:
                raise ValueError("Invalid mechanical edit record: " + name)
            expected = expected.replace(edit["old"], edit["new"])
        target = ROOT / record["target"]
        actual = target.read_bytes()
        if actual != expected.encode() or sha(actual) != record["organized_sha256"]:
            raise ValueError("Unrecorded code change: " + record["target"])
        ast.parse(actual, filename=str(target))
        checked.append({"original": name, "target": record["target"], "edits": len(record["edits"]),
                        "original_sha256": sha(data), "organized_sha256": sha(actual), "mechanical_replay_match": True})
    if {r["original"] for r in relocation["files"]} != set(frozen) or len(checked) != 39:
        raise ValueError("Frozen Python file coverage changed")
    internal_imports = 0
    for file in [*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py"), *(ROOT / "apps").rglob("*.py")]:
        for node in ast.walk(ast.parse(file.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                     [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name == "graspgen_libero_bridge" or name.startswith("graspgen_libero_bridge."):
                    path = ROOT / "src" / name.replace(".", "/")
                    if not path.with_suffix(".py").exists() and not (path / "__init__.py").exists():
                        raise ValueError("Unresolved internal import in " + str(file) + ": " + name)
                    internal_imports += 1
    original_catalog = (baseline / "ALL_TASKS.json").read_bytes()
    if original_catalog != (ROOT / "configs/libero/tasks.v10.json").read_bytes():
        raise ValueError("Task catalog bytes changed")
    tree = ast.parse((baseline / "run_task.py").read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "cfg" for t in n.targets))
    overrides = {kw.arg: ast.literal_eval(kw.value) for kw in assignment.value.keywords
                 if kw.arg not in ["simulation_gpu", "model_gpu", "planner_gpu", "init_state_id"]}
    if overrides != json.loads((ROOT / "configs/policy/v10_overrides.json").read_text()):
        raise ValueError("Extracted driver override values changed")
    if (baseline / "bridge/configs/server.json").read_bytes() != (ROOT / "configs/examples/single_task.v10.json").read_bytes():
        raise ValueError("Legacy single-task configuration changed")
    oracle = compile(ast.Module(body=[copy.deepcopy(assignment)], type_ignores=[]), "v10_config_oracle", "exec")
    comparisons = 0
    catalog = json.loads(original_catalog)
    for row in catalog:
        for gpu in [0, 1, 2, 3]:
            for init in [0, 1]:
                namespace = {"row": copy.deepcopy(row), "args": SimpleNamespace(gpu=gpu, init=init)}
                exec(oracle, namespace)
                expected = namespace["cfg"]
                expected["destination"] = expected["target_object"]
                new_row, actual = load_task(row["key"], gpu=gpu, init=init)
                if actual != expected or new_row != row:
                    differences.append(dict(key=row["key"], gpu=gpu, init=init))
                comparisons += 1
    if differences:
        raise ValueError("Effective configuration differs: " + json.dumps(differences))
    return {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "mapped_original_python_files": len(checked), "all_mechanical_edits_replay_exactly": True,
            "internal_import_paths_resolved": internal_imports,
            "task_catalog_bytes_unchanged": True, "single_task_example_bytes_unchanged": True,
            "literal_driver_overrides_unchanged": len(overrides), "task_count": len(catalog),
            "gpu_values": [0, 1, 2, 3], "init_values_for_config_binding_check": [0, 1],
            "effective_config_comparisons": comparisons, "effective_config_differences": differences,
            "source_files": checked,
            "scope": "Static/mechanical preservation plus exact effective-configuration comparison; not a fresh GPU trajectory equivalence test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT.parent / "pipeline-organization/baseline/v10")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.baseline)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_files"}, indent=2))


if __name__ == "__main__":
    main()
