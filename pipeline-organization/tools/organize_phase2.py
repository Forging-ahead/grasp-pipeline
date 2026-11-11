#!/usr/bin/env python3
"""Reproduce the mechanical phase2 relocation from the frozen v10 snapshot.

Creates original-derived files only. Refuses to overwrite a different file.
The exact replacement list is retained for independent preservation checks.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

WORKSPACE = Path(__file__).resolve().parents[2]
BASE = WORKSPACE / "pipeline-organization/baseline/v10"
DEST = Path(os.environ.get("GRASP_PIPELINE_BUILD_DEST", str(WORKSPACE / "grasp-pipeline"))).resolve()
PKG = "graspgen_libero_bridge"
GROUPS = {
    "policy": ["evaluate", "task_policy", "skills", "placement", "transfer", "surface_packing"],
    "models": ["graspgen_worker"],
    "planning": ["planner_worker", "planning_geometry", "cartesian", "grasp_candidate", "grasp_contact"],
    "perception": ["pointcloud"],
    "adapters.libero": ["env_adapter", "scene", "regions", "placement_geometry", "articulation"],
    "control": ["controller"],
    "common": ["frames", "rpc", "errors"],
}
MODULES = {stem: PKG + "." + group + "." + stem for group, stems in GROUPS.items() for stem in stems}
records = []


def put(name, text):
    path = DEST / name
    data = text if isinstance(text, bytes) else text.encode()
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError("Refusing to replace edited file: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def relocate(original, target, extra=()):
    source = (BASE / original).read_text()
    edits = []

    def replace(old, new, category):
        nonlocal source
        count = source.count(old)
        if count:
            edits.append(dict(old=old, new=new, count=count, category=category))
            source = source.replace(old, new)

    # Longest names first: placement must not consume placement_geometry.
    for stem, module in sorted(MODULES.items(), key=lambda item: -len(item[0])):
        replace("from ." + stem + " import", "from " + module + " import", "import_relocation")
        # Includes module names in worker command strings and embedded RPC test code.
        replace(PKG + "." + stem, module, "module_relocation")
    replace("from " + PKG + " import evaluate", "from " + PKG + ".policy import evaluate", "import_relocation")
    for old, new, category in extra:
        if old not in source:
            raise RuntimeError("Expected edit source absent in " + original + ": " + old)
        replace(old, new, category)
    put(target, source)
    records.append(dict(original=original, target=target,
                        original_sha256=hashlib.sha256((BASE / original).read_bytes()).hexdigest(),
                        organized_sha256=hashlib.sha256(source.encode()).hexdigest(), edits=edits))


def main():
    for group, stems in GROUPS.items():
        for stem in stems:
            extra = []
            if stem == "evaluate":
                extra = [
                    ('from pathlib import Path', 'from pathlib import Path\nfrom graspgen_libero_bridge.common.layout import source_root, source_fingerprints', 'runtime_import'),
                    ('{p.name: hashlib.sha256(p.read_bytes()).hexdigest()\n            for p in sorted(Path(__file__).resolve().parent.glob("*.py"))}', 'source_fingerprints()', 'source_provenance'),
                    ('root = str(Path(__file__).resolve().parent.parent)', 'root = str(source_root())', 'worker_pythonpath'),
                ]
            if stem == "skills":
                extra = [
                    ("str(Path(__file__).resolve().parents[1])", "str(Path(__file__).resolve().parents[2])", "worker_pythonpath"),
                ]
            relocate("bridge/" + PKG + "/" + stem + ".py", "src/" + MODULES[stem].replace(".", "/") + ".py", extra)
    relocate("bridge/" + PKG + "/__init__.py", "src/" + PKG + "/__init__.py")
    relocate("bridge/" + PKG + "/__main__.py", "src/" + PKG + "/runtime/legacy_cli.py", [
        ("root = Path(__file__).resolve().parent.parent", "root = Path(__file__).resolve().parents[3]", "checkout_path"),
        ('root / "configs/server.json"', 'root / "configs/examples/single_task.v10.json"', 'config_location'),
        ('parser.add_argument("--candidate-offset", type=int)', 'parser.add_argument("--candidate-offset", type=int)\n    parser.add_argument("--paths", type=Path)', 'launcher_options'),
        ('config = json.loads(args.config.read_text())', 'config = json.loads(args.config.read_text())\n    if args.paths is not None:\n        from graspgen_libero_bridge.runtime.config import PATH_KEYS, load_paths\n        deployment = load_paths(args.paths)\n        config.update({key: deployment[key] for key in PATH_KEYS})', 'legacy_deployment_paths'),
    ])
    for test in sorted((BASE / "bridge/tests").glob("*.py")):
        extra = []
        if test.name == "test_rpc.py":
            extra = [('Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parents[2]', 'test_checkout_path'),
                     ('str(project)])', 'str(project / "src")])', 'test_pythonpath')]
        relocate("bridge/tests/" + test.name, "tests/v10/" + test.name, extra)

    source = (BASE / "run_task.py").read_text()
    start = source.index("rows=json.loads")
    end = source.index("# Some unsupported-task")
    config_block = source[start:end]
    tree = ast.parse(source)
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "cfg" for t in n.targets))
    overrides = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords
                 if kw.arg not in ["simulation_gpu", "model_gpu", "planner_gpu", "init_state_id"]}
    relocate("run_task.py", "apps/libero/run_task.py", [
        ("args=p.parse_args()", "p.add_argument('--paths',type=Path)\np.add_argument('--output-root',type=Path)\nargs=p.parse_args()", "launcher_options"),
        ("r=Path(__file__).resolve().parent", "r=Path(__file__).resolve().parents[2]", "checkout_path"),
        ("str(r/'bridge')", "str(r/'src')", "source_pythonpath"),
        (config_block, "from graspgen_libero_bridge.runtime.config import load_task\nrow,cfg=load_task(args.key,gpu=args.gpu,init=args.init,paths_file=args.paths)\n", "config_loading"),
        ("out=r/'runs'/args.tag/args.key", "out=(args.output_root or r/'runs')/args.tag/args.key", "output_location"),
        ("(r/'bridge/graspgen_libero_bridge').glob('*.py')", "(r/'src/graspgen_libero_bridge').rglob('*.py')", "source_provenance"),
    ])
    for name, data in {
        "configs/libero/tasks.v10.json": (BASE / "ALL_TASKS.json").read_bytes(),
        "configs/examples/single_task.v10.json": (BASE / "bridge/configs/server.json").read_bytes(),
        "configs/policy/v10_overrides.json": (json.dumps(overrides, indent=2) + "\n").encode(),
    }.items():
        put(name, data)
    catalog = json.loads((BASE / "ALL_TASKS.json").read_text())
    paths = {k: catalog[0]["config"][k] for k in
             ["libero_repo", "grasp_repo", "graspgen_python", "graspgen_config", "planner_python"]}
    paths["simulation_python"] = "/mnt/public/changwei/Graspgen/libero-final130-four4090-20261111/runtime/openpi/bin/python"
    put("configs/runtime/paths.server.json", json.dumps(paths, indent=2) + "\n")

    source_zip = WORKSPACE / "outputs/libero-final130-four4090-20261111/delivery/research-source.zip"
    with zipfile.ZipFile(source_zip) as archive:
        for name in archive.namelist():
            if "/" not in name and name.endswith(".py"):
                put("tools/legacy_evaluation/" + name, archive.read(name))
    package_dirs = set()
    for f in (DEST / "src" / PKG).rglob("*.py"):
        package_dirs.update(p for p in f.parents if p != DEST / "src" and DEST / "src" in p.parents)
    package_dirs.add(DEST / "src" / PKG / "runtime")
    for folder in package_dirs:
        if not (folder / "__init__.py").exists():
            put(str((folder / "__init__.py").relative_to(DEST)), '"""' + folder.name + ' components; backend imports remain lazy."""\n')
    put("migration/V10_RELOCATION.json", json.dumps({"policy_baseline": "v10", "files": records,
        "scope": "Mechanical module moves/imports, process search paths, configuration loading, provenance, and output location only; no policy algorithm edits"}, indent=2) + "\n")
    print(json.dumps({"derived_python_files": len(records), "policy_override_keys": len(overrides), "output": str(DEST)}, indent=2))


if __name__ == "__main__":
    main()
