#!/usr/bin/env python3
"""Verify the integration guide, examples and existing lightweight CLI.

No simulator, model or GPU is started. --schema uses optional jsonschema.
--schema-only verifies just the portable public-payload examples.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/integration"
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_schemas():
    from graspgen_libero_bridge.contracts.validation import validate_payload
    import importlib.metadata
    names = ["task_spec.example.json", "capabilities.example.json"]
    for name in names:
        validate_payload(read(GUIDE / "templates" / name))
    return dict(public_examples_validated=names, jsonschema_version=importlib.metadata.version("jsonschema"),
                example_sha256={name: digest(GUIDE / "templates" / name) for name in names},
                schema_sha256=digest(ROOT / "contracts/pipeline-v1.schema.json"))


def check_local():
    report = {}
    files = [ROOT.parent / "README.md", ROOT / "README.md", *(ROOT / "docs").rglob("*.md"),
             ROOT.parent / "pipeline-organization/README.md", ROOT.parent / "pipeline-organization/PLAN.md"]
    links = 0
    for file in files:
        if not file.exists():
            continue  # Optional workspace navigation is absent in a standalone copy.
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", file.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            if not (file.parent / unquote(target.split("#")[0])).exists():
                raise ValueError("Broken document link: " + str(file) + ": " + target)
            links += 1
    report["local_document_links_checked"] = links
    manifest = read(ROOT / "migration/phase3/TESTED_SOURCE_SHA256.json")
    for name, expected in manifest.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Phase3 source/config/reference changed: " + name)
    report["phase3_source_and_contract_files_unchanged"] = len(manifest)
    mapping = read(GUIDE / "FILE_CHANGE_MAP.json")
    for record in mapping["existing_sources"]:
        path = ROOT / record["source"]
        if digest(path) != record["sha256"]:
            raise ValueError("Source-change guide anchor drift: " + str(path))
        nodes = ast.parse(path.read_text()).body
        for piece in record["symbol"].split("."):
            node = next(n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == piece)
            nodes = node.body
        if node.lineno != record["line"]:
            raise ValueError("Source-change guide line drift: " + str(path))
    if any(record["kind"] != "proposed_not_implemented" for record in mapping["proposed_paths"]):
        raise ValueError("Proposed integration paths must not claim implementation")
    report["source_change_anchors_checked"] = len(mapping["existing_sources"])
    report["proposed_path_patterns_marked_unimplemented"] = len(mapping["proposed_paths"])
    sources = read(GUIDE / "SOURCE_REFERENCES.json")
    rpent = ROOT.parent / sources["rpent_reference"]["root"]
    if rpent.exists():
        head = subprocess.check_output(["git", "-C", str(rpent), "rev-parse", "HEAD"], text=True).strip()
        if head != sources["rpent_reference"]["head"]:
            raise ValueError("RPent reference checkout HEAD changed")
        for name, expected in sources["rpent_reference"]["files"].items():
            if digest(rpent / name) != expected:
                raise ValueError("RPent reference source changed: " + name)
        report["rpent_reference_files_checked"] = len(sources["rpent_reference"]["files"])
    else:
        report["rpent_reference_files_checked"] = 0
        report["rpent_reference_note"] = "Optional checkout absent; no local RPent source verification claimed"
    template = read(GUIDE / "templates/integration_spec.example.json")
    if template["template_only"] is not True or template["consumed_by_current_cli"] is not False:
        raise ValueError("Design sheet must be explicitly illustrative")
    from graspgen_libero_bridge.runtime.config import load_paths
    load_paths(GUIDE / "templates/paths.libero.example.json")
    try:
        load_paths(GUIDE / "templates/integration_spec.example.json")
    except ValueError:
        pass
    else:
        raise ValueError("Design sheet unexpectedly accepted as runtime paths")
    capabilities = read(GUIDE / "templates/capabilities.example.json")
    if capabilities["binding_status"] != "reference_mapping_only" or any(capabilities["features"].values()) or capabilities["available_skills"]:
        raise ValueError("Unimplemented capabilities example must not claim readiness")
    report["configuration_examples_checked"] = 4
    key = "libero_object_pick_up_the_orange_juice_and_place_it_in_the_basket"
    commands = [["--help"], ["list", "--json"], ["config", key, "--gpu", "0", "--init", "0"],
                ["run", key, "--gpu", "0", "--init", "0", "--tag", "integration-preview", "--dry-run"]]
    for arguments in commands:
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), *arguments], cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=20)
        if arguments[0] == "list" and len(json.loads(result.stdout)) != 130:
            raise ValueError("Unexpected task catalog length")
        if arguments[0] == "run" and json.loads(result.stdout)["simulation_started"] is not False:
            raise ValueError("Dry-run launched simulation")
    report["lightweight_cli_commands_checked"] = len(commands)
    report["schema_examples_checked"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {} if args.schema_only else check_local()
    if args.schema or args.schema_only:
        report.update(check_schemas(), schema_examples_checked=True)
    report.update(passed=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  benchmark_adapters_implemented=False, gpu_rollouts_started=0,
                  scope="Integration documentation/examples and unchanged existing code; no new benchmark or task validation")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
