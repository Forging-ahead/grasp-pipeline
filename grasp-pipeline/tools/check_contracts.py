#!/usr/bin/env python3
"""Offline schema, reference, source-map and success-accounting validation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.contracts.validation import readiness, success_fields, validate_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    bundle = json.loads((ROOT / "contracts/examples/libero_v10_reference.json").read_text())
    outcomes = json.loads((ROOT / "contracts/examples/libero_v10_outcomes.json").read_text())
    for item in [bundle, *outcomes]: validate_payload(item)
    counts = dict(tasks=len(outcomes), native_success=sum(o["native_success"] is True for o in outcomes),
                  full_pipeline_success=sum(success_fields(o)["full_pipeline_success"] for o in outcomes),
                  candidate_success=sum(success_fields(o)["candidate_success"] for o in outcomes))
    if counts != dict(tasks=130, native_success=88, full_pipeline_success=86, candidate_success=85):
        raise ValueError("Contract normalization changed v10 accounting")
    mapping = json.loads((ROOT / "contracts/LIBERO_V10_BINDINGS.json").read_text())
    for anchor in mapping["ports"]:
        if hashlib.sha256((ROOT / anchor["source"]).read_bytes()).hexdigest() != anchor["sha256"]:
            raise ValueError("Reference source anchor changed: " + anchor["source"])
    old = json.loads((ROOT / "migration/TESTED_SOURCE_SHA256.json").read_text())
    differences = [f for f, sha in old.items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != sha]
    if differences: raise ValueError("Phase2 source/config changed: " + str(differences))
    report = dict(passed=True, checked_at_utc=datetime.now(timezone.utc).isoformat(), jsonschema_version=importlib.metadata.version("jsonschema"),
                  normalized_outcomes_validated=len(outcomes), reference_bundle_valid=True, original_success_counts=counts,
                  phase2_files_unchanged=len(old), source_anchors_checked=len(mapping["ports"]),
                  reference_readiness=readiness(bundle), live_adapter_implemented=False, gpu_rollouts_started=0)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
