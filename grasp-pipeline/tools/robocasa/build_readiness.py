#!/usr/bin/env python3
"""Build E2 readiness only from real native development episodes and control checks."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.adapters.robocasa.config import catalog, load_config
from graspgen_libero_bridge.runtime.robocasa_provenance import source_identity, digest_json


def require_fresh_native(record):
    if record.get("budget_profile", "native") != "native":
        raise ValueError("extended collection budget cannot qualify native main evaluation")
    if record.get("paired_diagnostic") or record.get("candidate_source", "fresh_graspgen_inference") != "fresh_graspgen_inference":
        raise ValueError("cached or geometric diagnostic cannot qualify fresh-model main evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--control-validation", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = source_identity()
    policy = load_config("PickPlaceCounterToStove", policy_file=args.policy)["policy"]
    policy_digest = digest_json(policy)
    errors, evidence, successes = [], [], {}
    for task in (t for t in catalog() if t["role"] == "development"):
        successes[task["task_id"]] = 0
        for seed in range(5):
            directory = args.development_root / "cases" / (task["task_id"] + f"__seed{seed:06d}")
            path = directory / "RESULT.json"
            try:
                record = json.loads(path.read_text())
                lock = json.loads((directory / "ENVIRONMENT_LOCK.json").read_text())
                if record["task_id"] != task["task_id"] or record["seed"] != seed or record["split"] != "pretrain":
                    raise ValueError("episode identity mismatch")
                if record["mode"] != "run" or record["source_digest"] != identity["source_digest"] or record["policy_digest"] != policy_digest:
                    raise ValueError("source/policy/mode mismatch")
                if lock["benchmark"] != "RoboCasa365" or not record["source_unchanged_during_run"]:
                    raise ValueError("not a consistent native RoboCasa episode")
                require_fresh_native(record)
                supervisor_path = directory / "SUPERVISOR.json"
                if supervisor_path.is_file():
                    supervisor = json.loads(supervisor_path.read_text())
                    if supervisor.get("timed_out") or supervisor.get("interrupted_signal"):
                        raise ValueError("episode exceeded its wall-clock budget or was interrupted")
                if record.get("pipeline_success"):
                    recording = json.loads((directory / "RECORDING.json").read_text())
                    if not recording["alignment_valid"] or not record["video_available"]:
                        raise ValueError("successful episode lacks valid recording")
                    if not record.get("native_success_at_end") or record.get("native_success_at_reset") or record.get("physical_steps", 0) <= 0:
                        raise ValueError("no newly achieved physical native success")
                    successes[task["task_id"]] += 1
                evidence.append(str(path.resolve()))
            except (OSError, ValueError, KeyError) as exc:
                errors.append(str(path) + ": " + str(exc))
        minimum = 3 if task["task_id"] in {"PickPlaceCounterToStove", "PickPlaceSinkToCounter"} else 1
        if successes[task["task_id"]] < minimum:
            errors.append(task["task_id"] + ": insufficient native development successes")
    try:
        result = json.loads((args.control_validation / "RESULT.json").read_text())
        controls = json.loads((args.control_validation / "CONTROL_CHECKS.json").read_text())
        lock = json.loads((args.control_validation / "ENVIRONMENT_LOCK.json").read_text())
        if result["mode"] != "validate" or result["status"] != "validation_passed" or not result["validation_executed"]:
            raise ValueError("native control validation did not pass")
        if (lock["benchmark"] != "RoboCasa365" or result["source_digest"] != identity["source_digest"]
                or result["policy_digest"] != policy_digest or not result.get("source_unchanged_during_run")
                or sum(bool(c["executed"]) for c in controls) < 10):
            raise ValueError("control evidence version/count mismatch")
    except (OSError, ValueError, KeyError) as exc:
        errors.append("control validation: " + str(exc))
    report = {"schema_version": "robocasa-readiness-1", "passed": not errors,
              "source_digest": identity["source_digest"], "policy_digest": policy_digest,
              "successes": successes, "evidence": evidence, "errors": errors}
    if args.output.exists():
        parser.error("Refusing to overwrite existing readiness evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "errors": len(errors), "evidence_files": len(evidence)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
