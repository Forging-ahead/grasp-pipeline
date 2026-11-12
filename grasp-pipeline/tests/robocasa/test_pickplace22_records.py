"""Coverage denominators and success eligibility for the 22-task rerun."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools/robocasa"
sys.path.insert(0, str(TOOLS))
from pickplace22_records import summarize
from graspgen_libero_bridge.runtime.robocasa_provenance import digest_json


def fixture(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    policy = {"control_freq": 20, "lift_clearance_m": .15}
    matrix = {"planned_episodes": 22, "source_digest": "new", "policy_digest": digest_json(policy),
              "baseline_source_digest": "old", "cells": []}
    frozen = {}
    for index in range(22):
        task = {"task_id": "Task" + str(index)}; name = task["task_id"]
        baseline = tmp_path / "baseline" / name; case = tmp_path / "cases" / name
        config = {"gpu": 0, "policy": policy, "task": task}
        write(baseline / "RESULT.json", {"source_digest": "old", "status": "failed"})
        write(case / "CONFIG.json", config); frozen[name] = config
        write(case / "LAUNCHER.json", {"return_code": 0})
        write(case / "INITIAL_COMPARISON.json", {"matched": True})
        write(case / "RESULT.json", {"task_id": name, "seed": 0, "split": "pretrain", "mode": "run",
            "source_digest": "new", "source_unchanged_during_run": True, "policy_digest": digest_json(policy),
            "candidate_source": "fresh_graspgen_inference", "paired_diagnostic": False, "status": "failed",
            "initial_state_verified": True, "native_success_at_reset": False, "physical_steps": 0})
        matrix["cells"].append({"case": name, "task_id": name, "seed": 0, "split": "pretrain",
            "baseline_case": str(baseline), "task_binding_digest": digest_json(task)})
    write(tmp_path / "MATRIX.json", matrix); write(tmp_path / "FROZEN_CONFIGS.json", frozen)
    write(tmp_path / "SWEEP_COMPLETE.json", {"completed": 22})
    return tmp_path


@pytest.mark.parametrize("change", ["valid", "zero_actions", "wrong_version", "wrong_initial_state"])
def test_only_a_real_same_version_same_initial_state_completion_counts(tmp_path, change):
    root = fixture(tmp_path); case = root / "cases/Task0"
    p = case / "RESULT.json"; result = json.loads(p.read_text())
    result.update(status="success", pipeline_success=True, native_success_at_end=True, physical_steps=100,
                  native_horizon=600, within_native_horizon=True, native_budget_success=True)
    if change == "zero_actions": result["physical_steps"] = 0
    if change == "wrong_version": result["source_digest"] = "other"
    if change == "wrong_initial_state": (case / "INITIAL_COMPARISON.json").write_text('{"matched":false}')
    p.write_text(json.dumps(result))
    summary = summarize(root)
    assert summary["complete"] and summary["planned"] == 22 and summary["completed"] == 22
    assert summary["success"] == (1 if change == "valid" else 0)
    assert summary["success_rate"] == (1 / 22 if change == "valid" else 0)


def test_a_missing_case_is_not_silently_dropped_from_the_denominator(tmp_path):
    root = fixture(tmp_path)
    (root / "cases/Task21/LAUNCHER.json").unlink()
    summary = summarize(root)
    assert not summary["complete"] and summary["planned"] == 22 and summary["completed"] == 21
    assert summary["success_rate"] is None


def test_setup_recovery_must_finish_before_publishing_final_rate(tmp_path):
    root = fixture(tmp_path)
    path = root / "INITIALIZATION_RECOVERY.json"
    path.write_text('{"complete":false,"no_policy_outcome_retry":true}')
    assert not summarize(root)["complete"] and summarize(root)["success_rate"] is None
    path.write_text('{"complete":true,"no_policy_outcome_retry":true,"rejections":[{"physical_steps":0}]}')
    assert summarize(root)["complete"] and summarize(root)["planned"] == 22
