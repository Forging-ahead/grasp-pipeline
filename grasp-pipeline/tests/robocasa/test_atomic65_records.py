"""Coverage probes must never become successes or disappear from the denominator."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools/robocasa"
sys.path.insert(0, str(TOOLS))
from report_atomic65 import summarize
from graspgen_libero_bridge.runtime.robocasa_provenance import digest_json


def fixture(tmp_path):
    policy = {"budget_profile": "libero-v10-teacher"}
    tasks = [{"task_id": "task_" + str(i), "stages": []} for i in range(65)]
    bindings = {"tasks": tasks}
    cells = [{"task_id": t["task_id"], "seed": 0, "split": "pretrain", "category": "door", "role": "atomic65_coverage",
              "execution_mode": "probe", "binding_status": "unsupported_no_task_action", "unsupported_reason": "disabled"} for t in tasks]
    matrix = {"planned_episodes": 65, "cells": cells, "source_digest": "frozen", "policy_digest": digest_json(policy),
              "task_bindings_digest": digest_json(bindings), "budget_profile": "libero-v10-teacher", "work_deadline_s": 890}
    (tmp_path / "MATRIX.json").write_text(json.dumps(matrix))
    (tmp_path / "TASK_BINDINGS.json").write_text(json.dumps(bindings))
    case = tmp_path / "cases/task_0__seed000000"; case.mkdir(parents=True)
    (case / "CONFIG.json").write_text(json.dumps({"policy": policy, "task": tasks[0]}))
    (case / "LAUNCHER.json").write_text('{"return_code":0}')
    result = {"task_id": "task_0", "seed": 0, "split": "pretrain", "mode": "probe", "source_digest": "frozen",
              "source_unchanged_during_run": True, "policy_digest": digest_json(policy), "budget_profile": "libero-v10-teacher",
              "status": "probe_passed", "pipeline_success": True, "native_success_at_end": True,
              "native_success_at_reset": True, "physical_steps": 0}
    (case / "RESULT.json").write_text(json.dumps(result))
    return case, result


def test_native_probe_pass_and_reset_success_are_not_new_policy_successes(tmp_path):
    fixture(tmp_path)
    summary = summarize(tmp_path)
    assert summary["groups"]["all"]["planned"] == 65
    assert summary["groups"]["all"]["completed"] == 1
    assert summary["groups"]["all"]["unsupported"] == 1
    assert summary["groups"]["all"]["success"] == 0
    assert summary["overall_success_rate"] is None


def test_unknown_actions_in_a_declared_zero_action_probe_are_rejected(tmp_path):
    case, result = fixture(tmp_path)
    result["physical_steps"] = 1
    (case / "RESULT.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="unexpectedly executed"):
        summarize(tmp_path)


@pytest.mark.parametrize("change", [{"source_digest": "different"}, {"paired_diagnostic": True}])
def test_mixed_actor_or_cached_candidates_are_not_valid_coverage_evidence(tmp_path, change):
    case, result = fixture(tmp_path)
    result.update(change)
    (case / "RESULT.json").write_text(json.dumps(result))
    summary = summarize(tmp_path)
    assert summary["rows"][0]["failure_code"] == "VERSION_OR_IDENTITY_MISMATCH"
    assert summary["groups"]["all"]["success"] == 0


def test_binding_edits_after_freezing_are_detected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "TASK_BINDINGS.json"
    binding = json.loads(path.read_text()); binding["tasks"][0]["stages"] = [{"skill": "pick_place"}]
    path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="changed semantic bindings"):
        summarize(tmp_path)
