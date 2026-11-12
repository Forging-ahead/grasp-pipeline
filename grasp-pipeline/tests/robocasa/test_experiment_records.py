"""A missing, mixed-version or timed-out run must never become a success."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/robocasa" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matrix(tmp_path):
    cells = [{"task_id": "PickPlaceCounterToStove", "seed": i, "split": "target",
              "category": "open_area_pick_place", "role": "development"} for i in range(3)]
    (tmp_path / "MATRIX.json").write_text(json.dumps({"cells": cells, "planned_episodes": 3,
                "source_digest": "code", "policy_digest": "policy"}))
    return cells


def result(tmp_path, cell, **extra):
    directory = tmp_path / "cases" / (cell["task_id"] + f"__seed{cell['seed']:06d}")
    directory.mkdir(parents=True)
    record = dict(cell, source_digest="code", policy_digest="policy", mode="run",
                  source_unchanged_during_run=True, native_success_at_end=True,
                  pipeline_success=True, training_candidate=False)
    record.update(extra)
    (directory / "RESULT.json").write_text(json.dumps(record))
    return directory


def test_pending_cells_keep_denominator_without_claiming_final_success_rate(tmp_path):
    cells = matrix(tmp_path)
    result(tmp_path, cells[0])
    summary = tool("build_report").summarize(tmp_path)
    group = summary["groups"]["all"]
    assert (group["planned"], group["attempted"], group["pending"], group["pipeline_success"]) == (3, 1, 2, 1)
    assert group["pipeline_success_rate"] is None and group["wilson_95_percent"] is None


def test_mixed_version_and_timed_out_success_flags_are_excluded(tmp_path):
    cells = matrix(tmp_path)
    result(tmp_path, cells[0])
    result(tmp_path, cells[1], source_digest="old-code")
    directory = result(tmp_path, cells[2])
    (directory / "SUPERVISOR.json").write_text(json.dumps({"timed_out": True}))
    summary = tool("build_report").summarize(tmp_path)
    assert summary["groups"]["all"]["pipeline_success"] == 1
    assert summary["groups"]["all"]["native_success"] == 1
    assert summary["failures"] == {"VERSION_OR_IDENTITY_MISMATCH": 1, "WALL_TIMEOUT": 1}


def test_complete_matrix_reports_uncertainty_and_all_failures(tmp_path):
    cells = matrix(tmp_path)
    for i, cell in enumerate(cells):
        result(tmp_path, cell, pipeline_success=i == 0, native_success_at_end=i == 0,
               failure_code=None if i == 0 else "GRASP_FAILED")
    summary = tool("build_report").summarize(tmp_path)
    group = summary["groups"]["all"]
    assert group["complete"] and group["pipeline_success_rate"] == 1 / 3
    lo, hi = group["wilson_95_percent"]
    assert 0 < lo < 1 / 3 < hi < 1
    assert summary["failures"]["GRASP_FAILED"] == 2


def test_duplicate_cells_cannot_inflate_denominator(tmp_path):
    cells = matrix(tmp_path)
    path = tmp_path / "MATRIX.json"
    data = json.loads(path.read_text()); data["cells"][1] = cells[0]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        tool("build_report").summarize(tmp_path)


def test_interrupted_episode_cannot_be_a_success(tmp_path):
    cells = matrix(tmp_path)
    directory = result(tmp_path, cells[0])
    (directory / "SUPERVISOR.json").write_text(json.dumps({"interrupted_signal": 15}))
    summary = tool("build_report").summarize(tmp_path)
    assert summary["rows"][0]["failure"] == "PROCESS_INTERRUPTED"
    assert summary["groups"]["all"]["pipeline_success"] == 0
    assert summary["task_coverage"] == {"planned_tasks": 1, "tasks_with_pipeline_success": 0}


@pytest.mark.parametrize("extra", [{"paired_diagnostic": True}, {"budget_profile": "libero-v10-teacher"}])
def test_cached_or_extended_budget_results_cannot_enter_native_fresh_matrix(tmp_path, extra):
    cells = matrix(tmp_path)
    result(tmp_path, cells[0], **extra)
    summary = tool("build_report").summarize(tmp_path)
    assert summary["groups"]["all"]["pipeline_success"] == 0
    assert summary["rows"][0]["failure"] == "VERSION_OR_IDENTITY_MISMATCH"


def test_missing_or_invalid_horizon_never_establishes_native_budget_success():
    report = tool("build_report")
    record = {"pipeline_success": True, "native_success_at_end": True, "physical_steps": 710}
    for horizon in (None, 0, True, float("inf")):
        assert not report.native_budget_success(dict(record, native_horizon=horizon))
    assert not report.native_budget_success(dict(record, native_horizon=600))
    assert report.native_budget_success(dict(record, native_horizon=900))


def test_diagnostics_cannot_satisfy_main_evaluation_readiness():
    readiness = tool("build_readiness")
    readiness.require_fresh_native({"budget_profile": "native", "candidate_source": "fresh_graspgen_inference"})
    for record in ({"budget_profile": "libero-v10-teacher"}, {"paired_diagnostic": True},
                   {"candidate_source": "recorded_identical_pool"}, {"candidate_source": "partial_cloud_geometric"}):
        with pytest.raises(ValueError):
            readiness.require_fresh_native(record)
