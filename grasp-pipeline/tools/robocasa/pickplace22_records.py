"""One row per frozen rerun; never substitute historical successes."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.runtime.robocasa_provenance import digest_json
from build_report import explain_failure, native_budget_success


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def summarize(root):
    root = Path(root)
    matrix = read(root / "MATRIX.json")
    if matrix["planned_episodes"] != 22 or len(matrix["cells"]) != 22 or len({r["task_id"] for r in matrix["cells"]}) != 22:
        raise ValueError("The denominator must be the same 22 baseline tasks")
    frozen = read(root / "FROZEN_CONFIGS.json")
    rows = []
    for cell in matrix["cells"]:
        case = root / "cases" / cell["case"]
        previous = Path(cell["baseline_case"])
        baseline = read(previous / "RESULT.json") or {}
        config = read(case / "CONFIG.json") or {}
        result = read(case / "RESULT.json") or {}
        supervisor = read(case / "SUPERVISOR.json") or {}
        completed = bool(read(case / "LAUNCHER.json"))
        initial = read(case / "INITIAL_COMPARISON.json") or {}
        expected = {k: v for k, v in frozen[cell["case"]].items() if k != "gpu"}
        actual = {k: v for k, v in config.items() if k != "gpu"}
        identity = bool(result and config and all(result.get(k) == cell[k] for k in ("task_id", "seed", "split"))
            and result.get("mode") == "run" and result.get("source_digest") == matrix["source_digest"]
            and result.get("source_unchanged_during_run") and result.get("policy_digest") == matrix["policy_digest"]
            and digest_json(config["policy"]) == matrix["policy_digest"] and actual == expected
            and digest_json(config["task"]) == cell["task_binding_digest"]
            and result.get("candidate_source") == "fresh_graspgen_inference" and not result.get("paired_diagnostic")
            and baseline.get("source_digest") == matrix["baseline_source_digest"])
        interrupted = bool(supervisor.get("interrupted_signal"))
        timed_out = bool(supervisor.get("timed_out") or result.get("failure_code") == "WALL_TIMEOUT")
        success = bool(completed and identity and initial.get("matched") and result.get("initial_state_verified")
            and not timed_out and not interrupted and result.get("status") == "success" and result.get("pipeline_success")
            and result.get("native_success_at_end") and not result.get("native_success_at_reset") and result.get("physical_steps", 0) > 0)
        failure = result.get("failure_code")
        if not completed:
            status, failure = ("running", "RUNNING") if config else ("pending", "NOT_RUN")
        elif timed_out:
            status, failure = "technical_failure", "WALL_TIMEOUT"
        elif interrupted:
            status, failure = "technical_failure", "PROCESS_INTERRUPTED"
        elif not result:
            status, failure = "technical_failure", "PROCESS_ERROR"
        elif failure == "INITIAL_STATE_MISMATCH":
            status = "technical_failure"
        elif not identity:
            status, failure = "technical_failure", "VERSION_OR_IDENTITY_MISMATCH"
        elif success:
            status, failure = "success", None
        else:
            failure = failure or "NATIVE_GOAL_FALSE"
            technical = {"ENVIRONMENT_ERROR", "ASSET_MISSING", "MODEL_ERROR", "PLANNER_ERROR", "ATTACHMENT_ERROR", "STORAGE_ERROR",
                         "ROBOT_MODEL_MISMATCH", "ENVIRONMENT_VERSION_MISMATCH", "SOURCE_CHANGED_DURING_RUN"}
            status = "technical_failure" if failure in technical else "policy_failure"
        closures = [read(p) for p in case.glob("stages/*/grasp_*/CLOSE_EVIDENCE.json")]
        lifts = [read(p) for p in case.glob("stages/*/grasp_*/LIFT_EVIDENCE.json")]
        contexts = [read(p) for p in case.glob("stages/*/grasp_*/PRE_LIFT_CONTEXT.json")]
        limit = config.get("policy", {}).get("lift_clearance_m", .15) * .5
        cache_hits, uncached_start_failures, plans = 0, 0, 0
        for path in case.glob("plans/*/BRIDGE_RESULT.json"):
            record = read(path) or {}
            if record.get("backend") == "curobo_global":
                plans += 1
                cache_hits += bool(record.get("reused_start_rejection"))
                uncached_start_failures += record.get("reason") == "invalid_start_state" and not record.get("reused_start_rejection")
        last_phase = result.get("failure_stage", "initialization")
        decisions = case / "DECISIONS.jsonl"
        if decisions.is_file():
            lines = decisions.read_text().splitlines()
            if lines:
                try:
                    last_phase = json.loads(lines[-1])["phase"]
                except (json.JSONDecodeError, KeyError):
                    pass
        recording = read(case / "RECORDING.json") or {}
        hz = recording.get("fps", config.get("policy", {}).get("control_freq", 20))
        lock = read(case / "ENVIRONMENT_LOCK.json") or {}
        baseline_ok = bool(baseline.get("status") == "success" and baseline.get("pipeline_success") and baseline.get("native_success_at_end"))
        rows.append(dict(cell, completed=completed, status=status, failure_code=failure,
            reason=explain_failure(failure, result) if failure else "完整成功", identity_valid=identity,
            initial_state_verified=bool(initial.get("matched") and (not result or result.get("initial_state_verified"))),
            success=success, native_budget_success=bool(success and native_budget_success(result)),
            physical_steps=result.get("physical_steps", 0), wall_s=result.get("wall_s", supervisor.get("wall_s")),
            simulation_action_s=result.get("physical_steps", 0) / hz,
            real_closure_completed=bool(closures), ever_passed_lift=any(p and p.get("native_bilateral_grasp") and p.get("object_lift_m", 0.) >= limit for p in lifts),
            attached_before_lift=bool(contexts) and all(c and c.get("attached_before_lift") for c in contexts),
            reused_start_rejections=cache_hits, uncached_start_rejections=uncached_start_failures, global_plan_requests=plans,
            last_recorded_phase=last_phase, video_available=(case / "VIDEO.mp4").is_file(),
            alignment_valid=bool(recording.get("alignment_valid")), instruction=lock.get("episode_meta", {}).get("lang", ""),
            baseline_success=baseline_ok, baseline_native_budget_success=bool(baseline_ok and native_budget_success(baseline)),
            result=result, baseline_result=baseline, supervisor=supervisor))
    counts = Counter(r["status"] for r in rows)
    recovery = read(root / "INITIALIZATION_RECOVERY.json")
    complete = (all(r["completed"] for r in rows) and bool(read(root / "SWEEP_COMPLETE.json"))
                and (recovery is None or bool(recovery.get("complete"))))
    return {"schema_version": "robocasa-pickplace22-summary-1", "complete": complete, "source_digest": matrix["source_digest"],
        "policy_digest": matrix["policy_digest"], "baseline_source_digest": matrix["baseline_source_digest"],
        "planned": 22, "completed": sum(r["completed"] for r in rows), "success": sum(r["success"] for r in rows),
        "native_budget_success": sum(r["native_budget_success"] for r in rows), "baseline_success": sum(r["baseline_success"] for r in rows),
        "baseline_native_budget_success": sum(r["baseline_native_budget_success"] for r in rows),
        "success_rate": sum(r["success"] for r in rows) / 22 if complete else None,
        "status_counts": dict(counts), "failure_counts": dict(Counter(r["failure_code"] for r in rows if r["completed"] and r["failure_code"])),
        "initial_states_verified": sum(r["initial_state_verified"] for r in rows),
        "real_closures": sum(r["real_closure_completed"] for r in rows), "passed_lift": sum(r["ever_passed_lift"] for r in rows),
        "newly_successful": [r["task_id"] for r in rows if r["success"] and not r["baseline_success"]],
        "regressed": [r["task_id"] for r in rows if r["completed"] and r["baseline_success"] and not r["success"]],
        "reused_start_rejections": sum(r["reused_start_rejections"] for r in rows),
        "initialization_recovery": recovery, "matrix": matrix, "rows": rows}
