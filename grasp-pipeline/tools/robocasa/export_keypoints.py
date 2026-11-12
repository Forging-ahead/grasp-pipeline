#!/usr/bin/env python3
"""Export successful teacher tool/pose candidates aligned to decision-before observations."""
import argparse
import json
from pathlib import Path

KEY_TOOLS = {"execute_grasp", "lift_object", "place_object", "open_gripper", "navigate_base", "articulate_joint"}
ROBOT_INPUT_FIELDS = {"q_arm", "qvel_arm", "q_fingers", "joint_names", "finger_joint_names", "world_T_eef", "sim_time"}


def teacher_execution_provenance(result):
    native_limit = result.get("native_horizon")
    within = (type(native_limit) in (int, float) and native_limit > 0
              and result.get("physical_steps", 0) <= native_limit)
    return {"budget_profile": result.get("budget_profile", "native"),
            "native_horizon": native_limit,
            "runner_step_limit": result.get("runner_step_limit", native_limit),
            "physical_steps": result.get("physical_steps", 0),
            "native_budget_success": bool(within and result.get("pipeline_success") and result.get("native_success_at_end")),
            "paired_diagnostic": bool(result.get("paired_diagnostic")),
            "candidate_source": result.get("candidate_source", "legacy_record_not_explicit"),
            "usage": "teacher trajectory audit metadata; not student input"}


def student_robot_state(case, observation_index, steps):
    if observation_index == 0:
        state = json.loads((case / "ROBOT_STATE.json").read_text())
    else:
        record = steps[observation_index - 1]
        if record["step"] != observation_index:
            raise ValueError("Robot state and decision observation indices differ")
        state = record["after"]
    # Full simulator qpos contains object poses. It is a replay artifact and
    # must never be silently passed to a student as robot proprioception.
    return {k: state[k] for k in ROBOT_INPUT_FIELDS if k in state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.case / "RESULT.json").read_text())
    recording = json.loads((args.case / "RECORDING.json").read_text())
    if (result.get("mode") != "run" or not result.get("pipeline_success") or result.get("native_success_at_reset")
            or not result.get("native_success_at_end") or result.get("physical_steps", 0) <= 0):
        parser.error("Only successful, physically executed native task episodes can supply training candidates")
    if not result.get("source_unchanged_during_run") or not recording.get("alignment_valid"):
        parser.error("Source identity or observation/action alignment failed")
    supervisor_file = args.case / "SUPERVISOR.json"
    if supervisor_file.is_file():
        supervisor = json.loads(supervisor_file.read_text())
        if supervisor.get("timed_out") or supervisor.get("interrupted_signal"):
            parser.error("Interrupted or timed-out episodes cannot supply training candidates")
    if not (args.case / "TRAJECTORY.h5").is_file():
        parser.error("Recorded observations are missing")
    config = json.loads((args.case / "CONFIG.json").read_text())
    environment = json.loads((args.case / "ENVIRONMENT_LOCK.json").read_text())
    steps = [json.loads(line) for line in (args.case / "STEPS.jsonl").read_text().splitlines() if line.strip()]
    decisions = [json.loads(line) for line in (args.case / "DECISIONS.jsonl").read_text().splitlines() if line.strip()]
    candidates = []
    for decision in decisions:
        if decision["tool_name"] not in KEY_TOOLS:
            continue
        index = decision["observation_index"]
        if not 0 <= index < recording["observation_frames"]:
            raise ValueError("Decision observation index is outside the recorded trajectory")
        candidates.append({"schema_version": "robocasa-teacher-keypoint-2",
            "task_id": result["task_id"], "seed": result["seed"], "split": result["split"],
            "source_digest": result["source_digest"], "decision": decision,
            "policy_digest": result["policy_digest"], "task_instruction": environment["episode_meta"]["lang"],
            "teacher_execution": teacher_execution_provenance(result),
            "student_robot_state": student_robot_state(args.case, index, steps),
            "student_rgb_refs": [{"dataset": "observations/" + camera + "_image", "observation_index": index,
                                  "image_convention": "top_down"} for camera in config["policy"]["camera_names"]],
            "student_input_availability_audit": "pending; RGB and listed robot fields only; no object poses or GT masks",
            "trajectory_file": str((args.case / "TRAJECTORY.h5").resolve()),
            "label_source": "code_policy_teacher", "vlm_decision": False,
            "student_observability_audit": "pending", "ready_for_sft": False})
    if not candidates:
        parser.error("No high-level tool/pose decisions found; no training data was generated")
    if args.output.exists():
        parser.error("Refusing to overwrite a candidate export")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in candidates))
    print(json.dumps({"keypoints": len(candidates), "ready_for_sft": False, "next_audit": "student observation sufficiency"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
