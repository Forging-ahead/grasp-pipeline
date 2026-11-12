import json

from test_experiment_records import tool


def test_student_robot_fields_do_not_include_simulator_object_state(tmp_path):
    exporter = tool("export_keypoints")
    state = {"q_arm": [0.] * 7, "q_fingers": [.04, -.04], "joint_names": ["j"] * 7,
             "full_simulator_qpos": [1., 2., 3.], "object_pose": "privileged", "world_T_base": "not yet audited"}
    (tmp_path / "ROBOT_STATE.json").write_text(json.dumps(state))
    output = exporter.student_robot_state(tmp_path, 0, [])
    assert set(output) == {"q_arm", "q_fingers", "joint_names"}
    assert "object_pose" not in output and "full_simulator_qpos" not in output


def test_decision_uses_measured_robot_state_at_the_same_observation_index(tmp_path):
    exporter = tool("export_keypoints")
    after = {"q_arm": [1.] * 7, "qvel_arm": [0.] * 7, "q_fingers": [.01, -.01]}
    steps = [{"step": 1, "before": {"q_arm": [0.] * 7}, "after": after}]
    assert exporter.student_robot_state(tmp_path, 1, steps) == after


def test_collection_budget_and_replayed_candidates_stay_explicit_in_export_metadata():
    exporter = tool("export_keypoints")
    provenance = exporter.teacher_execution_provenance({
        "budget_profile": "libero-v10-teacher", "native_horizon": 600,
        "runner_step_limit": 5000, "physical_steps": 730, "pipeline_success": True,
        "native_success_at_end": True, "paired_diagnostic": True, "candidate_source": "recorded_identical_pool"})
    assert provenance["runner_step_limit"] == 5000 and provenance["native_horizon"] == 600
    assert not provenance["native_budget_success"]
    assert provenance["paired_diagnostic"] and provenance["candidate_source"] == "recorded_identical_pool"
