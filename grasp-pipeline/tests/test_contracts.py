"""Offline boundary contracts: no simulator, renderer, model or GPU is used."""
import copy
import json
from pathlib import Path
import unittest

from graspgen_libero_bridge.common.layout import project_root
from graspgen_libero_bridge.contracts.validation import ContractError, readiness, success_fields, validate_payload, validate_plan_context


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = project_root() / "contracts/examples"
        cls.reference = json.loads((root / "libero_v10_reference.json").read_text())
        cls.outcomes = json.loads((root / "libero_v10_outcomes.json").read_text())

    def setUp(self):
        self.bundle = copy.deepcopy(self.reference)

    def request(self):
        b = self.bundle
        return dict(schema_version="1.0", payload_type="PlanRequest", request_id="test-plan", episode_id=b["observation"]["episode_id"],
                    state_index=b["observation"]["state_index"], scene_id=b["scene"]["scene_id"], robot_id=b["robot"]["robot_id"],
                    phase="transport", backend="curobo", goal=dict(kind="tcp_pose", world_T_tcp=b["observation"]["robot_state"]["world_T_tcp"]),
                    contact_policy=dict(optimizer_excluded_geometry_ids=[], allowed_contact_pairs=[], attachments=[], description="No contact permission in this test"), timeout_s=12)

    def test_real_libero_reference_and_components_validate(self):
        validate_payload(self.bundle)
        for name in ["observation", "robot", "task", "scene", "capabilities", "outcome", "recording"]:
            validate_payload(self.bundle[name])

    def test_all130_outcomes_keep_all_three_success_definitions(self):
        for outcome in self.outcomes:
            validate_payload(outcome)
        self.assertEqual(len(self.outcomes), 130)
        self.assertEqual(sum(o["native_success"] is True for o in self.outcomes), 88)
        self.assertEqual(sum(success_fields(o)["full_pipeline_success"] for o in self.outcomes), 86)
        self.assertEqual(sum(success_fields(o)["candidate_success"] for o in self.outcomes), 85)
        self.assertFalse(any(success_fields(o)["sft_eligible"] for o in self.outcomes))

    def test_cloud_must_be_world_meters(self):
        self.bundle["observation"]["target_cloud"]["units"] = "cm"
        with self.assertRaises(ContractError): validate_payload(self.bundle)

    def test_cloud_count_and_identity_are_checked(self):
        self.bundle["observation"]["target_cloud"]["target_entity_id"] = "not-in-this-task"
        with self.assertRaises(ContractError): validate_payload(self.bundle)
        self.bundle = copy.deepcopy(self.reference)
        self.bundle["observation"]["target_cloud"]["sample_count"] += 1
        with self.assertRaises(ContractError): validate_payload(self.bundle)

    def test_pose_and_nonfinite_values_are_rejected(self):
        self.bundle["robot"]["grasp_T_tcp"][0][0] = 2.
        with self.assertRaises(ContractError): validate_payload(self.bundle)
        self.bundle = copy.deepcopy(self.reference)
        self.bundle["observation"]["robot_state"]["q"][0] = float("nan")
        with self.assertRaises(ContractError): validate_payload(self.bundle)

    def test_model_and_measured_joint_order_must_agree(self):
        self.bundle["observation"]["robot_state"]["joint_names"].reverse()
        with self.assertRaises(ContractError): validate_payload(self.bundle)

    def test_sensor_mode_cannot_hide_truth(self):
        self.bundle["observation"]["observation_mode"] = "sensor"
        with self.assertRaises(ContractError): validate_payload(self.bundle)

    def test_unknown_goal_argument_and_order_cycle_are_rejected(self):
        task = self.bundle["task"]
        task["goals"][0]["arguments"][0] = "not-an-entity"
        with self.assertRaises(ContractError): validate_payload(task)
        task = copy.deepcopy(self.reference["task"])
        task["goals"].append({**task["goals"][0], "goal_id": "goal_1"})
        task["required_order"] = [["goal_0", "goal_1"], ["goal_1", "goal_0"]]
        with self.assertRaises(ContractError): validate_payload(task)

    def test_scene_requires_shape_dimensions(self):
        geometry = next(g for g in self.bundle["scene"]["geometries"] if g["shape"] == "box")
        del geometry["parameters"]["dims_m"]
        with self.assertRaises(ContractError): validate_payload(self.bundle)

    def test_reference_is_not_claimed_to_be_execution_ready(self):
        report = readiness(self.bundle)
        validate_payload(report)
        self.assertFalse(report["execution_ready"])
        for blocker in ["adapter_binding_not_implemented", "target_cloud_freshness_unverified", "goal_region_geometry_missing", "gravity_not_exported"]:
            self.assertIn(blocker, report["blockers"])

    def test_mesh_assets_are_checked_when_scene_contains_a_mesh(self):
        geometry = self.bundle["scene"]["geometries"][0]
        geometry["shape"] = "mesh"
        geometry["parameters"] = dict(mesh=dict(uri="fixture-only://missing.obj", media_type="model/obj", sha256=None,
                                                availability="external", description="Synthetic missing-mesh test, not real LIBERO data"), scale=[1., 1., 1.])
        self.assertIn("collision_mesh_assets_unresolved", readiness(self.bundle)["blockers"])

    def test_execution_reached_is_not_a_claim_of_grasp_success(self):
        request = dict(schema_version="1.0", payload_type="ExecuteRequest", episode_id="synthetic", plan_id="p", expected_state_index=0,
                       phase="approach", gripper_command="hold", max_control_steps=1)
        validate_payload(request)
        report = dict(schema_version="1.0", payload_type="ExecutionReport", episode_id="synthetic", plan_id="p", phase="approach", backend="test",
                      status="reached", start_state_index=0, end_state_index=1, actions_executed=1,
                      final_robot_state=self.bundle["observation"]["robot_state"], terminated=False, truncated=False,
                      grasp_assessment="unmeasured", failure=None)
        validate_payload(report)
        self.assertEqual(report["grasp_assessment"], "unmeasured")
        report["actions_executed"] = 2
        with self.assertRaises(ContractError): validate_payload(report)

    def test_missing_capability_is_reported_before_planning(self):
        self.bundle["capabilities"]["features"]["contacts"] = False
        self.assertIn("missing_capability:contacts", readiness(self.bundle)["blockers"])

    def test_goal_logic_outside_current_policy_is_reported(self):
        self.bundle["task"]["goal_logic"] = "any"
        self.assertIn("unsupported_goal_logic", readiness(self.bundle)["blockers"])

    def test_articulation_does_not_require_graspgen_point_cloud(self):
        self.bundle["task"]["goals"][0].update(predicate="open", arguments=["basket_1"])
        self.bundle["observation"]["target_cloud"] = None
        self.bundle["capabilities"]["features"]["target_cloud"] = False
        blockers = readiness(self.bundle)["blockers"]
        self.assertIn("articulation_geometry_missing", blockers)
        self.assertNotIn("target_cloud_missing_or_too_sparse", blockers)
        self.assertNotIn("missing_capability:target_cloud", blockers)

    def test_unsupported_native_predicate_is_not_silently_dropped(self):
        self.bundle["task"]["goals"][0]["predicate"] = "pour"
        self.assertIn("unsupported_predicate:pour", readiness(self.bundle)["blockers"])

    def test_plan_context_checks_state_and_scene_identity(self):
        request = self.request()
        validate_plan_context(request, self.bundle)
        request["state_index"] += 1
        with self.assertRaises(ContractError): validate_plan_context(request, self.bundle)

    def test_contact_permission_requires_real_geometry_ids(self):
        request = self.request()
        request["contact_policy"]["allowed_contact_pairs"] = [["made-up-finger", "made-up-obstacle"]]
        with self.assertRaises(ContractError): validate_plan_context(request, self.bundle)

    def test_failed_plan_cannot_return_an_executable_path(self):
        request = self.request()
        result = dict(schema_version="1.0", payload_type="PlanResult", request_id=request["request_id"], plan_id="plan0", episode_id=request["episode_id"],
                      state_index=0, scene_id=request["scene_id"], backend="curobo", status="no_path", waypoints=[], wall_seconds=1.,
                      failure=dict(domain="planning", code="no_path", phase="transport", message="No path within the requested budget", details={}))
        validate_payload(result)
        result["waypoints"] = [dict(q=self.bundle["observation"]["robot_state"]["q"], world_T_tcp=request["goal"]["world_T_tcp"])]
        with self.assertRaises(ContractError): validate_payload(result)

    def test_preview_restore_failure_can_be_recorded_but_not_passed(self):
        preview = dict(schema_version="1.0", payload_type="PreviewResult", episode_id="ep", plan_id="p", state_index_before=0, state_index_after=1,
                       passed=False, state_restored=False, failure=dict(domain="integration", code="preview_restore_failed", phase="preflight", message="State restoration failed; abort episode", details={}))
        validate_payload(preview)
        preview["passed"] = True
        with self.assertRaises(ContractError): validate_payload(preview)

    def test_native_true_and_pipeline_abort_are_separate(self):
        outcome = next(o for o in self.outcomes if o["native_success"] and not o["pipeline_complete"])
        validate_payload(outcome)
        self.assertFalse(success_fields(outcome)["full_pipeline_success"])

    def test_trajectory_and_post_action_keypoint_alignment(self):
        record = self.bundle["recording"]
        record["state_count"] -= 1
        with self.assertRaises(ContractError): validate_payload(record)
        record = copy.deepcopy(self.reference["recording"])
        record["keypoints"][0]["state_index"] = record["keypoints"][0]["action_index"]
        with self.assertRaises(ContractError): validate_payload(record)

    def test_vlm_label_requires_actual_decision_evidence(self):
        self.bundle["recording"]["keypoints"][0]["label_source"] = "vlm"
        with self.assertRaises(ContractError): validate_payload(self.bundle)


if __name__ == "__main__":
    unittest.main()
