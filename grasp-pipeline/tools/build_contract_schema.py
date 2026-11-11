#!/usr/bin/env python3
"""Build the versioned integration data contract. No runtime-policy imports."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = {}


def ref(name):
    return {"$ref": "#/$defs/" + name}


def arr(items, minimum=0, maximum=None):
    result = {"type": "array", "items": items, "minItems": minimum}
    if maximum is not None:
        result["maxItems"] = maximum
    return result


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


def enum(*values):
    return {"enum": list(values)}


def nullable(value):
    return {"anyOf": [value, {"type": "null"}]}


S = {"type": "string", "minLength": 1}
N = {"type": "number"}
I = {"type": "integer", "minimum": 0}
B = {"type": "boolean"}
V3 = arr(N, 3, 3)
M4 = arr(arr(N, 4, 4), 4, 4)
STRINGS = arr(S)


def payload(name, fields):
    D[name] = obj({"schema_version": {"const": "1.0"}, "payload_type": {"const": name}, **fields})


def main():
    D["Source"] = obj({"types": {**arr(enum("sensor", "robot_state", "configuration", "task_definition", "simulator_truth", "estimation", "memory", "code_policy"), 1), "uniqueItems": True},
                       "evidence": STRINGS, "description": S})
    D["Artifact"] = obj({"uri": S, "media_type": S, "sha256": nullable({"type": "string", "pattern": "^[0-9a-f]{64}$"}),
                         "availability": enum("local", "external", "missing"), "description": S})
    D["Failure"] = obj({"domain": enum("perception", "task", "configuration", "planning", "preflight", "control", "grasp_check", "goal_check", "budget", "integration", "unknown"),
                        "code": S, "phase": nullable(S), "message": S, "details": {"type": "object"}})
    D["RobotState"] = obj({"robot_id": S, "joint_names": {**arr(S, 1), "uniqueItems": True}, "q": arr(N, 1), "qvel": nullable(arr(N, 1)),
                           "finger_joint_names": {**STRINGS, "uniqueItems": True}, "finger_q": arr(N),
                           "world_T_base": M4, "world_T_tcp": M4, "source": ref("Source")})
    D["Camera"] = obj({"camera_id": S, "rgb": nullable(ref("Artifact")), "depth_m": nullable(ref("Artifact")),
                       "instance_mask": nullable(ref("Artifact")), "intrinsic": nullable(arr(V3, 3, 3)),
                       "world_T_camera": nullable(M4), "convention": enum("opencv", "unknown"),
                       "capture_state_index": nullable(I), "unavailable_reason": nullable(S), "source": ref("Source")})
    D["Cloud"] = obj({"target_entity_id": S, "frame": {"const": "world"}, "units": {"const": "m"},
                      "points": arr(V3, 1), "sample_count": {"type": "integer", "minimum": 1}, "partial": B,
                      "capture_state_index": nullable(I), "source": ref("Source")})
    payload("Observation", {"episode_id": S, "observation_id": S, "state_index": I, "sim_time_s": nullable(N),
                            "observation_mode": enum("teacher", "sensor"), "robot_state": ref("RobotState"),
                            "cameras": arr(ref("Camera")), "target_cloud": nullable(ref("Cloud"))})
    D["Joint"] = obj({"name": S, "kind": enum("revolute", "continuous", "prismatic"), "units": enum("rad", "m"),
                      "lower": nullable(N), "upper": nullable(N)})
    payload("RobotSpec", {"robot_id": S, "robot_model": S, "model_ref": ref("Artifact"), "model_format": S,
                          "base_frame": S, "tcp_frame": S, "arm_joints": arr(ref("Joint"), 1),
                          "finger_joints": arr(ref("Joint")), "collision_geometry_ids": STRINGS, "gripper_model": S, "grasp_T_tcp": M4,
                          "action_encoding": S, "action_dimension": {"type": "integer", "minimum": 1},
                          "control_hz": {"type": "number", "exclusiveMinimum": 0}, "source": ref("Source")})
    D["Goal"] = obj({"goal_id": S, "predicate": S, "arguments": arr(S, 1), "source": ref("Source")})
    payload("TaskSpec", {"benchmark": S, "task_id": S, "language": S, "definition_ref": ref("Artifact"),
                         "entity_ids": {**STRINGS, "uniqueItems": True}, "region_ids": {**STRINGS, "uniqueItems": True},
                         "goal_logic": enum("all", "any"), "goals": arr(ref("Goal"), 1),
                         "required_order": arr(arr(S, 2, 2)), "instruction_constraints_complete": B})
    D["Region"] = obj({"region_id": S, "owner_entity_id": S, "owner_T_region": M4,
                       "bounds_min_m": V3, "bounds_max_m": V3, "source": ref("Source")})
    D["Geometry"] = obj({"geometry_id": S, "owner_entity_id": nullable(S), "world_T_geometry": M4,
                         "shape": enum("box", "sphere", "cylinder", "mesh"),
                         "parameters": obj({"dims_m": V3, "radius_m": N, "length_m": N, "mesh": ref("Artifact"), "scale": V3}, []),
                         "representation": S, "source": ref("Source")})
    D["EntityPose"] = obj({"entity_id": S, "world_T_object": M4, "source": ref("Source")})
    D["Articulation"] = obj({"articulation_id": S, "owner_entity_id": S, "joint_name": S, "kind": enum("hinge", "slide"),
                             "units": enum("rad", "m"), "q": N, "limits": arr(N, 2, 2), "axis_world": V3, "anchor_world_m": V3,
                             "handle_geometry_ids": arr(S, 1), "native_goal_rule_ref": ref("Artifact"), "source": ref("Source")})
    payload("SceneSnapshot", {"episode_id": S, "scene_id": S, "state_index": nullable(I), "gravity_world_m_s2": nullable(V3),
                              "coverage": enum("complete_simulator_collision_world", "observed", "mapped", "unknown"),
                              "unknown_space_policy": enum("not_applicable", "reject", "observe_first", "assume_free_explicit"),
                              "geometries": arr(ref("Geometry")), "entity_poses": arr(ref("EntityPose")),
                              "regions": arr(ref("Region")), "articulations": arr(ref("Articulation")), "source": ref("Source")})
    FEATURES = ["target_cloud", "robot_state", "robot_model", "joint_limits", "object_pose", "collision_geometry", "goal_region",
                "local_ik", "contacts", "preview_restore", "tcp_tracking", "gripper_control", "native_success", "articulation_state"]
    payload("Capabilities", {"benchmark": S, "binding_status": enum("reference_mapping_only", "implemented"),
                             "features": obj({key: B for key in FEATURES}), "available_skills": arr(enum("pick_place", "articulate", "push")),
                             "description": S})
    D["GoalPose"] = {"oneOf": [obj({"kind": {"const": "tcp_pose"}, "world_T_tcp": M4}),
                                obj({"kind": {"const": "joint"}, "joint_names": arr(S, 1), "q": arr(N, 1)})]}
    D["ContactPolicy"] = obj({"optimizer_excluded_geometry_ids": STRINGS,
                              "allowed_contact_pairs": arr(arr(S, 2, 2)),
                              "attachments": arr(obj({"entity_id": S, "tcp_T_object": M4, "geometry_ids": arr(S, 1)})), "description": S})
    payload("PlanRequest", {"request_id": S, "episode_id": S, "state_index": I, "scene_id": S, "robot_id": S,
                            "phase": S, "backend": S, "goal": ref("GoalPose"), "contact_policy": ref("ContactPolicy"),
                            "timeout_s": {"type": "number", "exclusiveMinimum": 0}})
    D["Waypoint"] = obj({"q": arr(N, 1), "world_T_tcp": M4})
    payload("PlanResult", {"request_id": S, "plan_id": S, "episode_id": S, "state_index": I, "scene_id": S,
                           "backend": S, "status": enum("planned", "no_path", "invalid_start", "unsupported", "budget_exhausted", "error"),
                           "waypoints": arr(ref("Waypoint")), "wall_seconds": {"type": "number", "minimum": 0}, "failure": nullable(ref("Failure"))})
    payload("ExecuteRequest", {"episode_id": S, "plan_id": S, "expected_state_index": I, "phase": S,
                               "gripper_command": enum("open", "close", "hold"), "max_control_steps": I})
    payload("ExecutionReport", {"episode_id": S, "plan_id": S, "phase": S, "backend": S,
                                "status": enum("reached", "not_reached", "collision", "timeout", "episode_ended", "error"),
                                "start_state_index": I, "end_state_index": I, "actions_executed": I,
                                "final_robot_state": ref("RobotState"), "terminated": B, "truncated": B,
                                "grasp_assessment": enum("not_attempted", "unmeasured", "lift_verified", "lift_failed"),
                                "failure": nullable(ref("Failure"))})
    payload("PreviewResult", {"episode_id": S, "plan_id": S, "state_index_before": I, "state_index_after": I,
                              "passed": B, "state_restored": B, "failure": nullable(ref("Failure"))})
    payload("ReadinessReport", {"transport_valid": B, "execution_ready": B, "blockers": STRINGS,
                                "description": S})
    payload("TaskOutcome", {"episode_id": S, "pipeline_complete": B, "native_success": nullable(B), "steps_executed": I,
                            "instruction_order": enum("not_checked", "satisfied", "violated", "not_applicable"),
                            "trajectory_quality": enum("not_checked", "accepted", "rejected"),
                            "observation_records_complete": B, "failure": nullable(ref("Failure")), "source_record": ref("Artifact")})
    D["Keypoint"] = obj({"action_index": I, "state_index": I, "phase": S, "world_T_tcp": M4,
                         "pose_source": {"const": "measured_post_action"}, "label_source": enum("code_policy", "vlm"),
                         "decision_record": nullable(ref("Artifact"))})
    payload("TrajectoryRecord", {"episode_id": S, "state_count": I, "action_count": I, "phase_count": I,
                                 "states_ref": ref("Artifact"), "trajectory_ref": ref("Artifact"), "video_ref": nullable(ref("Artifact")),
                                 "keypoints": arr(ref("Keypoint")), "raw_observations_complete": B,
                                 "action_label_source": enum("code_policy", "vlm"), "notes": S})
    payload("ReferenceBundle", {"observation": ref("Observation"), "robot": ref("RobotSpec"), "task": ref("TaskSpec"),
                                "scene": ref("SceneSnapshot"), "capabilities": ref("Capabilities"),
                                "outcome": ref("TaskOutcome"), "recording": ref("TrajectoryRecord"), "limitations": arr(S, 1)})
    public = [name for name, definition in D.items() if "payload_type" in definition.get("properties", {})]
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Grasp pipeline integration contract v1",
              "description": "Transport contract and reference data. Not an implemented universal adapter or a runtime policy change.",
              "oneOf": [ref(name) for name in public], "$defs": D}
    folder = ROOT / "contracts"; folder.mkdir(exist_ok=True)
    (folder / "pipeline-v1.schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"payload_types": public, "definitions": len(D)}))


if __name__ == "__main__":
    main()
