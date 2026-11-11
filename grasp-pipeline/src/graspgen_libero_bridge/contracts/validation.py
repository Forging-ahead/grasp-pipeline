"""Offline contract validation and reporting. No policy or simulator imports.

JSON Schema checking uses the optional development dependency jsonschema.
Semantic checks use only the standard library. Nothing is wired into the actor.
"""
import json
import math
from pathlib import Path

from graspgen_libero_bridge.common.layout import project_root


class ContractError(ValueError):
    pass


def success_fields(outcome):
    full = bool(outcome["pipeline_complete"] and outcome["native_success"] is True and outcome["steps_executed"] > 0)
    return {"full_pipeline_success": full,
            "candidate_success": full and outcome["instruction_order"] != "violated",
            "sft_eligible": full and outcome["instruction_order"] in ("satisfied", "not_applicable")
            and outcome["trajectory_quality"] == "accepted" and outcome["observation_records_complete"]}


def _pose(value, path):
    if len(value) != 4 or any(len(row) != 4 for row in value):
        raise ContractError(path + ": expected 4x4 transform")
    if any(abs(value[3][j] - [0, 0, 0, 1][j]) > 1e-5 for j in range(4)):
        raise ContractError(path + ": invalid homogeneous last row")
    r = [row[:3] for row in value[:3]]
    if any(abs(sum(r[k][i] * r[k][j] for k in range(3)) - (1 if i == j else 0)) > 1e-5
           for i in range(3) for j in range(3)):
        raise ContractError(path + ": rotation is not orthonormal")
    det = sum(r[0][i] * (r[1][(i+1) % 3] * r[2][(i+2) % 3] - r[1][(i+2) % 3] * r[2][(i+1) % 3]) for i in range(3))
    if abs(det - 1) > 1e-5:
        raise ContractError(path + ": rotation determinant must be +1")


def _walk(value, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(path + ": nonfinite value")
    if isinstance(value, dict):
        for key, child in value.items():
            if "_T_" in key and child is not None:
                _pose(child, path + "." + key)
            _walk(child, path + "." + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _walk(child, path + "[" + str(i) + "]")


def _uses_truth(value):
    if isinstance(value, dict):
        return (isinstance(value.get("types"), list) and "simulator_truth" in value["types"]) or any(_uses_truth(v) for v in value.values())
    return isinstance(value, list) and any(_uses_truth(v) for v in value)


def _semantics(p):
    kind = p["payload_type"]
    if kind == "Observation":
        state = p["robot_state"]
        if len(state["joint_names"]) != len(state["q"]) or (state["qvel"] is not None and len(state["qvel"]) != len(state["q"])):
            raise ContractError("Robot state joint name/value lengths differ")
        if len(state["finger_joint_names"]) != len(state["finger_q"]):
            raise ContractError("Finger joint name/value lengths differ")
        cloud = p["target_cloud"]
        if cloud is not None and cloud["sample_count"] != len(cloud["points"]):
            raise ContractError("Target cloud sample_count differs from points")
        if cloud is not None and cloud["capture_state_index"] is not None and cloud["capture_state_index"] > p["state_index"]:
            raise ContractError("Target cloud capture is in the future")
        if p["observation_mode"] == "sensor" and _uses_truth(p):
            raise ContractError("Sensor observation contains simulator truth")
    elif kind == "RobotSpec":
        names = [j["name"] for j in p["arm_joints"] + p["finger_joints"]]
        if len(names) != len(set(names)):
            raise ContractError("Robot joint names are not unique")
        for joint in p["arm_joints"] + p["finger_joints"]:
            if joint["units"] != ("m" if joint["kind"] == "prismatic" else "rad"):
                raise ContractError("Joint units do not match its kind")
            if joint["lower"] is not None and joint["upper"] is not None and joint["lower"] >= joint["upper"]:
                raise ContractError("Joint limits are inverted")
    elif kind == "TaskSpec":
        goals = {g["goal_id"] for g in p["goals"]}
        if len(goals) != len(p["goals"]):
            raise ContractError("Goal IDs are not unique")
        entities = set(p["entity_ids"]) | set(p["region_ids"])
        if any(arg not in entities for g in p["goals"] for arg in g["arguments"]):
            raise ContractError("Goal refers to an unknown entity or region")
        remaining = list(p["required_order"])
        if any(a not in goals or b not in goals or a == b for a, b in remaining):
            raise ContractError("Ordering constraint has an invalid goal ID")
        while remaining:
            starts = {a for a, _ in remaining} - {b for _, b in remaining}
            if not starts:
                raise ContractError("Ordering constraints contain a cycle")
            remaining = [(a, b) for a, b in remaining if a not in starts]
    elif kind == "SceneSnapshot":
        ids = [g["geometry_id"] for g in p["geometries"]]
        if len(ids) != len(set(ids)):
            raise ContractError("Geometry IDs are not unique")
        for g in p["geometries"]:
            params = g["parameters"]
            required = {"box": ["dims_m"], "sphere": ["radius_m"], "cylinder": ["radius_m", "length_m"], "mesh": ["mesh", "scale"]}[g["shape"]]
            if any(k not in params for k in required):
                raise ContractError("Geometry missing shape parameters: " + g["geometry_id"])
            numbers = [v for key in ["dims_m", "scale"] for v in params.get(key, [])]
            numbers.extend(params[k] for k in ["radius_m", "length_m"] if k in params)
            if any(n <= 0 for n in numbers):
                raise ContractError("Geometry dimensions must be positive")
        for r in p["regions"]:
            if any(a >= b for a, b in zip(r["bounds_min_m"], r["bounds_max_m"])):
                raise ContractError("Region bounds are inverted")
        for joint in p["articulations"]:
            if joint["units"] != ("rad" if joint["kind"] == "hinge" else "m") or joint["limits"][0] >= joint["limits"][1]:
                raise ContractError("Articulation units or limits are invalid")
            if abs(sum(v*v for v in joint["axis_world"]) - 1.) > 1e-5:
                raise ContractError("Articulation axis must be a world-frame unit vector")
            if set(joint["handle_geometry_ids"]) - set(ids):
                raise ContractError("Handle geometry is missing from scene")
    elif kind == "PlanRequest" and p["goal"]["kind"] == "joint":
        if len(p["goal"]["q"]) != len(p["goal"]["joint_names"]):
            raise ContractError("Joint goal names and values differ")
    elif kind == "PlanResult":
        if p["status"] == "planned" and (not p["waypoints"] or p["failure"] is not None):
            raise ContractError("Planned result requires waypoints and no failure")
        if p["status"] != "planned" and (p["waypoints"] or p["failure"] is None):
            raise ContractError("Failed plan must have a failure and no executable waypoints")
    elif kind == "PreviewResult":
        if p["state_restored"] and p["state_index_before"] != p["state_index_after"]:
            raise ContractError("Restored preview changed the real episode revision")
        if p["passed"] and (not p["state_restored"] or p["failure"] is not None):
            raise ContractError("Successful preview must restore state and have no failure")
        if not p["passed"] and p["failure"] is None:
            raise ContractError("Failed preview must preserve a failure record")
    elif kind == "ExecutionReport":
        if p["end_state_index"] - p["start_state_index"] != p["actions_executed"]:
            raise ContractError("Execution action/state indices differ")
        if p["status"] == "reached" and p["failure"] is not None:
            raise ContractError("Reached execution cannot carry a failure")
    elif kind == "TrajectoryRecord":
        if p["state_count"] != p["action_count"] + 1 or p["phase_count"] != p["action_count"]:
            raise ContractError("Trajectory must contain N actions and N+1 states")
        for point in p["keypoints"]:
            if point["action_index"] >= p["action_count"] or point["state_index"] != point["action_index"] + 1:
                raise ContractError("Keypoint must use the measured post-action state")
            if point["label_source"] == "vlm" and point["decision_record"] is None:
                raise ContractError("VLM label requires a real decision record")
    elif kind == "ReferenceBundle":
        for key in ["observation", "robot", "task", "scene", "capabilities", "outcome", "recording"]:
            _semantics(p[key])
        episode = p["observation"]["episode_id"]
        if any(p[k]["episode_id"] != episode for k in ["scene", "outcome", "recording"]):
            raise ContractError("Episode IDs differ across payloads")
        names = [j["name"] for j in p["robot"]["arm_joints"]]
        if names != p["observation"]["robot_state"]["joint_names"]:
            raise ContractError("Robot model and state joint order differ")
        cloud = p["observation"]["target_cloud"]
        if cloud is not None and cloud["target_entity_id"] not in p["task"]["entity_ids"]:
            raise ContractError("Point cloud target is not a task entity")
        if p["observation"]["observation_mode"] == "sensor" and _uses_truth(p):
            raise ContractError("Sensor-mode bundle contains privileged scene truth")


def validate_payload(payload):
    from jsonschema import Draft202012Validator
    schema = json.loads((project_root() / "contracts/pipeline-v1.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda e: str(e.path))
    if errors:
        raise ContractError("Schema mismatch: " + errors[0].message[:500])
    _walk(payload)
    _semantics(payload)


def readiness(bundle):
    validate_payload(bundle)
    blockers = []
    caps = bundle["capabilities"]
    if caps["binding_status"] != "implemented":
        blockers.append("adapter_binding_not_implemented")
    profile = json.loads((project_root() / "contracts/profiles/libero_v10_teacher.json").read_text())
    blockers.extend("missing_capability:" + key for key in profile["required_features"] if not caps["features"][key])
    robot = bundle["robot"]
    if (robot["robot_model"] != profile["robot_model"] or robot["gripper_model"] != profile["gripper_model"]
            or len(robot["arm_joints"]) != profile["arm_joint_count"]):
        blockers.append("robot_or_gripper_profile_mismatch")
    if bundle["observation"]["observation_mode"] != profile["observation_mode"]:
        blockers.append("observation_mode_profile_mismatch")
    if bundle["task"]["goal_logic"] != profile["goal_logic"]:
        blockers.append("unsupported_goal_logic")
    predicates = {g["predicate"].lower() for g in bundle["task"]["goals"]}
    blockers.extend("unsupported_predicate:" + p for p in sorted(predicates - set(profile["supported_predicates"])))
    skills = set()
    if predicates & {"on", "in"}:
        skills.add("push" if "push" in bundle["task"]["language"].lower() else "pick_place")
    if predicates & {"open", "close", "turnon", "turnoff"}:
        skills.add("articulate")
        blockers.extend("missing_capability:" + key for key in profile["additional_features_for_articulation"] if not caps["features"][key])
        if not bundle["scene"]["articulations"]:
            blockers.append("articulation_geometry_missing")
    for skill in sorted(skills):
        if skill not in caps["available_skills"]:
            blockers.append("missing_skill:" + skill)
        key = "additional_features_for_" + skill
        if key in profile:
            blockers.extend("missing_capability:" + feature for feature in profile[key] if not caps["features"][feature])
    cloud = bundle["observation"]["target_cloud"]
    if "pick_place" in skills and (cloud is None or len(cloud["points"]) < profile["minimum_input_points"]):
        blockers.append("target_cloud_missing_or_too_sparse")
    elif "pick_place" in skills and (cloud["capture_state_index"] is None or cloud["capture_state_index"] != bundle["observation"]["state_index"]):
        blockers.append("target_cloud_freshness_unverified")
    if bundle["scene"]["state_index"] != bundle["observation"]["state_index"]:
        blockers.append("scene_and_robot_revision_mismatch")
    required_regions = {g["arguments"][-1] for g in bundle["task"]["goals"] if g["predicate"].lower() in ("on", "in")}
    if required_regions - {r["region_id"] for r in bundle["scene"]["regions"]}:
        blockers.append("goal_region_geometry_missing")
    if bundle["scene"]["gravity_world_m_s2"] is None:
        blockers.append("gravity_not_exported")
    if any(g["shape"] == "mesh" and g["parameters"]["mesh"]["availability"] != "local" for g in bundle["scene"]["geometries"]):
        blockers.append("collision_mesh_assets_unresolved")
    if bundle["scene"]["coverage"] != "complete_simulator_collision_world" and bundle["scene"]["unknown_space_policy"] == "not_applicable":
        blockers.append("unknown_space_handling_missing")
    return {"schema_version": "1.0", "payload_type": "ReadinessReport", "transport_valid": True,
            "execution_ready": not blockers, "blockers": blockers,
            "description": "Checks the declared contract and current teacher profile; does not test reachability or guarantee success."}


def validate_plan_context(request, bundle):
    validate_payload(request)
    validate_payload(bundle)
    observation, scene, robot = bundle["observation"], bundle["scene"], bundle["robot"]
    if (request["episode_id"] != observation["episode_id"] or request["state_index"] != observation["state_index"]
            or request["scene_id"] != scene["scene_id"] or request["robot_id"] != robot["robot_id"]):
        raise ContractError("Plan request uses a different episode, robot, state or scene")
    if scene["state_index"] != observation["state_index"]:
        raise ContractError("Scene and observed robot state are not synchronized")
    if request["goal"]["kind"] == "joint" and request["goal"]["joint_names"] != observation["robot_state"]["joint_names"]:
        raise ContractError("Joint goal ordering differs from the robot")
    world = {g["geometry_id"] for g in scene["geometries"]}
    ids = world | set(robot["collision_geometry_ids"])
    policy = request["contact_policy"]
    if set(policy["optimizer_excluded_geometry_ids"]) - world:
        raise ContractError("Optimizer exclusion references an unknown world geometry")
    if any(a not in ids or b not in ids for a, b in policy["allowed_contact_pairs"]):
        raise ContractError("Contact permission references an unknown geometry")
    for attachment in policy["attachments"]:
        if attachment["entity_id"] not in bundle["task"]["entity_ids"] or set(attachment["geometry_ids"]) - world:
            raise ContractError("Attachment refers to an unknown entity or geometry")
