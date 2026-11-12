"""Independent RoboCasa configuration; no changes to the frozen LIBERO loader."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from graspgen_libero_bridge.common.layout import project_root


PATH_KEYS = {"simulation_python", "robocasa_repo", "grasp_repo", "graspgen_python",
             "graspgen_config", "planner_python", "robosuite_assets"}


def read_json(path):
    return json.loads(Path(path).read_text())


def catalog():
    return read_json(project_root() / "configs/robocasa/tasks.json")["tasks"]


def libero_v10_rules():
    """Read frozen shared rules, rejecting any silently task-specific value."""
    root = project_root()
    rows = read_json(root / "configs/libero/tasks.v10.json")
    fields = ("approach_distances_m", "transfer_heights_m", "transfer_yaw_degrees",
              "plan_timeout_s", "native_descent_fallback", "grasp_control_target_margin_m",
              "planning_joint_margin_rad", "joint_reference_margin_rad",
              "joint_recovery_reference_margin_rad", "joint_recovery_steps",
              "position_integral_gain", "position_integral_bound_m", "waypoint_max_steps")
    rules = {}
    for field in fields:
        values = [row["config"][field] for row in rows]
        if not values or any(value != values[0] for value in values):
            raise ValueError("Not a shared v10 rule: " + field)
        rules[field] = copy.deepcopy(values[0])
    # PickPlace.run uses this default even though the catalog omits the key.
    # Retain the effective bound when mapping native placement regions.
    limits = [row["config"].get("placement_orientation_trials", 4) for row in rows]
    if any(type(value) is not int or value <= 0 or value != limits[0] for value in limits):
        raise ValueError("Not a shared v10 rule: placement_orientation_trials")
    rules["placement_orientation_trials"] = limits[0]
    overrides = read_json(root / "configs/policy/v10_overrides.json")
    for field in ("selection_rounds", "max_candidates_per_round", "native_candidates_per_round",
                  "selection_timeout_s", "minimum_downward_approach", "geometric_placement_search",
                  "surface_release_search", "open_container_release_search", "container_xy_search",
                  "static_container_candidate_search", "place_tracking_extra_steps",
                  "place_tracking_extension_radius_m"):
        rules[field] = copy.deepcopy(overrides[field])
    # The native RoboCasa horizon remains authoritative; v10 max_steps=5000 is
    # deliberately not inherited as an undisclosed episode-budget extension.
    return rules


def load_config(task_id, *, seed=0, split="pretrain", gpu=0, paths_file=None,
                policy_file=None, budget_profile="native"):
    if budget_profile not in {"native", "libero-v10-teacher"}:
        raise ValueError("Unknown episode budget profile")
    if split not in {"pretrain", "target"}:
        raise ValueError("split must be pretrain or target")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(gpu, bool) or int(gpu) != gpu or gpu < 0:
        raise ValueError("gpu must be a nonnegative integer")
    rows = [row for row in catalog() if row["task_id"] == task_id]
    if len(rows) != 1:
        raise ValueError("Unknown RoboCasa task: " + str(task_id))
    row = copy.deepcopy(rows[0])
    if split not in row["available_splits"]:
        raise ValueError("Task has no registered " + split + " split")
    policy = read_json(policy_file or project_root() / "configs/robocasa/policy.json")
    if policy.get("schema_version") != "robocasa-policy-1":
        raise ValueError("Unsupported RoboCasa policy configuration")
    if policy.get("rule_profile") == "libero-v10":
        policy["libero_v10_rules"] = libero_v10_rules()
        policy["plan_timeout_s"] = policy["libero_v10_rules"]["plan_timeout_s"]
    policy["budget_profile"] = budget_profile
    if budget_profile == "libero-v10-teacher":
        policy["teacher_step_limit"] = int(read_json(project_root() / "configs/policy/v10_overrides.json")["max_steps"])
    else:
        policy.pop("teacher_step_limit", None)
    positive = ("camera_size", "control_freq", "episode_wall_seconds_atomic", "episode_wall_seconds_composite",
                "position_tolerance_m", "rotation_tolerance_rad", "pose_gain", "move_timeout_s",
                "joint_soft_limit_tolerance_rad", "joint_limit_margin_rad",
                "settle_seconds", "close_seconds", "open_seconds", "grasp_proposals", "candidate_check_limit",
                "physical_grasp_attempts", "plan_timeout_s", "pregrasp_distance_m", "lift_clearance_m",
                "release_clearance_m", "retreat_distance_m", "minimum_target_points",
                "model_sample_points", "max_contact_penetration_m", "base_max_speed_m_s",
                "base_max_yaw_rate_rad_s", "base_grid_resolution_m", "base_footprint_margin_m",
                "navigation_position_tolerance_m", "navigation_rotation_tolerance_rad")
    for name in positive:
        value = policy.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Policy field must be finite and positive: " + name)
    if policy["robot"] != "PandaOmron" or policy["model_sample_points"] != 2048:
        raise ValueError("This integration requires PandaOmron and the 2048-point Panda checkpoint")
    if policy["grasp_ranking"] not in {"discriminator", "seeded_random"}:
        raise ValueError("Unknown grasp ranking")
    if policy["global_planner"] not in {"curobo", "local_ik"}:
        raise ValueError("Unknown global planner")
    if policy.get("rule_profile") == "libero-v10" and policy["global_planner"] != "curobo":
        raise ValueError("The inherited v10 selector uses cuRobo; a local-only ablation is not implemented for this profile")
    if policy.get("grasp_generator", "graspgen") not in {"graspgen", "geometric"}:
        raise ValueError("Unknown grasp generator")
    for field in ("camera_size", "control_freq", "grasp_proposals", "candidate_check_limit", "physical_grasp_attempts", "model_sample_points", "minimum_target_points"):
        if not isinstance(policy[field], int):
            raise ValueError("Policy count must be an integer: " + field)
    if type(policy.get("plan_validation_retries")) is not int or not 0 <= policy["plan_validation_retries"] <= 3:
        raise ValueError("plan_validation_retries must be an integer in [0, 3]")
    if type(policy.get("max_path_shortcut_checks")) is not int or not 0 <= policy["max_path_shortcut_checks"] <= 32:
        raise ValueError("max_path_shortcut_checks must be an integer in [0, 32]")
    heights = policy.get("transfer_clearance_candidates_m")
    if (not isinstance(heights, list) or not 1 <= len(heights) <= 6
            or any(type(h) not in (int, float) or not math.isfinite(h) or not .01 <= h <= .3 for h in heights)
            or len(set(heights)) != len(heights)):
        raise ValueError("Transfer clearances must be a unique finite list in 0.01–0.30 m")
    if not isinstance(policy["camera_names"], list) or not policy["camera_names"] or len(set(policy["camera_names"])) != len(policy["camera_names"]):
        raise ValueError("Camera names must be a nonempty unique list")
    for field in ("base_motion_enabled", "articulation_enabled", "composite_enabled", "carried_contents_enabled"):
        if not isinstance(policy.get(field, False), bool):
            raise ValueError("Capability flag must be boolean: " + field)
    paths = None
    if paths_file is not None:
        paths = read_json(paths_file)
        if set(paths) != PATH_KEYS:
            raise ValueError("RoboCasa paths require exactly: " + ", ".join(sorted(PATH_KEYS)))
        if any(not isinstance(v, str) or not Path(v).is_absolute() for v in paths.values()):
            raise ValueError("Deployment paths must be absolute strings")
    return {"schema_version": "robocasa-run-1", "benchmark": "RoboCasa365",
            "task": row, "seed": int(seed), "split": split, "gpu": int(gpu),
            "policy": policy, "paths": paths,
            "input_mode": "rgbd_partial_cloud_privileged_teacher"}
