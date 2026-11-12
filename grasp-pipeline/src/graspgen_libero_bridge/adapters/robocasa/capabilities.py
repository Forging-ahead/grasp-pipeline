"""Report experimental skill gates before issuing any physical task action."""


def capabilities(config):
    policy, task = config["policy"], config["task"]
    blockers = []
    if len(task["stages"]) > 1 and not policy["composite_enabled"]:
        blockers.append("composite_enabled")
    for stage in task["stages"]:
        flag = {"articulation": "articulation_enabled", "navigate": "base_motion_enabled"}.get(stage["skill"])
        if flag and not policy[flag]:
            blockers.append(flag)
        if stage.get("carried_contents") and not policy["carried_contents_enabled"]:
            blockers.append("carried_contents_enabled")
    return {"schema_version": "robocasa-capabilities-1", "task_id": task["task_id"],
            "robot": policy["robot"], "arm_global_planner": policy["global_planner"],
            "grasp_generator": policy["grasp_generator"], "input_mode": config["input_mode"],
            "experimental_flags": {k: policy[k] for k in ("articulation_enabled", "base_motion_enabled",
                                                         "composite_enabled", "carried_contents_enabled")},
            "missing_enabled_skills": sorted(set(blockers)),
            "static_task_gate_passed": not blockers,
            "physical_readiness_claimed": False,
            "limitations": ["No automatic base reposition for pick-place", "No whole-body cuRobo planning",
                            "Public nine-port schema binding pending", "No sensor-only or VLM decision policy"]}
