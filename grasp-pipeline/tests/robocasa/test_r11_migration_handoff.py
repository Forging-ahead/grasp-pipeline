"""Regression contracts for the actual v10-to-native stage boundaries."""
import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from graspgen_libero_bridge.adapters.robocasa.config import load_config
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
from graspgen_libero_bridge.planning.robocasa_planner import RoboCasaPlanner
from graspgen_libero_bridge.policy.robocasa_skills import lift_with_attachment
from graspgen_libero_bridge.policy.robocasa_v10 import place_v10, transfer_preflight
from graspgen_libero_bridge.runtime.robocasa_initial_state import verify_initial_state


def test_effective_legacy_orientation_default_and_atomic_wall_budget_are_retained():
    config = load_config("PickPlaceCounterToStove", budget_profile="libero-v10-teacher")
    assert config["policy"]["libero_v10_rules"]["placement_orientation_trials"] == 4
    assert config["policy"]["episode_wall_seconds_atomic"] == 1200
    assert config["policy"]["teacher_step_limit"] == 5000


def test_native_region_mapping_bounds_orientations_before_inherited_transfer(monkeypatch, tmp_path):
    from graspgen_libero_bridge.adapters.robocasa import goals
    from graspgen_libero_bridge.adapters.libero import placement_geometry
    poses = [np.eye(4) for _ in range(9)]
    for index, pose in enumerate(poses):
        pose[0, 3] = index
    region = {"name": "native", "frame": np.eye(4), "half_size": np.ones(3), "surface_height_m": 0.}
    monkeypatch.setattr(goals, "v10_placement_regions", lambda *args: [region])
    monkeypatch.setattr(goals, "fits_native_opening", lambda *args: True)
    monkeypatch.setattr(placement_geometry, "orientation_candidates", lambda *args, **kwargs: poses)
    observed = []
    def transfer(*args):
        observed.extend(job.placement_object_options)
        raise RuntimeError("stop at the transfer boundary")
    job = NS(config={"placement_orientation_trials": 4}, transfer_with_alternatives=transfer)
    carried = {"v10_job": job, "tcp_T_object": np.eye(4), "object_points": np.zeros((4, 3)), "initial_object_pose": np.eye(4)}
    with pytest.raises(RuntimeError, match="transfer boundary"):
        place_v10(NS(check_budget=lambda: None, steps=0), None, None, {"destination": "fixture"}, carried, tmp_path)
    assert [pose[0, 3] for pose in observed] == [0, 1, 2, 3]


@pytest.mark.parametrize("native_success", [True, False])
def test_lift_registers_attachment_before_any_plan_and_limits_support_permissions(monkeypatch, tmp_path, native_success):
    from graspgen_libero_bridge.policy import robocasa_skills
    monkeypatch.setattr(robocasa_skills, "object_contacts", lambda *args: {1, 2})
    events = []
    local = NS(allowed_support_geoms=set())
    env = NS(config={"policy": {"lift_clearance_m": .15}}, robot_state=lambda: {"world_T_eef": np.eye(4)})
    def attach(target, contents):
        events.append("attached")
    def contact(goal, **kwargs):
        assert events[0] == "attached"
        assert local.allowed_support_geoms == {0, 1, 2}
        assert goal[2, 3] == .15
        events.append("native_plan")
        return {"success": native_success, "path_file": "lift.npz", "state_restored": True, "reason": "cartesian_ik_collision_blocked"}
    def global_plan(target, goal, **kwargs):
        assert events[0] == "attached"
        assert kwargs["ignore_obstacles"] == ["floor", "rim"]
        assert kwargs["allow_target_contact"]
        events.append("global_plan")
        return {"success": True, "path_file": "fallback.npz"}
    planner = NS(attach=attach, contact_plan=contact, global_plan=global_plan,
        scene=lambda target: {"obstacles": [{"name": "floor", "geom_id": 0}, {"name": "rim", "geom_id": 1}, {"name": "cabinet", "geom_id": 9}]})
    executor = NS(ensure_joint_margin=lambda *a, **k: None, follow_path=lambda *a, **k: events.append("executed"),
                  hold=lambda *a, **k: events.append("hold"))
    lift_with_attachment(env, executor, local, NS(decision=lambda *a: None), planner,
                         {"target": "obj"}, tmp_path, {0})
    assert events == (["attached", "native_plan", "executed", "hold"] if native_success else
                      ["attached", "native_plan", "global_plan", "executed", "hold"])
    audit = json.loads((tmp_path / "PRE_LIFT_CONTEXT.json").read_text())
    assert audit["attached_before_lift"] and audit["native_robot_world_contacts_still_checked"]


def planner_fixture(tmp_path, reason="invalid_start_state"):
    env = NS(output=tmp_path, clock=lambda: 0., remaining_seconds=lambda: 1200., check_budget=lambda: None,
             config={"policy": {"global_planner": "curobo", "plan_timeout_s": 12., "plan_validation_retries": 0}})
    planner = RoboCasaPlanner(env, NS())
    planner.ensure_configuration = lambda target: None
    scene = {"q_arm": [0.] * 7, "q_fingers": [.04, -.04], "obstacles": [{"name": "floor", "dims": [2., 2., .1]}]}
    planner.scene = lambda target: copy.deepcopy(scene)
    requests = []
    def request(payload):
        requests.append(payload)
        return {"success": False, "reason": reason, "status": "WORLD_COLLISION"}
    planner._request = request
    return planner, scene, requests


def test_unchanged_invalid_start_is_reused_but_each_goal_still_gets_native_fallback(tmp_path):
    planner, scene, requests = planner_fixture(tmp_path)
    fallbacks = []
    planner.contact_plan = lambda *a, **k: fallbacks.append(k) or {"success": False, "reason": "cartesian_collision"}
    for index in range(2):
        goal = np.eye(4); goal[0, 3] = index
        reply = transfer_preflight(planner, "obj", goal, label="transfer")
        assert not reply["success"]
    assert len(requests) == 1 and len(fallbacks) == 2
    assert reply["global_failure"]["reused_start_rejection"]


def test_start_rejection_is_not_reused_after_state_world_or_contact_policy_changes(tmp_path):
    planner, scene, requests = planner_fixture(tmp_path)
    def plan(**kwargs):
        return planner.global_plan("obj", np.eye(4), label="transfer", **kwargs)
    plan(); scene["q_arm"][0] = .01; plan()
    scene["obstacles"][0]["dims"][0] = 3.; plan()
    scene["q_fingers"][0] = .02; plan()
    plan(allow_target_contact=True)
    plan(ignore_obstacles=["floor"])
    assert len(requests) == 6
    assert not any(request.get("reused_start_rejection") for request in requests)


def test_goal_dependent_ik_failure_is_never_cached_as_a_start_failure(tmp_path):
    planner, scene, requests = planner_fixture(tmp_path, reason="curobo_plan_failed")
    planner.global_plan("obj", np.eye(4), label="a")
    planner.global_plan("obj", np.eye(4), label="b")
    assert len(requests) == 2 and not planner.invalid_start_cache


@pytest.mark.parametrize("changed_field", [None, "ctrl", "qpos"])
def test_exact_initial_state_check_rejects_a_different_rerun_before_execution(tmp_path, changed_field):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    identity = {"task_id": "Task", "seed": 0, "split": "pretrain"}
    (old / "RESULT.json").write_text(json.dumps(identity))
    lock = {"scene_xml_sha256": "scene", "success_source_sha256": "predicate", "action_spec": {"dimension": 12}}
    state = {"qpos": np.zeros(7), "qvel": np.zeros(7), "ctrl": np.zeros(7), "sim_time": np.array(0.)}
    for directory in (old, new):
        (directory / "ENVIRONMENT_LOCK.json").write_text(json.dumps(lock))
        np.savez(directory / "INITIAL_STATE.npz", **state)
    if changed_field:
        state[changed_field] = state[changed_field].copy(); state[changed_field][0] = .001
        np.savez(new / "INITIAL_STATE.npz", **state)
    config = {"task": {"task_id": "Task"}, "seed": 0, "split": "pretrain"}
    if changed_field:
        with pytest.raises(EpisodeFailure, match="initial state"):
            verify_initial_state(config, new, old)
    else:
        assert verify_initial_state(config, new, old)["matched"]
