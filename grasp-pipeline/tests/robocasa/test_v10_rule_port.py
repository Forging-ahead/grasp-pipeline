"""The port must call mature implementations and preserve real-state isolation."""
from types import SimpleNamespace as NS
import json

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.adapters.robocasa.config import load_config, libero_v10_rules
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
from graspgen_libero_bridge.planning.robocasa_local import LocalModel
from graspgen_libero_bridge.planning.robocasa_planner import RoboCasaPlanner
from graspgen_libero_bridge.policy.evaluate import PickPlace
from graspgen_libero_bridge.policy.robocasa_v10 import V10Selection, transfer_preflight
from graspgen_libero_bridge.runtime.robocasa_replay import replay_candidates
from test_isolated_preview import ModelView


def test_selector_and_static_placement_are_original_methods_not_reimplementations():
    for name in ("select_candidate", "recover_native_descent", "transfer_with_alternatives", "place_in_container_static"):
        assert getattr(V10Selection, name) is getattr(PickPlace, name)
    rules = load_config("PickPlaceCounterToStove")["policy"]["libero_v10_rules"]
    assert rules == libero_v10_rules()
    assert (rules["selection_rounds"], rules["max_candidates_per_round"]) == (6, 30)
    assert rules["approach_distances_m"] == [.10, .05]
    assert "max_steps" not in rules


class NamesView(ModelView):
    def __getattr__(self, name):
        if name == "joint_name2id":
            return lambda value: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, value)
        return super().__getattr__(name)


def preview_environment(tmp_path):
    xml = ['<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>',
           '<geom name="obstacle" type="box" pos="2 2 2" size=".03 .03 .03"/><body name="root">']
    for i, axis in enumerate(["0 0 1", "0 1 0", "1 0 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"]):
        xml.append(f'<body name="link{i}" pos="0 0 .08"><joint name="j{i}" axis="{axis}" range="-2 2" limited="true"/><geom type="sphere" size=".01" mass=".1"/>')
    xml += ['<site name="tcp"/>', '</body>' * 8,
            '<body name="target" pos="1 0 0"><freejoint/><geom type="sphere" size=".02" mass=".1"/></body></worldbody></mujoco>']
    raw = mujoco.MjModel.from_xml_string(''.join(xml)); data = mujoco.MjData(raw)
    mujoco.mj_forward(raw, data)
    view = NamesView(raw)
    env = NS(output=tmp_path, steps=0, env=NS(sim=NS(model=view, data=data), obj_body_id={"obj": view.body_name2id("target")}),
             arm=NS(qpos_index=list(range(7)), qvel_index=list(range(7)), joint_index=list(range(7))),
             robot=NS(robot_model=NS(root_body="root")), config={"seed": 0, "policy": {"max_contact_penetration_m": .002}},
             check_budget=lambda: None)
    env.robot_state = lambda: {"arm_base_body": "root", "eef_site": "tcp", "joint_names": [f"j{i}" for i in range(7)],
                              "finger_joint_names": []}
    local = LocalModel(env); local.target = "obj"
    return env, local, raw, data


def test_original_contact_planner_runs_on_shadow_data_without_a_proxy_veto(tmp_path):
    env, local, raw, live = preview_environment(tmp_path)
    before = live.qpos.copy(); before_time = live.time
    planner = RoboCasaPlanner(env, local)
    planner.check_proxy_path = lambda *a, **k: pytest.fail("v10 native contact paths must not acquire an extra sphere veto")
    goal = local.tcp().copy(); goal[:3, :3] = Rotation.from_euler("z", .03).as_matrix() @ goal[:3, :3]
    reply = planner.contact_plan(goal, label="native_contact")
    assert reply["success"], reply
    assert reply["state_restored"] and reply["isolated_preview"]
    assert reply["planner"] == "mujoco_cartesian_dls"
    np.testing.assert_array_equal(live.qpos, before)
    assert live.time == before_time and env.steps == 0


def test_original_contact_planner_still_rejects_real_geometry_collisions(tmp_path):
    env, local, raw, live = preview_environment(tmp_path)
    raw.geom_pos[0] = local.tcp()[:3, 3]
    mujoco.mj_forward(raw, live)
    before = live.qpos.copy()
    reply = RoboCasaPlanner(env, local).contact_plan(local.tcp(), label="blocked")
    assert not reply["success"] and reply["reason"] == "cartesian_start_collision"
    assert reply["state_restored"]
    np.testing.assert_array_equal(live.qpos, before)


def test_v10_transfer_fallback_keeps_native_collision_failure_as_a_failure():
    calls = []
    planner = NS(global_plan=lambda *a, **k: {"success": False, "reason": "invalid_start_state", "status": "WORLD_COLLISION"},
                 contact_plan=lambda *a, **k: calls.append(k) or {"success": False, "reason": "cartesian_collision"})
    reply = transfer_preflight(planner, "obj", np.eye(4), label="transfer")
    assert len(calls) == 1 and not reply["success"]
    assert reply["global_failure"]["reason"] == "invalid_start_state"
    planner.global_plan = lambda *a, **k: {"success": False, "reason": "invalid_start_state", "status": "JOINT_LIMIT"}
    transfer_preflight(planner, "obj", np.eye(4), label="transfer")
    assert len(calls) == 1


def replay_fixture(tmp_path):
    source, output = tmp_path / "source", tmp_path / "current"
    directory = output / "stages/00_obj/grasp_00"
    old = source / directory.relative_to(output)
    old.mkdir(parents=True); directory.mkdir(parents=True)
    (source / "RESULT.json").write_text(json.dumps({"task_id": "task", "seed": 0, "split": "pretrain", "source_digest": "old"}))
    for case in (source, output):
        (case / "ENVIRONMENT_LOCK.json").write_text(json.dumps({"scene_xml_sha256": "same"}))
        np.savez(case / "INITIAL_STATE.npz", qpos=np.arange(14))
    cloud = np.arange(60, dtype=np.float32).reshape(-1, 3)
    np.savez(old / "CLOUD.npz", points_world=cloud)
    np.savez(old / "CANDIDATES.npz", poses_world_grasp=np.eye(4)[None], scores=[.8])
    env = NS(output=output, config={"candidate_replay_case": str(source), "task": {"task_id": "task"}, "seed": 0, "split": "pretrain"})
    return env, directory, cloud


def test_paired_candidates_require_exact_scene_initial_state_and_cloud(tmp_path):
    env, directory, cloud = replay_fixture(tmp_path)
    metadata = replay_candidates(env, directory, cloud)
    assert metadata["cloud_exact_match"] and metadata["generation_calls"] == 0
    with np.load(directory / "CANDIDATES.npz") as pool:
        np.testing.assert_allclose(pool["scores"], [.8])
    with pytest.raises(EpisodeFailure, match="cloud differs"):
        replay_candidates(env, directory, cloud + .001)
    np.savez(env.output / "INITIAL_STATE.npz", qpos=np.zeros(14))
    with pytest.raises(EpisodeFailure, match="qpos differs"):
        replay_candidates(env, directory, cloud)
