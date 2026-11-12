"""Real MuJoCo kinematics and contacts, without kitchen assets or a GPU."""
from types import SimpleNamespace as NS

import mujoco
import numpy as np

from graspgen_libero_bridge.planning.robocasa_local import LocalModel
from graspgen_libero_bridge.adapters.robocasa.native import matrix


class ModelView:
    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        if name == "body_name2id":
            return lambda value: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, value)
        if name == "body_id2name":
            return lambda value: mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, value)
        if name == "site_name2id":
            return lambda value: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, value)
        return getattr(self._model, name)


def model_fixture():
    pieces = ['<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody><geom name="obstacle" type="box" pos="2 2 2" size=".03 .03 .03"/><body name="root">']
    axes = ["0 0 1", "0 1 0", "1 0 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"]
    for i, axis in enumerate(axes):
        pieces.append(f'<body name="link{i}" pos="0 0 .08"><joint name="j{i}" axis="{axis}" range="-2 2" limited="true"/><geom type="sphere" size=".01" mass=".1"/>')
    pieces.append('<site name="tcp" pos="0 0 0"/>')
    pieces.extend(['</body>'] * 7)
    pieces.append('</body></worldbody></mujoco>')
    raw = mujoco.MjModel.from_xml_string(''.join(pieces))
    data = mujoco.MjData(raw)
    mujoco.mj_forward(raw, data)
    view = ModelView(raw)
    env = NS(env=NS(sim=NS(model=view, data=data), obj_body_id={}),
             arm=NS(qpos_index=list(range(7)), qvel_index=list(range(7)), joint_index=list(range(7))),
             robot=NS(robot_model=NS(root_body="root")),
             config={"seed": 0, "policy": {"max_contact_penetration_m": .001}},
             check_budget=lambda: None)
    env.robot_state = lambda: {"arm_base_body": "root", "eef_site": "tcp"}
    return env, raw, data


def test_ik_uses_isolated_data_and_does_not_change_real_state():
    env, raw, live = model_fixture()
    local = LocalModel(env)
    original = live.qpos.copy()
    local.set_q([.06, -.07, .04, .03, -.03, .03, .02])
    goal = local.tcp().copy()
    result = local.solve(goal)
    assert result["success"], result
    np.testing.assert_array_equal(live.qpos, original)
    assert live.time == 0.


def test_calibration_twenty_samples_cannot_advance_live_physics():
    env, _, live = model_fixture()
    local = LocalModel(env)
    original = live.qpos.copy()
    samples = local.calibration_samples()
    assert len(samples) == 20
    assert len({tuple(np.round(s["q_arm"], 6)) for s in samples}) == 20
    np.testing.assert_array_equal(live.qpos, original)
    assert live.time == 0.


def test_real_mujoco_contact_rejects_collision_without_moving_robot():
    env, raw, live = model_fixture()
    local = LocalModel(env)
    geom = mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
    raw.geom_pos[geom] = local.tcp()[:3, 3]
    original = live.qpos.copy()
    result = local.validate_path([original])
    assert not result["valid"] and result["reason"] == "COLLISION_REJECTED"
    np.testing.assert_array_equal(live.qpos, original)


def test_path_from_a_different_robot_state_is_rejected():
    env, _, live = model_fixture()
    local = LocalModel(env)
    q = live.qpos.copy(); q[0] += .2
    result = local.validate_path([q])
    assert result == {"valid": False, "reason": "PATH_START_MISMATCH"}


def test_small_measured_soft_limit_excursion_can_only_recover_inward():
    env, raw, live = model_fixture()
    live.qpos[6] = 2.0002
    mujoco.mj_forward(raw, live)
    local = LocalModel(env)
    start = live.qpos.copy()
    middle = start.copy(); middle[6] = 2.0001
    end = start.copy(); end[6] = 1.99
    result = local.validate_path([start, middle, end])
    assert result["valid"] and result["bounded_inward_recovery"]
    np.testing.assert_array_equal(live.qpos, start)
    worsening = start.copy(); worsening[6] += .0001
    assert local.validate_path([start, worsening, end])["reason"] == "JOINT_LIMIT"
    assert local.validate_path([start, middle])["reason"] == "JOINT_LIMIT"


def test_large_measured_joint_violation_is_never_clipped_into_a_valid_start():
    env, raw, live = model_fixture()
    live.qpos[6] = 2.01; mujoco.mj_forward(raw, live)
    local = LocalModel(env); start = live.qpos.copy(); end = start.copy(); end[6] = 1.99
    assert local.validate_path([start, end])["reason"] == "JOINT_LIMIT"
    assert local.solve(local.tcp())["reason"] == "START_INVALID"
    np.testing.assert_array_equal(live.qpos, start)
