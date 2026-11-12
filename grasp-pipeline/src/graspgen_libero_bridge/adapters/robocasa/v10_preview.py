"""Bind the unchanged v10 contact/selection code to isolated RoboCasa data.

The facade deliberately exposes no live step/reset or controller mutation.
The seven Panda arm joints and two fingers are read from the native model;
the mobile base/torso stay at their measured positions in the shadow MjData.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
from graspgen_libero_bridge.adapters.robocasa.native import matrix
from graspgen_libero_bridge.planning.cartesian import _snapshot, _restore


class DataView:
    def __init__(self, data):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, key):
        return getattr(self._data, {"body_xpos": "xpos", "body_xmat": "xmat"}.get(key, key))


class PreviewSim:
    def __init__(self, local):
        self.local = local
        self.model = local.environment.env.sim.model
        self.data = DataView(local.data)
        self.restoring = False

    def forward(self):
        if not self.restoring:
            self.local.environment.check_budget()
        self.restoring = False
        self.local.mj.mj_forward(self.local.model, self.local.data)

    def get_state(self):
        return {"time": float(self.local.data.time), "qpos": self.local.data.qpos.copy(),
                "qvel": self.local.data.qvel.copy(), "act": self.local.data.act.copy()}

    def set_state(self, state):
        self.local.data.time = state["time"]
        for key in ("qpos", "qvel", "act"):
            getattr(self.local.data, key)[...] = state[key]
        # Snapshot restoration must remain possible after a budget exception.
        self.restoring = True


class V10PreviewAdapter:
    def __init__(self, environment, local, target):
        self.environment, self.local = environment, local
        state = environment.robot_state()
        model = environment.env.sim.model
        fingers = [int(model.jnt_qposadr[model.joint_name2id(n)]) for n in state["finger_joint_names"]]
        self.robot = SimpleNamespace(_ref_joint_pos_indexes=local.qpos_ids,
            _ref_joint_vel_indexes=local.qvel_ids, _ref_gripper_joint_pos_indexes=fingers,
            eef_site_id=local.site, robot_model=SimpleNamespace(joints=state["joint_names"]))
        self.env = SimpleNamespace(sim=PreviewSim(local))
        self.task_env = SimpleNamespace(robots=[self.robot], obj_body_id=environment.env.obj_body_id)
        self.target_object = target

    def target_pose(self, name):
        body = self.task_env.obj_body_id[name]
        return matrix(self.local.data.xpos[body], self.local.data.xmat[body])

    @property
    def step_count(self):
        return self.environment.steps

    def state(self):
        return {"q_arm": self.local.data.qpos[self.local.qpos_ids].copy(),
                "world_T_eef": self.local.tcp()}


class V10PreviewMonitor:
    def __init__(self, adapter):
        self.adapter = adapter
        self.local = adapter.local
        self.target = adapter.target_object
        self.target_bodies = body_subtree_ids(self.local.model, adapter.task_env.obj_body_id[self.target])

    @property
    def held(self):
        return bool(self.local.held_relations)

    @property
    def allowed_support_geoms(self):
        return self.local.allowed_support_geoms

    def check(self, allow_finger_target=False):
        original = self.local.allow_finger_target
        try:
            self.local.allow_finger_target = bool(allow_finger_target)
            return self.local.contacts()
        finally:
            self.local.allow_finger_target = original

    def validate_path(self, path, allow_finger_target=False, max_samples=1000):
        # A selected descent starts at a *future* pregrasp. Validate all its
        # samples here; the executor checks the actual start before execution.
        sim = self.adapter.env.sim
        saved = _snapshot(sim)
        original = self.local.allow_finger_target
        try:
            self.local.allow_finger_target = bool(allow_finger_target)
            result = self.local.validate_path(path, expected_start=np.asarray(path)[0])
            return {"passed": result["valid"], "failures": result.get("contacts", []),
                    "native_validation": result, "samples": result.get("samples", 0)}
        finally:
            self.local.allow_finger_target = original
            _restore(sim, saved)
