"""Isolated MuJoCo IK and collision preview. Never edits the live MjData."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.adapters.libero.env_adapter import body_subtree_ids
from graspgen_libero_bridge.adapters.robocasa.native import matrix


class LocalModel:
    def __init__(self, environment):
        import mujoco
        self.mj = mujoco
        self.environment = environment
        self.model = getattr(environment.env.sim.model, "_model", environment.env.sim.model)
        self.data = mujoco.MjData(self.model)
        self.qpos_ids = np.asarray(environment.arm.qpos_index, dtype=int)
        self.qvel_ids = np.asarray(environment.arm.qvel_index, dtype=int)
        self.joint_ids = np.asarray(environment.arm.joint_index, dtype=int)
        self.limits = np.asarray(self.model.jnt_range[self.joint_ids], float)
        self.soft_limit_tolerance = float(environment.config["policy"].get("joint_soft_limit_tolerance_rad", .001))
        self.joint_margin = float(environment.config["policy"].get("joint_limit_margin_rad", .01))
        state = environment.robot_state()
        self.site = int(environment.env.sim.model.site_name2id(state["eef_site"]))
        root = environment.env.sim.model.body_name2id(environment.robot.robot_model.root_body)
        self.robot_bodies = body_subtree_ids(self.model, root)
        arm_root = environment.env.sim.model.body_name2id(state["arm_base_body"])
        self.arm_bodies = body_subtree_ids(self.model, arm_root)
        self.allowed_finger_bodies = {i for i in self.arm_bodies
                if "finger" in (environment.env.sim.model.body_id2name(i) or "")}
        self.held_relations = {}
        self.target = None
        self.allow_finger_target = False
        self.allowed_support_geoms = set()
        self.allowed_target_geoms = None
        self.articulation_preview = {}
        self.sync()

    def sync(self):
        live = getattr(self.environment.env.sim.data, "_data", self.environment.env.sim.data)
        for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "userdata", "eq_active"):
            if hasattr(live, name) and hasattr(self.data, name):
                getattr(self.data, name)[...] = getattr(live, name)
        self.data.time = live.time
        self.mj.mj_forward(self.model, self.data)

    def tcp(self):
        return matrix(self.data.site_xpos[self.site], self.data.site_xmat[self.site])

    def set_q(self, q):
        self.data.qpos[self.qpos_ids] = q
        for joint_id, value in self.articulation_preview.items():
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = value
        self.mj.mj_forward(self.model, self.data)
        for name, relation in self.held_relations.items():
            body = self.environment.env.obj_body_id[name]
            joint = int(self.model.body_jntadr[body])
            if joint < 0 or int(self.model.jnt_type[joint]) != 0:
                raise ValueError("Held object must have a free joint")
            address = int(self.model.jnt_qposadr[joint])
            goal = self.tcp() @ relation
            quat = Rotation.from_matrix(goal[:3, :3]).as_quat()[[3, 0, 1, 2]]
            self.data.qpos[address:address + 7] = np.r_[goal[:3, 3], quat]
        self.mj.mj_forward(self.model, self.data)

    def attach(self, names):
        self.sync()
        inverse = np.linalg.inv(self.tcp())
        self.held_relations = {name: inverse @ self.environment.object_pose(name) for name in names}

    def detach(self):
        self.held_relations.clear()

    def contacts(self, data=None):
        data = self.data if data is None else data
        target_bodies = set()
        if self.target is not None:
            target_body = self.environment.entity_body(self.target) if hasattr(self.environment, "entity_body") else self.environment.env.obj_body_id.get(self.target)
            if target_body is not None:
                target_bodies = body_subtree_ids(self.model, target_body)
        held_bodies = set()
        for name in self.held_relations:
            held_bodies |= body_subtree_ids(self.model, self.environment.env.obj_body_id[name])
        errors = []
        tolerance = float(self.environment.config["policy"]["max_contact_penetration_m"])
        for contact in data.contact[:data.ncon]:
            if contact.dist >= -tolerance:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if r1 and r2:
                w1, w2 = int(self.model.body_weldid[b1]), int(self.model.body_weldid[b2])
                p1 = int(self.model.body_weldid[self.model.body_parentid[w1]])
                p2 = int(self.model.body_weldid[self.model.body_parentid[w2]])
                if w1 == w2 or p1 == w2 or p2 == w1 or (b1 in self.allowed_finger_bodies and b2 in self.allowed_finger_bodies):
                    continue
            elif r1 != r2:
                rb, ob, og = (b1, b2, g2) if r1 else (b2, b1, g1)
                # Wheel/support contact with the floor is intrinsic to a mobile robot.
                if rb not in self.arm_bodies and int(self.model.geom_type[og]) == 0:
                    continue
                permitted_target = og in self.allowed_target_geoms if self.allowed_target_geoms is not None else ob in target_bodies
                if self.allow_finger_target and rb in self.allowed_finger_bodies and permitted_target:
                    continue
            elif (b1 in held_bodies) != (b2 in held_bodies):
                other = g2 if b1 in held_bodies else g1
                if other in self.allowed_support_geoms:
                    continue
            else:
                continue
            errors.append({"geom1": g1, "geom2": g2, "body1": b1, "body2": b2,
                           "penetration_m": float(-contact.dist)})
        return errors

    def check(self):
        live = getattr(self.environment.env.sim.data, "_data", self.environment.env.sim.data)
        return self.contacts(live)

    def solve(self, goal, start=None, *, max_iterations=150):
        self.sync()
        q = np.asarray(self.data.qpos[self.qpos_ids] if start is None else start, float).copy()
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("IK start must contain seven measured joint values")
        if np.any(q < self.limits[:, 0] - self.soft_limit_tolerance) or np.any(q > self.limits[:, 1] + self.soft_limit_tolerance):
            return {"success": False, "reason": "START_INVALID", "q": q.tolist()}
        # This is an optimization seed, not a replacement for measured state.
        # The returned path still starts at the unmodified measured joints.
        q = np.clip(q, self.limits[:, 0] + self.joint_margin, self.limits[:, 1] - self.joint_margin)
        goal = np.asarray(goal, float)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for iteration in range(max_iterations):
            self.environment.check_budget()
            self.set_q(q)
            pose = self.tcp()
            error = np.r_[goal[:3, 3] - pose[:3, 3],
                          Rotation.from_matrix(goal[:3, :3] @ pose[:3, :3].T).as_rotvec()]
            if np.linalg.norm(error[:3]) < .002 and np.linalg.norm(error[3:]) < .02:
                contacts = self.contacts()
                return {"success": not bool(contacts), "q": q.tolist(),
                        "reason": None if not contacts else "COLLISION_REJECTED",
                        "contacts": contacts, "iterations": iteration,
                        "position_error_m": float(np.linalg.norm(error[:3])),
                        "rotation_error_rad": float(np.linalg.norm(error[3:]))}
            self.mj.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jacobian = np.r_[jp[:, self.qvel_ids], jr[:, self.qvel_ids]]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .003 ** 2 * np.eye(6), error)
            delta *= min(1., .12 / max(np.max(np.abs(delta)), 1e-12))
            q = np.clip(q + delta, self.limits[:, 0] + self.joint_margin, self.limits[:, 1] - self.joint_margin)
        return {"success": False, "reason": "IK_NOT_CONVERGED", "q": q.tolist()}

    def validate_path(self, q_path, *, expected_start=None):
        self.sync()
        initial = np.asarray(self.data.qpos[self.qpos_ids] if expected_start is None else expected_start).copy()
        q_path = np.asarray(q_path, float)
        if q_path.ndim != 2 or q_path.shape[1] != 7 or not np.isfinite(q_path).all():
            raise ValueError("Invalid joint path")
        if not len(q_path) or np.max(np.abs(q_path[0] - initial)) > .03:
            return {"valid": False, "reason": "PATH_START_MISMATCH"}
        violation = lambda q: np.maximum(np.maximum(self.limits[:, 0] - q, q - self.limits[:, 1]), 0.)
        measured_violation = violation(initial)
        if measured_violation.max() > self.soft_limit_tolerance:
            return {"valid": False, "reason": "JOINT_LIMIT", "index": 0,
                    "measured_violation_rad": measured_violation.tolist()}
        previous_violation = measured_violation.copy()
        for index, q in enumerate(q_path):
            self.environment.check_budget()
            current_violation = violation(q)
            if (np.any(current_violation > previous_violation + 1e-10)
                    or (index == len(q_path) - 1 and np.any(current_violation > 1e-10))):
                return {"valid": False, "reason": "JOINT_LIMIT", "index": index}
            previous_violation = current_violation
            self.set_q(q)
            failures = self.contacts()
            if failures:
                return {"valid": False, "reason": "COLLISION_REJECTED", "index": index, "contacts": failures}
        return {"valid": True, "samples": len(q_path), "backend": "isolated_mujoco_static_preview",
                "measured_soft_limit_violation_rad": measured_violation.tolist(),
                "start_is_observed": expected_start is None,
                "bounded_inward_recovery": bool(measured_violation.max() > 0)}

    def calibration_samples(self, count=20):
        self.sync()
        initial = self.data.qpos[self.qpos_ids].copy()
        rng = np.random.default_rng(self.environment.config["seed"])
        samples = []
        for i in range(count):
            q = initial if i == 0 else np.clip(initial + rng.uniform(-.12, .12, 7),
                                             self.limits[:, 0] + .001, self.limits[:, 1] - .001)
            self.set_q(q)
            samples.append({"q_arm": q.tolist(), "world_T_eef": self.tcp().tolist()})
        return samples
