"""Explicit cuRobo global planning and isolated MuJoCo contact-phase paths."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from graspgen_libero_bridge.common.rpc import WorkerClient
from graspgen_libero_bridge.common.frames import interpolate_pose, pose_error
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
from graspgen_libero_bridge.adapters.robocasa.native import write_json
from graspgen_libero_bridge.adapters.robocasa.scene import export_scene
from graspgen_libero_bridge.runtime.robocasa_worker_env import worker_environment


class RoboCasaPlanner:
    def __init__(self, environment, local):
        self.environment, self.local = environment, local
        self.worker = None
        self.configured_base = None
        self.attached_target = None
        self.carried_contents = []
        self.plans = 0

    def scene(self, target):
        return export_scene(self.environment, target,
                            self.carried_contents if self.attached_target == target else ())

    def _request(self, request):
        self.environment.check_budget()
        try:
            return self.worker.request(request, timeout=max(.1, self.environment.remaining_seconds()))
        except (RuntimeError, OSError, TimeoutError) as exc:
            code = "STORAGE_ERROR" if "No space left on device" in str(exc) else "PLANNER_ERROR"
            raise EpisodeFailure(code, str(exc), details={"command": request.get("command")}) from exc

    def configure(self, target):
        env = self.environment
        if self.worker is None:
            paths = env.config["paths"]
            self.worker = WorkerClient(paths["planner_python"], "graspgen_libero_bridge.planning.robocasa_planner_worker",
                env.output / "planner_worker", arguments=("--robosuite-assets", paths["robosuite_assets"]),
                environment=worker_environment(env.output),
                timeout=min(180., env.remaining_seconds()))
        scene = self.scene(target)
        samples = self.local.calibration_samples()
        for sample in samples:
            sample.update(q_fingers=scene["q_fingers"], world_T_base=scene["world_T_base"])
        response = self._request({"command": "configure", "scene": scene,
                   "calibration_samples": samples, "output_dir": str(env.output / "curobo_model")})
        if not response.get("success"):
            raise EpisodeFailure("ROBOT_MODEL_MISMATCH", "cuRobo configuration failed", details=response)
        self.configured_base = np.asarray(scene["world_T_base"])
        self.attached_target = None

    def ensure_configuration(self, target):
        current = np.asarray(self.environment.robot_state()["world_T_base"])
        if self.worker is None:
            self.configure(target)
        elif not np.allclose(current, self.configured_base, atol=1e-7, rtol=0):
            held, contents = self.attached_target, list(self.carried_contents)
            self.configure(target)
            if held is not None:
                self.attached_target, self.carried_contents = held, contents
                self.local.attach([held, *contents])
                self._request({"command": "attach", "scene": self.scene(held)})

    def global_joint_plan(self, target, q_goal, *, label):
        self.local.sync()
        self.local.set_q(q_goal)
        goal = self.local.tcp()
        return self.global_plan(target, goal, label=label, joint_goal=q_goal)

    def _shortcut(self, target, response, directory, *, allow_target_contact, deadline):
        from graspgen_libero_bridge.planning.robocasa_shortcut import shortcut_path
        with np.load(response["path_file"], allow_pickle=False) as path:
            original = np.asarray(path["q_arm"])
        def validate(segment):
            native = self.local.validate_path(segment, expected_start=segment[0])
            if not native["valid"]:
                return False
            return self.check_proxy_path(target, segment, allow_target_contact=allow_target_contact)["valid"]
        shortened, report = shortcut_path(original, validate,
            max_checks=self.environment.config["policy"].get("max_path_shortcut_checks", 0),
            remaining_seconds=lambda: deadline - self.environment.clock())
        if report["accepted_shortcuts"]:
            # Every replaced segment passed both engines. Validate the stitched
            # path again from the real observed start before making it executable.
            final = self.local.validate_path(shortened)
            report["stitched_native_preview"] = final
            if final["valid"]:
                poses = []
                self.local.sync()
                for q in shortened:
                    self.local.set_q(q); poses.append(self.local.tcp())
                path = directory / "SHORTCUT_PATH.npz"
                np.savez_compressed(path, q_arm=shortened, world_T_eef=np.asarray(poses))
                response.update(original_path_file=response["path_file"], path_file=str(path),
                                path_postprocess="native_and_curobo_validated_joint_shortcuts")
        write_json(directory / "SHORTCUT.json", report)
        response["shortcut_report"] = report
        return response

    def global_plan(self, target, goal, *, label, allow_target_contact=False, joint_goal=None):
        env = self.environment
        if env.config["policy"]["global_planner"] == "local_ik":
            return self.local_plan(goal, label=label)
        self.ensure_configuration(target)
        if self.attached_target is not None:
            self.local.attach([self.attached_target, *self.carried_contents])
            self._request({"command": "detach"})
            self._request({"command": "attach", "scene": self.scene(self.attached_target)})
        self.plans += 1
        directory = env.output / "plans" / (f"{self.plans:04d}_" + label)
        request = {"command": "plan_pose", "scene": self.scene(target), "goal_pose": np.asarray(goal).tolist(),
                   "allow_target_contact": bool(allow_target_contact),
                   "out_dir": str(directory), "timeout_s": max(.1, min(120., env.config["policy"]["plan_timeout_s"], env.remaining_seconds()))}
        if joint_goal is not None:
            request["q_goal"] = np.asarray(joint_goal, float).tolist()
        deadline = env.clock() + request["timeout_s"]
        attempts = []
        response = {"success": False, "reason": "PLANNING_BUDGET_EXHAUSTED"}
        for attempt in range(1 + env.config["policy"]["plan_validation_retries"]):
            remaining = deadline - env.clock()
            if remaining < .1:
                break
            attempt_directory = directory if attempt == 0 else directory.with_name(directory.name + f"_retry{attempt}")
            request.update(out_dir=str(attempt_directory), timeout_s=max(.1, remaining),
                           command=("plan_joint" if attempt == 0 else "plan_joint_graph") if joint_goal is not None else
                                   ("plan_pose" if attempt == 0 else "plan_pose_graph"))
            response = self._request(request)
            response.setdefault("algorithm", "curobo_trajopt" if attempt == 0 else "curobo_graph_only")
            if response.get("success"):
                with np.load(response["path_file"], allow_pickle=False) as path:
                    validation = self.local.validate_path(path["q_arm"])
                write_json(attempt_directory / "MUJOCO_PREVIEW.json", validation)
                if not validation["valid"]:
                    response.update(success=False, reason="MUJOCO_PREVIEW_REJECTED", native_preview=validation)
            attempts.append({"attempt": attempt, "timeout_s": request["timeout_s"],
                             "success": response.get("success"), "reason": response.get("reason"),
                             "algorithm": response["algorithm"]})
            # The optimizer can accept its coarse samples while the returned
            # interpolation still hits an obstacle. Re-plan within the same
            # deadline; never execute that rejected path or relax collisions.
            if response.get("reason") not in {"interpolated_path_validation_failed", "MUJOCO_PREVIEW_REJECTED"}:
                break
        response["planning_attempts"] = attempts
        response["backend"] = "curobo_global"
        if (response.get("success") and "graph" in response.get("algorithm", "")
                and env.config["policy"].get("max_path_shortcut_checks", 0) > 0):
            response = self._shortcut(target, response, directory,
                                      allow_target_contact=allow_target_contact, deadline=deadline)
        write_json(directory / "BRIDGE_RESULT.json", response)
        return response

    def check_proxy_path(self, target, joints, *, allow_target_contact=False):
        self.ensure_configuration(target)
        return self._request({"command": "check_joint_path", "scene": self.scene(target),
                              "q_arm": np.asarray(joints).tolist(), "allow_target_contact": bool(allow_target_contact)})

    def contact_plan(self, goal, *, label, start_q=None):
        """Run the unchanged v10 contact planner against shadow MuJoCo data.

        As in v10, actual collision geometry is authoritative for these local
        contact phases. No extra sphere-proxy veto or contact exemption is added.
        Global cuRobo paths still require their own constraints and native checks.
        """
        from graspgen_libero_bridge.adapters.robocasa.v10_preview import V10PreviewAdapter, V10PreviewMonitor
        from graspgen_libero_bridge.planning.cartesian import plan_cartesian
        env, local = self.environment, self.local
        env.check_budget()
        local.sync()
        if start_q is not None:
            local.set_q(np.asarray(start_q))
        self.plans += 1
        directory = env.output / "plans" / (f"{self.plans:04d}_" + label)
        adapter = V10PreviewAdapter(env, local, local.target)
        reply = plan_cartesian(adapter, V10PreviewMonitor(adapter), goal, directory,
                               allow_finger_target=local.allow_finger_target)
        reply.update(backend="libero_v10_mujoco_cartesian_dls", isolated_preview=True)
        env.check_budget()
        write_json(directory / "BRIDGE_RESULT.json", reply)
        return reply

    def local_plan(self, goal, *, label, check_proxy_target=None, allow_target_contact=False):
        env, local = self.environment, self.local
        self.plans += 1
        directory = env.output / "plans" / (f"{self.plans:04d}_" + label)
        directory.mkdir(parents=True)
        start = np.asarray(env.robot_state()["q_arm"])
        current = np.asarray(env.robot_state()["world_T_eef"])
        p, r = pose_error(goal, current)
        steps = max(2, int(np.ceil(max(np.linalg.norm(p) / .012, np.linalg.norm(r) / .08))) + 1)
        joints, poses = [start.copy()], [current.copy()]
        q = start.copy()
        for u in np.linspace(0., 1., steps)[1:]:
            waypoint = interpolate_pose(current, goal, float(u))
            solved = local.solve(waypoint, start=q)
            if not solved["success"]:
                response = {"success": False, "backend": "isolated_mujoco_local_ik",
                            "reason": solved["reason"], "waypoint": float(u), "ik": solved}
                write_json(directory / "BRIDGE_RESULT.json", response)
                return response
            target_q = np.asarray(solved["q"])
            count = max(1, int(np.ceil(np.max(np.abs(target_q - q)) / .025)))
            for alpha in np.linspace(0., 1., count + 1)[1:]:
                qi = (1. - alpha) * q + alpha * target_q
                local.set_q(qi)
                joints.append(qi.copy()); poses.append(local.tcp())
            q = target_q
        checked = local.validate_path(joints)
        response = {"success": checked["valid"], "backend": "isolated_mujoco_local_ik", "preview": checked}
        if checked["valid"] and check_proxy_target is not None and env.config["policy"]["global_planner"] == "curobo":
            proxy = self.check_proxy_path(check_proxy_target, joints, allow_target_contact=allow_target_contact)
            response["curobo_proxy_preview"] = proxy
            if not proxy["valid"]:
                response.update(success=False, reason="PLANNING_PROXY_PATH_REJECTED")
                write_json(directory / "BRIDGE_RESULT.json", response)
                return response
        if checked["valid"]:
            path = directory / "PATH.npz"
            np.savez_compressed(path, q_arm=np.asarray(joints), world_T_eef=np.asarray(poses))
            response.update(path_file=str(path), waypoints=len(joints))
        else:
            response["reason"] = checked["reason"]
        write_json(directory / "BRIDGE_RESULT.json", response)
        return response

    def attach(self, target, contents=()):
        self.local.attach([target, *contents])
        if self.environment.config["policy"]["global_planner"] == "curobo":
            self.ensure_configuration(target)
            self.carried_contents = list(contents)
            self.attached_target = target
            response = self._request({"command": "attach", "scene": self.scene(target)})
            if not response.get("success"):
                raise EpisodeFailure("ATTACHMENT_ERROR", "cuRobo object attachment failed", details=response)
        self.attached_target = target
        self.carried_contents = list(contents)

    def detach(self):
        self.local.detach()
        if self.worker is not None and self.attached_target is not None:
            self._request({"command": "detach"})
        self.attached_target = None
        self.carried_contents = []

    def close(self):
        if self.worker is not None:
            self.worker.close()
