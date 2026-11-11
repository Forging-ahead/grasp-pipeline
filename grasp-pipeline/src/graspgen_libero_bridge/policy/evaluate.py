"""Run an explicitly specified rigid-object pick-and-place task in LIBERO."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from graspgen_libero_bridge.common.layout import source_root, source_fingerprints
import subprocess
import time
import traceback

import numpy as np

from graspgen_libero_bridge.control.controller import Controller, Recorder, ExecutionError, descendants
from graspgen_libero_bridge.adapters.libero.env_adapter import EnvAdapter
from graspgen_libero_bridge.common.frames import grasp_to_control
from graspgen_libero_bridge.perception.pointcloud import sample_mujoco_object_points, robosuite_target_points
from graspgen_libero_bridge.common.rpc import WorkerClient
from graspgen_libero_bridge.adapters.libero.scene import export_scene
from graspgen_libero_bridge.planning.cartesian import plan_cartesian, preview_path_endpoint, _snapshot, _restore
from graspgen_libero_bridge.policy.placement import container_release_candidates, can_retrace_after_ik_failure
from graspgen_libero_bridge.policy.transfer import transfer_candidates
from graspgen_libero_bridge.planning.grasp_candidate import control_target_distance
from graspgen_libero_bridge.planning.grasp_contact import preview_grasp_contacts


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def progress(stage, **values):
    print(json.dumps({"stage": stage, **values}, allow_nan=False), flush=True)


def support_geoms(adapter, name):
    sim = adapter.env.sim
    bodies = descendants(sim.model, adapter.task_env.obj_body_id[name])
    names = set()
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(sim.model.geom_bodyid[g1]), int(sim.model.geom_bodyid[g2])
        if (b1 in bodies) != (b2 in bodies):
            g = g2 if b1 in bodies else g1
            if not (sim.model.geom_id2name(g) or "").startswith(("robot", "gripper")):
                names.add(sim.model.geom_id2name(g) or f"geom_{g}")
    return sorted(n for n in names if n)


def object_geom_names(adapter, name):
    model = adapter.env.sim.model
    if name not in adapter.task_env.obj_body_id:
        from graspgen_libero_bridge.adapters.libero.regions import support_surface
        return support_surface(adapter)["geom_names"]
    bodies = descendants(model, adapter.body_id(name))
    return [model.geom_id2name(i) for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies and model.geom_id2name(i)]


class PickPlace:
    def __init__(self, config, run_dir):
        self.config = dict(config)
        self.output = Path(run_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.adapter = EnvAdapter(config, self.output)
        self.recorder = None
        self.controller = None
        self.workers = []
        self.result = {"schema_version": 2, "status": "running", "task_success": False,
                       "pipeline_success": False, "grasp_success": None,
                       "grasp_assessment": "not_attempted", "close_sequence_completed": False,
                       "scope": "Explicit-target rigid-object pick-and-place; no learned language policy",
                       "input_mode": config.get("input_mode", "geometry"),
                       "initial_state_id": int(config.get("init_state_id", 0)),
                       "uses_ground_truth_object_identity": True,
                       "uses_ground_truth_segmentation": config.get("input_mode", "geometry") == "rgbd",
                       "uses_ground_truth_object_geometry": True,
                       "grasp_input_uses_full_object_geometry": config.get("input_mode", "geometry") == "geometry",
                       "grasp_input_uses_depth": config.get("input_mode", "geometry") == "rgbd",
                       "grasp_input_uses_rgb_features": False,
                       "placement_uses_ground_truth_geometry": True,
                       "execution_uses_ground_truth_object_poses": True,
                       "uses_ground_truth_collision_world": True,
                       "physical_attempts": 0}
        self.started = time.monotonic()
        self.result["bridge_source_sha256"] = source_fingerprints()
        self.result["planner_strategy"] = {"global_motion": "curobo", "final_grasp_approach": "curobo_batch_ik",
            "contact_phases": "MuJoCo Jacobian IK with actual-geometry contact validation", "execution": "default OSC_POSE"}
        write_json(self.output / "config.json", self.config)

    def scene(self):
        return export_scene(self.adapter, self.output / "geometry")

    def start_workers(self):
        config = self.config
        root = str(source_root())
        common = {"PYTHONPATH": root, "PYTHONUTF8": "1"}
        model = WorkerClient(config["graspgen_python"], "graspgen_libero_bridge.models.graspgen_worker", self.output / "model_worker",
            arguments=["--config", config["graspgen_config"], "--grasp-repo", config["grasp_repo"]],
            environment={**common, "CUDA_VISIBLE_DEVICES": str(config.get("model_gpu", 6))})
        self.workers.append(model)
        assets = str(Path(__import__("robosuite").__file__).parent / "models/assets")
        planner = WorkerClient(config["planner_python"], "graspgen_libero_bridge.planning.planner_worker", self.output / "planner_worker",
            arguments=["--robosuite-assets", assets],
            environment={**common, "CUDA_VISIBLE_DEVICES": str(config.get("planner_gpu", 7))})
        self.workers.append(planner)
        self.model, self.planner = model, planner

    def plan_and_execute(self, goal, phase, *, ignore=(), allow_contact=True, gripper=1):
        progress(phase + "_planning", steps=self.adapter.step_count)
        self.controller.ensure_joint_margin(phase, gripper=gripper, allow_finger_target=allow_contact)
        scene = self.scene()
        obstacle_names = {x["name"] for x in scene["obstacles"]}
        request = {"command": "plan_pose", "scene": scene, "goal_pose": np.asarray(goal).tolist(),
                   "out_dir": str(self.output / "plans" / phase), "timeout_s": self.config.get("plan_timeout_s", 20),
                   "allow_target_contact": bool(allow_contact), "ignore_obstacles": [n for n in ignore if n in obstacle_names]}
        if phase in ("lift", "place", "retreat"):
            directory = Path(request["out_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "SCENE.json", scene)
            write_json(directory / "CONTACT_POLICY.json", {"planner": "mujoco_cartesian_dls",
                "allow_finger_target": bool(allow_contact), "held_target": self.controller.monitor.held,
                "target_support_permissions": sorted(self.controller.monitor.allowed_support_geoms),
                "robot_world_contact_exemptions": [], "source": "actual MuJoCo collision geometry"})
            reply = plan_cartesian(self.adapter, self.controller.monitor, goal, directory,
                                   allow_finger_target=allow_contact)
            if (phase == "lift" and not reply["success"] and reply.get("state_restored") is True
                    and reply.get("reason") in {"cartesian_ik_stalled", "cartesian_ik_iteration_limit",
                        "cartesian_ik_collision_blocked", "cartesian_collision"}):
                original_failure = reply
                request["out_dir"] = str(directory / "global_fallback")
                reply = self.planner.request(request)
                self.result.setdefault("contact_planning_fallbacks", []).append(dict(
                    phase=phase,native_failure=original_failure,global_reply=reply,
                    execution_requires_actual_MuJoCo_preflight=True))
        else:
            reply = self.planner.request(request)
            if (phase in ("transfer", "approach_replan", "region_reposition") and not reply["success"]
                    and reply.get("reason") in ("invalid_start_state", "curobo_plan_failed")
                    and "JOINT_LIMIT" not in str(reply.get("status", ""))):
                global_failure = reply
                reply = plan_cartesian(self.adapter,self.controller.monitor,goal,
                    Path(request["out_dir"]) / "native_fallback",allow_finger_target=allow_contact)
                self.result.setdefault("transfer_geometry_fallbacks", []).append(dict(
                    global_failure=global_failure,native_reply=reply,
                    collision_exemptions_added=[],uses_actual_MuJoCo_geometry=True))
        if reply.get("planner") == "mujoco_cartesian_dls" and reply.get("state_restored") is not True:
            raise ExecutionError("Planning preview did not restore the physical state",
                code="planning_state_restore_failed",category="technical",phase=phase,details=reply)
        if not reply["success"]:
            raise ExecutionError(f"{phase} planning failed: {reply}", code="motion_plan_failed",
                                 category="planning", phase=phase, details=reply)
        self.controller.follow_path(reply["path_file"], gripper=gripper, phase=phase, allow_finger_target=allow_contact)
        return reply

    def run(self, doctor=False):
        try:
            progress("creating_libero")
            self.adapter.reset()
            self.recorder = Recorder(self.adapter, self.output)
            self.controller = Controller(self.adapter, self.recorder, self.adapter.target_object)
            if self.config.get("unsupported_skill"):
                self.recorder.phase = "unsupported_skill_no_policy"
                for _ in range(40):
                    self.adapter.obs, _, _, _ = self.adapter.env.step(np.zeros(7))
                    self.adapter.step_count += 1
                    self.recorder.capture(np.zeros(7))
                self.result.update(status="unsupported", reason="No drawer/push/stove skill implemented; scene-only video, no GraspGen attempt", task_success=False)
                return self.result
            initial_object = self.adapter.target_pose(self.adapter.target_object).copy()
            initial_support = support_geoms(self.adapter, self.adapter.target_object)
            self.result["initial_target_support_geoms"] = initial_support
            initial_scene = self.scene()
            samples = self.adapter.calibration_samples()
            write_json(self.output / "initial_scene.json", initial_scene)
            write_json(self.output / "calibration_samples.json", samples)
            self.result["provenance"] = self.adapter.provenance
            self.result["initial_forbidden_contacts"] = self.controller.monitor.check()
            if self.result["initial_forbidden_contacts"]:
                raise ExecutionError("Initial robot scene has forbidden contact", code="initial_contact",
                                     category="collision", phase="initial")
            points, metadata = self.get_points()
            points_path = self.output / "input_points.npz"
            np.savez_compressed(points_path, points_world=points)
            write_json(self.output / "perception.json", metadata)
            progress("libero_ready", points=len(points), collision_geometries=len(initial_scene["obstacles"]))
            if doctor:
                self.result.update(status="doctor_passed", stage="environment_and_geometry", cpu_kinematics_samples=len(samples))
                return self.result
            goals = self.adapter.task_env.parsed_problem["goal_state"]
            self.result["goal_predicates"] = goals
            expected = [self.config.get("goal_relation", "on"), self.adapter.target_object, self.config.get("goal_destination", self.adapter.destination)]
            normalized_goals = [[str(x).lower() for x in goal] for goal in goals]
            eligible = expected in normalized_goals and (len(goals) == 1 or self.config.get("composite_subgoal_execution", False))
            if expected[0] not in ("on", "in") or not eligible:
                raise ExecutionError("Task requires a policy outside supported explicit On/In goals",
                                     code="unsupported_task_policy", category="unsupported", phase="task_setup")
            self.start_workers()
            progress("qualifying_planner_fk")
            qualified = self.planner.request({"command": "configure", "scene": initial_scene,
                "calibration_samples": samples, "output_dir": str(self.output / "planner_config"),
                "max_robot_spheres": self.config.get("max_robot_spheres", 300),
                "attachment_slots": self.config.get("attachment_slots", 128)})
            self.result["fk_qualification"] = qualified
            progress("generating_grasps")
            grasp_reply = self.model.request({"command": "infer", "points_path": str(points_path),
                "output_path": str(self.output / "grasp_candidates.npz"), "seed": self.config.get("seed", 0),
                "num_grasps": self.config.get("num_grasps", 200)})
            self.result["grasp_generation"] = grasp_reply
            with np.load(self.output / "grasp_candidates.npz", allow_pickle=False) as d:
                poses, scores = d["poses_world_grasp"].copy(), d["scores"].copy()
            order = np.argsort(-scores, kind="stable")
            candidates = []
            engagement_rejections = []
            engagement_margin = float(self.config.get("grasp_control_target_margin_m", .01))
            if not np.isfinite(engagement_margin) or not 0 <= engagement_margin <= .03:
                raise ValueError("Invalid target-engagement margin")
            for index in order:
                # Engineering default restricts entry from above, not horizontal/under-table.
                if poses[index, 2, 2] > -float(self.config.get("minimum_downward_approach", .25)):
                    continue
                control = grasp_to_control(poses[index], self.config.get("grasp_control_offset_m", .097))
                distance = control_target_distance(control, initial_scene["target"]["bounds_world"])
                if distance > engagement_margin:
                    engagement_rejections.append(dict(generation_index=int(index),distance_m=distance))
                    continue
                candidates.append({"world_T_eef": control.tolist(),
                                   "score": float(scores[index]), "generation_index": int(index)})
            self.result["target_engagement_filter"] = dict(margin_m=engagement_margin,
                uses_ground_truth_collision_bounds=True,rejected=engagement_rejections,
                limitation="Necessary grip-centre proximity only; not a grasp-success guarantee")
            candidates = candidates[int(self.config.get("candidate_offset", 0)):]
            write_json(self.output / "screened_candidates.json", candidates)
            if self.config.get("prepare_open_gripper", True):
                # Approach execution commands an open gripper. Make that state
                # physical before generating/checking approach and descent paths.
                self.controller.hold(20, -1, "prepare_gripper", False)
                prepared = self.adapter.state()
                self.result["planning_gripper_state"] = dict(q_fingers=prepared["q_fingers"],
                    width_m=prepared["gripper_width"],source="measured after physical opening")
                if prepared["gripper_width"] < .07:
                    raise ExecutionError("Gripper did not reach the planned open aperture",
                        code="gripper_preparation_failed",category="control",phase="prepare_gripper")
            selected = self.select_candidate(candidates)
            self.result["selected_grasp"] = selected["selected"]
            progress("executing_grasp", generation_index=selected["selected"]["generation_index"])
            self.result["physical_attempts"] = 1
            self.result["grasp_assessment"] = "not_evaluated"
            self.execute_grasp_paths(selected)
            self.controller.hold(30, 1, "close_gripper", True)
            self.result["close_sequence_completed"] = True
            support = sorted(set(initial_support) | set(support_geoms(self.adapter, self.adapter.target_object)))
            self.result["departure_support_geoms"] = support
            self.planner.request({"command": "attach", "scene": self.scene()})
            self.controller.monitor.set_held(True, support)
            lift = np.asarray(self.adapter.state()["world_T_eef"])
            lift[2, 3] += float(self.config.get("lift_height_m", .15))
            self.plan_and_execute(lift, "lift", ignore=support)
            self.controller.monitor.allowed_support_geoms.clear()
            self.controller.hold(15, 1, "verify_hold", True)
            lifted_object = self.adapter.target_pose(self.adapter.target_object)
            lift_distance = float(lifted_object[2, 3]-initial_object[2, 3])
            self.result["object_lift_m"] = lift_distance
            self.result["grasp_success"] = lift_distance >= .06
            self.result["grasp_assessment"] = "passed" if self.result["grasp_success"] else "failed"
            if not self.result["grasp_success"]:
                raise ExecutionError(f"Object was not lifted: {lift_distance:.4f} m",
                                     code="lift_verification_failed", category="grasp_verification", phase="verify_hold")
            relation = np.linalg.inv(np.asarray(self.adapter.state()["world_T_eef"])) @ lifted_object
            # Re-measure after departure: physical closure may shift the object.
            self.planner.request({"command": "detach"})
            self.planner.request({"command": "attach", "scene": self.scene()})
            dest = self.adapter.target_pose(self.adapter.destination)
            target_points = sample_mujoco_object_points(self.adapter.env, self.adapter.target_object, count=4096, seed=0, geometry="collision")
            bottom_offset = float(lifted_object[2, 3]-target_points[:, 2].min())
            desired_object = initial_object.copy()
            desired_object[:2, 3] = dest[:2, 3]
            goal_destination = self.config.get("goal_destination", self.adapter.destination)
            if goal_destination == self.adapter.destination:
                dest_points = sample_mujoco_object_points(self.adapter.env, self.adapter.destination, count=4096, seed=0, geometry="collision")
                desired_object[2, 3] = float(dest_points[:, 2].max()) + bottom_offset + .015
            if goal_destination != self.adapter.destination:
                sim = self.adapter.env.sim
                site_id = sim.model.site_name2id(goal_destination)
                site_pos = np.asarray(sim.data.site_xpos[site_id])
                desired_object[:2, 3] = site_pos[:2]
                if self.config.get("goal_relation") == "in":
                    # Descend to the container floor, retaining physical collision checks.
                    extent = np.abs(np.asarray(sim.data.site_xmat[site_id]).reshape(3,3)) @ np.asarray(self.adapter.task_env.object_sites_dict[goal_destination].size)
                    surface_z = float(site_pos[2] - extent[2])
                else:
                    surface_z = float(site_pos[2])
                    if self.adapter.destination not in self.adapter.task_env.obj_body_id:
                        from graspgen_libero_bridge.adapters.libero.regions import support_surface
                        support = support_surface(self.adapter)
                        self.result["region_support"] = support
                        surface_z = support["physical_surface_z_m"]
                desired_object[2, 3] = surface_z + bottom_offset + .015
            if self.config.get('geometric_placement_search', False) and goal_destination != self.adapter.destination:
                from graspgen_libero_bridge.adapters.libero.placement_geometry import gravity_frame, orientation_candidates, region_support_height
                surface_z, physical_support = region_support_height(self.adapter, surface_z)
                desired_object[2, 3] = surface_z + bottom_offset + .015
                self.result['physical_destination_support'] = dict(height_m=surface_z, geom=physical_support)
                if self.config.get('goal_relation') == 'in':
                    region = np.eye(4)
                    region[:3, :3] = np.asarray(sim.data.site_xmat[site_id]).reshape(3, 3)
                    region[:3, 3] = site_pos
                    frame, size = gravity_frame(region, self.adapter.task_env.object_sites_dict[goal_destination].size)
                    local_points = (target_points-lifted_object[:3, 3]) @ lifted_object[:3, :3]
                    options = orientation_candidates(desired_object, local_points, frame, size, surface_z, require_origin_containment=True)
                    self.result['opening_orientation_search'] = dict(fitting_orientations=len(options),
                        uses_true_collision_points=True, preserved_initial_orientation=bool(options and np.allclose(options[0][:3,:3],initial_object[:3,:3])))
                    if not options:
                        raise ExecutionError('No bounded object orientation fits the goal opening',
                            code='opening_fit_failed',category='planning',phase='place')
                    desired_object = options[0]
                    self.placement_object_options=options[:int(self.config.get('placement_orientation_trials',4))]
                    self.opening_constraint = (local_points, frame, size)
            if 'packed_surface_xy_world_m' in self.config:
                desired_object[:2,3]=self.config['packed_surface_xy_world_m']
                self.result['surface_packing_target_xy_world_m']=desired_object[:2,3].tolist()
            self.result["placement_target_world"] = desired_object.tolist()
            desired_object, transfer_reply = self.transfer_with_alternatives(desired_object, relation)
            self.result["placement_target_world"] = desired_object.tolist()
            destination_geoms = object_geom_names(self.adapter, self.adapter.destination)
            self.controller.monitor.allowed_support_geoms = set(destination_geoms)
            open_container_search = self.config.get("goal_relation") == "in" and bool(
                self.config.get("open_container_release_search", self.adapter.destination.startswith("basket_")))
            if open_container_search:
                placement_path = self.place_in_container(desired_object, relation, target_points, lifted_object)
            elif self.adapter.destination not in self.adapter.task_env.obj_body_id:
                desired_object = self.place_on_region(desired_object, relation, destination_geoms)
                self.result["placement_target_world"] = desired_object.tolist()
            elif self.config.get("surface_release_search", False):
                desired_object, placement_path = self.place_above_support(desired_object, relation, destination_geoms)
                self.result["placement_target_world"] = desired_object.tolist()
            else:
                reply = self.plan_and_execute(desired_object @ np.linalg.inv(relation), "place", ignore=destination_geoms)
                placement_path = reply["path_file"]
            self.planner.request({"command": "detach"})
            self.controller.monitor.set_held(False)
            self.controller.hold(25, -1, "release", True)
            self.result["release_completed"] = True
            retreat = np.asarray(self.adapter.state()["world_T_eef"])
            retreat[2, 3] += .10
            try:
                self.plan_and_execute(retreat, "retreat", gripper=-1)
            except ExecutionError as exc:
                if (not open_container_search or exc.code != "motion_plan_failed"
                        or not can_retrace_after_ik_failure(exc.details)):
                    raise
                self.result["retreat_fallback"] = dict(policy="retrace_placement_and_transfer",
                    primary_failure=exc.record(), open_gripper_collision_checks=True)
                reverse_paths=[("reverse_place", placement_path)]
                reverse_paths.extend(("reverse_container_shift_%d"%i,path) for i,path in enumerate(reversed(getattr(self,'container_reposition_paths',[]))))
                reverse_paths.append(("reverse_transfer",transfer_reply['path_file']))
                for name, path in reverse_paths:
                    reverse = self.output / "plans" / name / "PATH.npz"
                    reverse.parent.mkdir(parents=True, exist_ok=True)
                    with np.load(path, allow_pickle=False) as recorded:
                        np.savez_compressed(reverse, q_arm=recorded["q_arm"][::-1].copy(),
                                            world_T_eef=recorded["world_T_eef"][::-1].copy())
                    # Recheck against the actual open gripper and released object.
                    self.controller.follow_path(reverse, phase=name, gripper=-1, allow_finger_target=True)
            self.controller.hold(40, -1, "settle")
            self.result.update(status="completed", pipeline_success=bool(self.adapter.check_success()),
                               task_success=bool(self.adapter.success_seen or self.adapter.check_success()))
            if not self.result["pipeline_success"]:
                self.result["reason"] = "LIBERO official task predicate was false after release and settling"
            progress("completed", task_success=self.result["task_success"], grasp_success=self.result["grasp_success"])
            return self.result
        except (Exception, KeyboardInterrupt) as exc:
            self.result.update(status="aborted" if isinstance(exc, ExecutionError) else "error",
                               reason=str(exc), traceback=traceback.format_exc())
            self.result["failure"] = (exc.record() if isinstance(exc, ExecutionError) else
                dict(code="unhandled_exception", category="technical", phase=None, details={"type": type(exc).__name__}))
            if self.result["failure"]["category"] == "unsupported":
                self.result["status"] = "unsupported"
            if self.result["failure"]["phase"] is None and self.recorder is not None:
                self.result["failure"]["phase"] = self.recorder.phase
            if isinstance(exc, KeyboardInterrupt):
                self.result.update(status="interrupted", reason="Bridge run interrupted; partial results preserved")
            if self.adapter.env is not None:
                self.result["official_predicate_at_failure"] = bool(self.adapter.check_success())
            progress(self.result["status"], reason=str(exc), failure=self.result["failure"])
            return self.result
        finally:
            if self.adapter.env is not None and self.adapter.obs is not None:
                final_success = bool(self.adapter.check_success())
                self.result["official_goal_at_end"] = final_success
                self.result["task_success"] = bool(self.adapter.success_seen or final_success)
            self.result["grasp_attempt_completed"] = self.result["grasp_success"] is not None
            if self.adapter.step_count == 0:
                self.result["physical_attempts"] = 0
                if self.result["grasp_success"] is None:
                    self.result["grasp_assessment"] = "not_attempted"
            self.result.update(wall_seconds=time.monotonic()-self.started, steps=self.adapter.step_count)
            if self.controller is not None:
                self.result["execution_checks"] = self.controller.events
            if self.recorder is not None:
                self.recorder.save()
            for worker in reversed(self.workers):
                worker.close()
            self.adapter.close()
            write_json(self.output / "RESULT.json", self.result)

    def get_points(self):
        args = dict(count=2048, seed=int(self.config.get("seed", 0)))
        mode = self.config.get("input_mode", "geometry")
        if mode == "geometry":
            return sample_mujoco_object_points(self.adapter.env, self.adapter.target_object, return_metadata=True, **args)
        if mode == "rgbd":
            return robosuite_target_points(self.adapter.env, self.adapter.obs, self.config.get("input_camera", "agentview"), self.adapter.target_object, **args)
        raise ValueError("input_mode must be geometry or rgbd")

    def execute_grasp_paths(self, selected):
        regenerate = False
        try:
            self.controller.follow_path(selected["approach_path_file"],phase="approach")
        except ExecutionError as exc:
            if exc.code != "joint_branch_deviation":
                raise
            self.result["approach_replanning"] = dict(trigger=exc.record(),
                policy="replan once from measured actual branch; retain original deviation guard")
            with np.load(selected["approach_path_file"],allow_pickle=False) as saved:
                goal = saved["world_T_eef"][-1].copy()
            self.plan_and_execute(goal,"approach_replan",gripper=-1,allow_contact=False)
            regenerate = True
        if not regenerate:
            try:
                self.controller.follow_path(selected["descend_path_file"],phase="descend",allow_finger_target=True)
                return
            except ExecutionError as exc:
                if exc.code != "path_preflight_rejected":
                    raise
                self.result["descent_replanning"] = dict(trigger=exc.record(),
                    policy="replan before descending using measured arm, open fingers and object poses")
        reply = plan_cartesian(self.adapter,self.controller.monitor,np.asarray(selected["selected"]["world_T_eef"]),
            self.output / "plans" / "descend_measured",allow_finger_target=True)
        self.result["measured_descent_plan"] = reply
        if not reply.get("success") or reply.get("state_restored") is not True:
            raise ExecutionError("Measured-state descent planning failed",code="motion_plan_failed",
                category="planning",phase="descend",details=reply)
        self.controller.follow_path(reply["path_file"],phase="descend",allow_finger_target=True)

    def transfer_with_alternatives(self, object_goal, relation):
        attempts = []
        self.result["transfer_search"] = {"attempts": attempts,
            "policy": "destination XY fixed; bounded yaw and preplace clearance search",
            "collision_exemptions_added": []}
        last_error = None
        alternatives=getattr(self,'placement_object_options',[object_goal])
        search=((orientation,option) for orientation,base in enumerate(alternatives)
            for option in transfer_candidates(base,relation,
                heights=self.config.get('transfer_heights_m',[.12,.08,.04,.16,.20]),
                yaw_degrees=self.config.get('transfer_yaw_degrees',[0,45,-45,90,-90,180])))
        for index,(orientation,option) in enumerate(search):
            if hasattr(self, 'opening_constraint'):
                from graspgen_libero_bridge.adapters.libero.placement_geometry import fits_opening
                if not fits_opening(option['object_goal'], *self.opening_constraint):
                    attempts.append(dict(height_m=option['height_m'], yaw_degrees=option['yaw_degrees'], rejected='object footprint outside opening'))
                    continue
            before = self.adapter.step_count
            try:
                reply = self.plan_and_execute(option["eef_goal"], "transfer")
            except ExecutionError as exc:
                attempts.append(dict(orientation_candidate=orientation,height_m=option["height_m"], yaw_degrees=option["yaw_degrees"],
                                     failure=exc.record()))
                # Never restart after partial execution or a control/contact error.
                if exc.code not in ("motion_plan_failed", "path_preflight_rejected") or self.adapter.step_count != before:
                    raise
                last_error = exc
                source = self.output / "plans" / "transfer"
                if source.exists():
                    import shutil
                    shutil.copytree(source, self.output / "plans" / ("transfer_rejected_%02d" % index))
                continue
            attempts.append(dict(orientation_candidate=orientation,height_m=option["height_m"], yaw_degrees=option["yaw_degrees"],success=True))
            self.result["transfer_search"]["selected"] = dict(orientation_candidate=orientation,height_m=option["height_m"],yaw_degrees=option["yaw_degrees"])
            return option["object_goal"], reply
        if last_error is not None:
            raise last_error
        raise ValueError("Transfer candidate configuration is empty")

    def place_on_region(self, object_goal, relation, destination_geoms):
        """Search interior XY points of a named arena region with real collisions.

        Unlike containers these regions are virtual goal areas on a support
        plane. The center may overlap a neighboring object; a region is not a
        single immutable coordinate. No obstacle/contact permission is removed.
        """
        sim = self.adapter.env.sim
        name = self.config["goal_destination"]
        site = int(sim.model.site_name2id(name))
        rotation = np.asarray(sim.data.site_xmat[site]).reshape(3, 3)
        size = np.asarray(self.adapter.task_env.object_sites_dict[name].size)
        attempts = []
        self.result["region_placement_search"] = dict(policy="center then interior XY offsets",
            offsets_as_fraction_of_half_size=.5, attempts=attempts, collision_exemptions_added=[])
        offsets = [(0.,0.)] + [(x*f,y*f) for f in (.5,.9) for x,y in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)]]
        self.result["region_placement_search"]["offset_fractions"] = [.5,.9]
        for index, (x,y) in enumerate(offsets):
            candidate = np.asarray(object_goal).copy()
            offset = rotation @ np.array([x*size[0],y*size[1],0.])
            candidate[:2,3] += offset[:2]
            try:
                if index:
                    above = candidate @ np.linalg.inv(relation)
                    above[2,3] = np.asarray(self.adapter.state()["world_T_eef"])[2,3]
                    self.plan_and_execute(above,"region_reposition",ignore=destination_geoms)
                self.plan_and_execute(candidate @ np.linalg.inv(relation), "place", ignore=destination_geoms)
            except ExecutionError as exc:
                attempts.append(dict(index=index,offset_world_m=offset.tolist(),success=False,failure=exc.record()))
                # Retry only a failed, restored kinematic preview, never a
                # control/collision failure after physical placement has begun.
                if exc.code != "motion_plan_failed" or exc.details.get("state_restored") is not True:
                    raise
                continue
            attempts.append(dict(index=index,offset_world_m=offset.tolist(),success=True))
            return candidate
        raise ExecutionError("No executable placement within the checked goal-region points",
            code="region_placement_exhausted",category="planning",phase="place",details={"attempts":attempts})

    def place_in_container(self, object_goal, relation, target_points_world, target_pose):
        """Search lowest collision-free release; gravity performs final settling."""
        self.controller.ensure_joint_margin("place", gripper=1, allow_finger_target=True)
        sim = self.adapter.env.sim
        region_name = self.config["goal_destination"]
        site = sim.model.site_name2id(region_name)
        region = np.eye(4)
        region[:3, :3] = np.asarray(sim.data.site_xmat[site]).reshape(3, 3)
        region[:3, 3] = np.asarray(sim.data.site_xpos[site])
        region_size = self.adapter.task_env.object_sites_dict[region_name].size
        if self.config.get('geometric_placement_search', False):
            from graspgen_libero_bridge.adapters.libero.placement_geometry import gravity_frame
            region, region_size = gravity_frame(region, region_size)
        local_points = (np.asarray(target_points_world)-target_pose[:3, 3]) @ target_pose[:3, :3]
        offsets=self.config.get("container_release_offsets_m", [0., .02, .04, .06, .08, .10, .12])
        shifts=[np.zeros(3)]
        if self.config.get('container_xy_search',False):
            from graspgen_libero_bridge.adapters.libero.placement_geometry import opening_offsets
            shifts=opening_offsets(object_goal,local_points,region,region_size)
        options=[]
        for order,shift in enumerate(shifts):
            shifted=np.asarray(object_goal).copy();shifted[:3,3]+=shift
            for option in container_release_candidates(shifted,relation,local_points,region,region_size,offsets):
                option.update(shift_world_m=shift.tolist(),shift_order=order)
                options.append(option)
        options.sort(key=lambda x:(x['offset_m'],x['shift_order']))
        self.container_reposition_paths=[]
        attempts = []
        self.result["placement_search"] = {"policy": "lowest_collision_free_release_over_opening",
            "opening_clearance_m": .002, "attempts": attempts,
            "robot_collision_exemptions_added": [], "max_release_offset_m": .12}
        if self.config.get('static_container_candidate_search', False):
            return self.place_in_container_static(options, attempts)
        for i, option in enumerate(options):
            directory = self.output / "plans" / "place" / f"release_{i:02d}"
            progress("place_candidate", offset_m=option["offset_m"],shift_world_m=option['shift_world_m'])
            if self.config.get('container_xy_search',False):
                current=np.asarray(self.adapter.state()['world_T_eef'])
                above=option['eef_pose'].copy()
                above[2,3]=max(current[2,3],above[2,3]+.02)
                if np.linalg.norm(above[:3,3]-current[:3,3])>.002:
                    before=self.adapter.step_count
                    try:
                        reposition=self.plan_and_execute(above,'region_reposition')
                    except ExecutionError as exc:
                        attempts.append(dict(offset_m=option['offset_m'],shift_world_m=option['shift_world_m'],reposition_failure=exc.record()))
                        if exc.code not in ('motion_plan_failed','path_preflight_rejected') or self.adapter.step_count!=before:raise
                        continue
                    import shutil
                    saved_path=self.output/'plans'/('container_shift_%03d'%len(self.container_reposition_paths))/'PATH.npz'
                    saved_path.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(reposition['path_file'],saved_path)
                    self.container_reposition_paths.append(str(saved_path))
            reply = plan_cartesian(self.adapter, self.controller.monitor, option["eef_pose"], directory,
                                   allow_finger_target=True)
            attempts.append(dict(offset_m=option["offset_m"],shift_world_m=option['shift_world_m'], success=bool(reply["success"]),
                reason=reply.get("reason"), collisions=reply.get("collisions", [])))
            if not reply["success"]:
                continue
            self.result["placement_search"].update(selected_offset_m=option["offset_m"],
                selected_object_pose=option["object_pose"].tolist(), selected_eef_pose=option["eef_pose"].tolist())
            self.controller.follow_path(reply["path_file"], phase="place", gripper=1, allow_finger_target=True)
            return reply["path_file"]
        raise ExecutionError("No collision-free release found in container placement search",
            code="container_release_search_failed", category="planning", phase="place",
            details=self.result["placement_search"])

    def place_in_container_static(self, options, attempts):
        """Check both candidate segments before physically moving the held item."""
        import shutil
        monitor = self.controller.monitor
        self.result['placement_search']['policy'] = 'static_two_segment_preflight_then_execute'
        for i, option in enumerate(options):
            directory = self.output/'plans'/'place'/('release_%02d' % i)
            before = self.adapter.step_count
            current = np.asarray(self.adapter.state()['world_T_eef'])
            above = option['eef_pose'].copy()
            above[2, 3] = max(current[2, 3], above[2, 3] + .02)
            progress('place_candidate_static', index=i, offset_m=option['offset_m'],
                     shift_world_m=option['shift_world_m'])
            reposition = plan_cartesian(self.adapter, monitor, above, directory/'reposition',
                                        allow_finger_target=True)
            entry = dict(index=i, offset_m=option['offset_m'], shift_world_m=option['shift_world_m'],
                         reposition_success=bool(reposition['success']), physical_steps_during_search=0)
            attempts.append(entry)
            if reposition.get('state_restored') is not True:
                raise ExecutionError('Candidate preview failed to restore state',
                    code='planning_state_restore_failed', category='technical', phase='place', details=reposition)
            if not reposition['success']:
                entry['reason'] = reposition.get('reason')
                continue
            with preview_path_endpoint(self.adapter, monitor, reposition['path_file']):
                descent = plan_cartesian(self.adapter, monitor, option['eef_pose'], directory/'descent_preview',
                                         allow_finger_target=True)
            entry.update(success=bool(descent['success']), reason=descent.get('reason'),
                         collisions=descent.get('collisions', []))
            if descent.get('state_restored') is not True or self.adapter.step_count != before:
                raise ExecutionError('Candidate continuation modified the physical episode',
                    code='planning_state_restore_failed', category='technical', phase='place', details=entry)
            if not descent['success']:
                continue
            # Execute exactly one chosen reposition. Control failures propagate;
            # they never trigger physical candidate hopping or state rollback.
            self.controller.follow_path(reposition['path_file'], phase='region_reposition',
                                        gripper=1, allow_finger_target=True)
            saved = self.output/'plans'/('container_shift_%03d' % len(self.container_reposition_paths))/'PATH.npz'
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reposition['path_file'], saved)
            self.container_reposition_paths.append(str(saved))
            # Recompute from measured post-execution joints, not the preview.
            fresh = plan_cartesian(self.adapter, monitor, option['eef_pose'], directory/'descent_actual',
                                   allow_finger_target=True)
            if fresh.get('state_restored') is not True or not fresh['success']:
                raise ExecutionError('Chosen container descent failed from measured state',
                    code='motion_plan_failed', category='planning', phase='place', details=fresh)
            self.result['placement_search'].update(selected_offset_m=option['offset_m'],
                selected_object_pose=option['object_pose'].tolist(), selected_eef_pose=option['eef_pose'].tolist())
            self.controller.follow_path(fresh['path_file'], phase='place', gripper=1, allow_finger_target=True)
            return fresh['path_file']
        raise ExecutionError('No collision-free paired reposition and release candidate',
            code='container_release_search_failed', category='planning', phase='place',
            details=self.result['placement_search'])

    def place_above_support(self, object_goal, relation, destination_geoms):
        """Use a bounded gravity release if low placement collides with a rim.

        This is not allowed to excuse a robot collision or a tracking timeout.
        Only failed restored previews are retried; the final official predicate
        must still hold after physical release and settling.
        """
        attempts = []
        self.result["surface_release_search"] = dict(attempts=attempts,
            offsets_m=[0., .02, .04, .06, .08], robot_collision_exemptions_added=[])
        for i, dz in enumerate([0., .02, .04, .06, .08]):
            pose = np.asarray(object_goal).copy()
            pose[2, 3] += dz
            before = self.adapter.step_count
            try:
                reply = self.plan_and_execute(pose @ np.linalg.inv(relation), 'place', ignore=destination_geoms)
            except ExecutionError as exc:
                attempts.append(dict(offset_m=dz, failure=exc.record()))
                if exc.code != 'motion_plan_failed' or self.adapter.step_count != before or not exc.details.get('state_restored'):
                    raise
                import shutil
                source = self.output/'plans'/'place'
                if source.exists():
                    shutil.copytree(source, self.output/'plans'/('surface_rejected_%02d' % i))
                continue
            attempts.append(dict(offset_m=dz, success=True))
            return pose, reply['path_file']
        raise ExecutionError('No collision-free surface release in bounded height search',
            code='surface_release_search_failed', category='planning', phase='place', details={'attempts': attempts})

    def recover_native_descent(self, selection_reply, candidates, round_id):
        audits = self.result.setdefault("native_descent_search", [])
        for attempt in selection_reply.get("attempts", [])[:int(self.config.get("native_candidates_per_round", 8))]:
            approach = attempt.get("approach", {})
            if attempt.get("descend", {}).get("success"):
                continue
            index = int(attempt["candidate_index"])
            candidate = candidates[index]
            directory = self.output / "selection" / ("native_round_%d_candidate_%d" % (round_id,index))
            if not approach.get("success"):
                pre = np.asarray(candidate["world_T_eef"],dtype=float).copy()
                pre[:3,3] -= float(selection_reply.get("approach_distance_m",.10))*pre[:3,2]
                approach = plan_cartesian(self.adapter,self.controller.monitor,pre,
                    directory / "approach",allow_finger_target=False)
                if approach.get("state_restored") is not True:
                    raise ExecutionError("Approach preview did not restore state",
                        code="planning_state_restore_failed",category="technical",phase="select_candidate",details=approach)
                if not approach["success"]:
                    audits.append(dict(round=round_id,candidate_index=index,
                        generation_index=candidate["generation_index"],native_approach=approach))
                    continue
            with np.load(approach["path_file"], allow_pickle=False) as saved:
                q_last = saved["q_arm"][-1].copy()
            sim = self.adapter.env.sim
            snapshot = _snapshot(sim)
            try:
                sim.data.qpos[self.adapter.robot._ref_joint_pos_indexes] = q_last
                sim.forward()
                reply = plan_cartesian(self.adapter, self.controller.monitor,
                    np.asarray(candidate["world_T_eef"]), directory, allow_finger_target=True)
            finally:
                _restore(sim, snapshot)
            audit = dict(round=round_id,candidate_index=index,generation_index=candidate["generation_index"],
                         native=reply, collision_exemptions_added=[])
            audits.append(audit)
            if not reply["success"]:
                continue
            for path, contact in ((approach["path_file"],False),(reply["path_file"],True)):
                with np.load(path,allow_pickle=False) as saved:
                    check = self.controller.monitor.validate_path(saved["q_arm"],contact)
                if not check["passed"]:
                    audit["preflight_rejected"] = check
                    break
            else:
                if self.config.get("bilateral_grasp_screen", True):
                    with np.load(reply["path_file"],allow_pickle=False) as saved:
                        quality=preview_grasp_contacts(self.adapter,self.controller.monitor,saved["q_arm"][-1])
                    audit["bilateral_contacts"]=quality
                    if not quality["passed"]:
                        continue
                audit["selected"] = True
                return dict(success=True,selected=candidate,selected_index=index,
                            approach_path_file=approach["path_file"],descend_path_file=reply["path_file"],
                            path_file=approach["path_file"],fallback="actual_MuJoCo_native_descent",
                            robot_collision_exemptions_added=[])
        return None

    def select_candidate(self, candidates):
        distances = list(self.config.get("approach_distances_m", [.10,.05]))
        if not distances or any(not np.isfinite(d) or not .01 <= d <= .30 for d in distances):
            raise ValueError("Invalid pregrasp approach distances")
        pools = {float(d):list(candidates) for d in distances}
        remaining = list(candidates)
        rejections = []
        for round_id in range(int(self.config.get("selection_rounds", 3))):
            distance = float(distances[round_id % len(distances)])
            remaining = pools[distance]
            if not remaining:
                continue
            progress("selecting_grasp", round=round_id, candidates=len(remaining),approach_distance_m=distance)
            maximum = min(int(self.config.get("max_candidates_per_round", 20)), len(remaining))
            result = self.planner.request({"command": "select", "scene": self.scene(), "candidates": remaining,
                "out_dir": str(self.output / "selection" / f"round_{round_id}"), "max_candidates": maximum,
                "approach_distance_m": distance, "allow_target_contact": True,
                "selection_timeout_s": float(self.config.get("selection_timeout_s", 360)),
                "timeout_s": float(self.config.get("plan_timeout_s", 12))}, timeout=1200)
            if not result["success"]:
                result["approach_distance_m"] = distance
                rejections.append(result)
                if self.config.get("native_descent_fallback", True):
                    recovered = self.recover_native_descent(result, remaining, round_id)
                    if recovered is not None:
                        write_json(self.output / "selection_rejections.json", rejections)
                        write_json(self.output / "selected.json", recovered)
                        return recovered
                attempted = set(result.get("attempted_indices", range(maximum)))
                remaining = [candidate for i, candidate in enumerate(remaining) if i not in attempted]
                pools[distance] = remaining
                continue
            passed = True
            for key, contact in [("approach_path_file", False), ("descend_path_file", True)]:
                with np.load(result[key], allow_pickle=False) as d:
                    check = self.controller.monitor.validate_path(d["q_arm"], contact)
                if not check["passed"]:
                    passed = False
                    rejections.append({"candidate": result["selected"], "path": key, "preflight": check})
                    break
            if passed:
                if self.config.get("bilateral_grasp_screen", True):
                    with np.load(result["descend_path_file"],allow_pickle=False) as saved:
                        quality=preview_grasp_contacts(self.adapter,self.controller.monitor,saved["q_arm"][-1])
                    result["bilateral_contacts"]=quality
                    if not quality["passed"]:
                        rejections.append(dict(candidate=result["selected"],bilateral_contacts=quality))
                        remaining=remaining[int(result["selected_index"])+1:]
                        pools[distance]=remaining
                        continue
                write_json(self.output / "selection_rejections.json", rejections)
                write_json(self.output / "selected.json", result)
                return result
            remaining = remaining[int(result["selected_index"])+1:]
            pools[distance] = remaining
        write_json(self.output / "selection_rejections.json", rejections)
        budget_hit = any(r.get("reason") == "selection_budget_exhausted" for r in rejections)
        self.result["selection_audit"] = dict(screened_candidates=len(candidates),
            unchecked_candidates=len({c["generation_index"] for pool in pools.values() for c in pool}),
            approach_distances_m=distances, rounds=len(rejections), budget_exhausted=budget_hit)
        raise ExecutionError("No executable path found in checked candidates; grasp not executed",
                             code="selection_budget_exhausted" if budget_hit or self.result["selection_audit"]["unchecked_candidates"] else "checked_candidates_rejected",
                             category="planning", phase="select_candidate", details=self.result["selection_audit"])


def execute(config, run_dir, doctor=False):
    return PickPlace(config, run_dir).run(doctor=doctor)
