"""Use the mature v10 selector itself, with native-environment bindings only."""
from __future__ import annotations

from types import SimpleNamespace
import inspect
import hashlib
import numpy as np

from graspgen_libero_bridge.common.errors import ExecutionError
from graspgen_libero_bridge.policy.evaluate import PickPlace
from graspgen_libero_bridge.adapters.robocasa.v10_preview import V10PreviewAdapter, V10PreviewMonitor
from graspgen_libero_bridge.adapters.robocasa.native import write_json
from graspgen_libero_bridge.common.layout import project_root


class NativePlannerClient:
    def __init__(self, planner, target):
        self.planner, self.target = planner, target

    def request(self, request, timeout=None):
        self.planner.ensure_configuration(self.target)
        payload = dict(request)
        if "selection_timeout_s" in payload:
            payload["selection_timeout_s"] = max(1., min(payload["selection_timeout_s"],
                                                    self.planner.environment.remaining_seconds()))
        try:
            return self.planner._request(payload)
        finally:
            self.planner.local.sync()


class V10Controller:
    def __init__(self, executor, adapter):
        self.executor, self.adapter = executor, adapter
        self.monitor = V10PreviewMonitor(adapter)

    def follow_path(self, path_file, *, phase, gripper=-1, allow_finger_target=False):
        from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
        original = self.adapter.local.allow_finger_target
        try:
            self.adapter.local.allow_finger_target = bool(allow_finger_target)
            try:
                return self.executor.follow_path(path_file, phase=phase, gripper=gripper)
            except EpisodeFailure as exc:
                if exc.code == "PATH_PREFLIGHT_REJECTED":
                    raise ExecutionError(str(exc), code="path_preflight_rejected", category="planning",
                                         phase=phase, details=exc.details) from exc
                raise
        finally:
            self.adapter.local.allow_finger_target = original
            self.adapter.local.sync()


class V10Selection(PickPlace):
    """Inherit selection and static paired placement, without copying them."""
    def __init__(self, environment, local, planner, target, output, executor=None):
        self.environment, self.native_planner, self.target = environment, planner, target
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = dict(environment.config["policy"]["libero_v10_rules"])
        self.result = {}
        self.adapter = V10PreviewAdapter(environment, local, target)
        self.controller = V10Controller(executor, self.adapter)
        self.planner = NativePlannerClient(planner, target)

    def scene(self):
        return self.native_planner.scene(self.target)

    def choose(self, candidates):
        self.adapter.local.sync()
        before = self.environment.steps
        try:
            return self.select_candidate(candidates)
        finally:
            write_json(self.output / "V10_RULE_AUDIT.json", {
                "implementation": "inherited PickPlace.select_candidate and recover_native_descent",
                "select_source_sha256": hashlib.sha256(inspect.getsource(PickPlace.select_candidate).encode()).hexdigest(),
                "recover_source_sha256": hashlib.sha256(inspect.getsource(PickPlace.recover_native_descent).encode()).hexdigest(),
                "effective_rules": self.config, "physical_steps_during_selection": self.environment.steps - before,
                "isolated_preview": True, "selection_result": self.result})
            if self.environment.steps != before:
                raise ExecutionError("Candidate search advanced the real environment", code="planning_state_restore_failed")

    def place_checked_options(self, options):
        self.adapter.local.sync()
        self.result["placement_search"] = {"attempts": []}
        self.container_reposition_paths = []
        try:
            return self.place_in_container_static(options, self.result["placement_search"]["attempts"])
        finally:
            write_json(self.output / "V10_PLACEMENT_AUDIT.json", {
                "implementation": "inherited PickPlace.place_in_container_static",
                "source_sha256": hashlib.sha256(inspect.getsource(PickPlace.place_in_container_static).encode()).hexdigest(),
                "placement_search": self.result["placement_search"]})

    def plan_and_execute(self, goal, phase, *, ignore=(), allow_contact=True, gripper=1):
        """The v10 transfer search calls this environment-specific boundary."""
        from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
        self.controller.executor.ensure_joint_margin(phase, gripper=gripper)
        if hasattr(self, "active_region"):
            from graspgen_libero_bridge.adapters.robocasa.goals import fits_native_opening
            raised_object = np.asarray(goal) @ self.relation
            if not fits_native_opening(raised_object, self.object_points, self.active_region):
                raise ExecutionError("Footprint misses native destination opening", code="motion_plan_failed",
                                     category="planning", phase=phase,
                                     details={"state_restored": True, "reason": "native_opening_mismatch"})
        reply = transfer_preflight(self.native_planner, self.target, goal, label=phase)
        if not reply.get("success"):
            raise ExecutionError("Native-bound v10 planning failed", code="motion_plan_failed",
                                 category="planning", phase=phase, details=reply)
        if self.controller.executor.recorder is not None:
            self.controller.executor.recorder.phase = "place_decision"
            self.controller.executor.recorder.decision("place_object", {"target": self.target,
                "destination": self.stage["destination"], "world_T_preplace_tcp": goal,
                "world_T_object_above_destination": np.asarray(goal) @ self.relation,
                "native_region": self.active_region["name"]})
        self.controller.follow_path(reply["path_file"], phase=phase, gripper=gripper,
                                    allow_finger_target=allow_contact)
        return reply


def record_rule_sources(output):
    """Leave a reviewable record of which mature methods are actually reused."""
    from graspgen_libero_bridge.planning.cartesian import plan_cartesian
    from graspgen_libero_bridge.planning.grasp_contact import preview_grasp_contacts
    from graspgen_libero_bridge.policy.placement import container_release_candidates
    from graspgen_libero_bridge.policy.transfer import transfer_candidates
    from graspgen_libero_bridge.control.controller import Controller
    functions = (PickPlace.select_candidate, PickPlace.recover_native_descent,
                 PickPlace.transfer_with_alternatives, PickPlace.place_in_container_static,
                 plan_cartesian, preview_grasp_contacts, container_release_candidates, transfer_candidates, Controller.move_to)
    write_json(output / "LIBERO_V10_RULE_SOURCES.json", {
        "reused_functions": [{"module": f.__module__, "function": f.__qualname__,
                              "sha256": hashlib.sha256(inspect.getsource(f).encode()).hexdigest()} for f in functions],
        "native_bindings": ["robot joint/TCP mapping", "isolated MuJoCo model", "native destination regions",
                            "composite action layout", "native success/horizon"],
        "v10_max_steps_5000_inherited": False,
        "contact_backend": "unchanged v10 plan_cartesian with real geometry; no sphere veto on local contact paths"})


def transfer_preflight(planner, target, goal, *, label):
    """v10: global transport, then actual-geometry local fallback before motion."""
    reply = planner.global_plan(target, goal, label=label)
    if (not reply.get("success") and reply.get("reason") in {"invalid_start_state", "curobo_plan_failed"}
            and "JOINT_LIMIT" not in str(reply.get("status", ""))):
        rejected = reply
        reply = planner.contact_plan(goal, label=label + "_native_fallback")
        reply["global_failure"] = rejected
    return reply


def execute_selected_grasp(environment, executor, planner, selected):
    """v10 measured-branch recovery; no physical rollback or candidate hopping."""
    from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
    regenerate = False
    try:
        executor.follow_path(selected["approach_path_file"], phase="pregrasp", gripper=-1)
    except EpisodeFailure as exc:
        if exc.code != "JOINT_BRANCH_DEVIATION":
            raise
        with np.load(selected["approach_path_file"], allow_pickle=False) as path:
            goal = path["world_T_eef"][-1]
        reply = transfer_preflight(planner, planner.local.target, goal, label="approach_replan")
        if not reply["success"]:
            raise EpisodeFailure("SEARCH_BUDGET_EXHAUSTED", "Measured-branch approach replan failed", details=reply)
        executor.follow_path(reply["path_file"], phase="approach_replan", gripper=-1)
        regenerate = True
    planner.local.allow_finger_target = True
    if not regenerate:
        try:
            executor.follow_path(selected["descend_path_file"], phase="approach", gripper=-1)
            return
        except EpisodeFailure as exc:
            if exc.code != "PATH_PREFLIGHT_REJECTED":
                raise
    executor.ensure_joint_margin("descend", gripper=-1)
    reply = planner.contact_plan(selected["selected"]["world_T_eef"], label="descend_measured")
    if not reply["success"]:
        raise EpisodeFailure("COLLISION_REJECTED", "Measured-state descent rejected", details=reply)
    executor.follow_path(reply["path_file"], phase="approach", gripper=-1)


def place_v10(environment, executor, planner, stage, carried, output):
    """Use v10 transfer, opening/height search and paired static placement."""
    from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
    from graspgen_libero_bridge.adapters.robocasa.goals import v10_placement_regions, fits_native_opening
    from graspgen_libero_bridge.adapters.libero.placement_geometry import orientation_candidates, opening_offsets
    from graspgen_libero_bridge.policy.placement import container_release_candidates
    job = carried["v10_job"]
    job.stage, job.relation, job.object_points = stage, carried["tcp_T_object"], carried["object_points"]
    audits = []
    selected_region = None
    for region in v10_placement_regions(environment, stage):
        environment.check_budget()
        job.active_region = region
        job.opening_constraint = (job.object_points, region["frame"], region["half_size"])
        pose = carried["initial_object_pose"].copy()
        orientations = orientation_candidates(pose, job.object_points, region["frame"], region["half_size"],
                                               region["surface_height_m"], require_origin_containment=False)
        fitting = [p for p in orientations if fits_native_opening(p, job.object_points, region)]
        orientation_limit = job.config["placement_orientation_trials"]
        job.placement_object_options = fitting[:orientation_limit]
        audits.append({"region": region, "fitting_orientation_count": len(fitting),
                       "tried_orientation_limit": orientation_limit,
                       "selected_orientation_count": len(job.placement_object_options)})
        if not job.placement_object_options:
            continue
        before = environment.steps
        try:
            object_goal, transfer = job.transfer_with_alternatives(job.placement_object_options[0], job.relation)
        except ExecutionError as exc:
            audits[-1]["transfer_failure"] = exc.record()
            if environment.steps != before or exc.code not in {"motion_plan_failed", "path_preflight_rejected"}:
                raise
            continue
        selected_region = region
        break
    write_json(output / "V10_REGION_SEARCH.json", audits)
    if selected_region is None:
        raise EpisodeFailure("SEARCH_BUDGET_EXHAUSTED", "No native region/orientation transfer found", details={"regions": audits})
    write_json(output / "V10_TRANSFER_AUDIT.json", job.result["transfer_search"])
    local = planner.local
    local.allowed_support_geoms = set(selected_region["support_geom_ids"])
    executor.ensure_joint_margin("place", gripper=1)
    shifts = opening_offsets(object_goal, job.object_points, selected_region["frame"], selected_region["half_size"])
    offsets = [0., .02, .04, .06, .08, .10, .12] if selected_region["container"] else [0., .02, .04, .06, .08]
    options = []
    for order, shift in enumerate(shifts):
        shifted = object_goal.copy(); shifted[:3, 3] += shift
        for option in container_release_candidates(shifted, job.relation, job.object_points,
                selected_region["frame"], selected_region["half_size"], offsets):
            if fits_native_opening(option["object_pose"], job.object_points, selected_region):
                option.update(shift_world_m=shift.tolist(), shift_order=order)
                options.append(option)
    options.sort(key=lambda option: (option["offset_m"], option["shift_order"]))
    lower_path = job.place_checked_options(options)
    write_json(output / "V10_JOB_AUDIT.json", job.result)
    return {"region": selected_region, "object_goal": object_goal, "transfer_path": transfer["path_file"],
            "placement_path": lower_path, "container_reposition_paths": list(job.container_reposition_paths)}


def withdraw_v10(environment, executor, planner, carried, placed, *, stable_goal):
    """v10 upward retreat and checked path retracing, then native far-goal route."""
    from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
    from graspgen_libero_bridge.policy.placement import can_retrace_after_ik_failure
    local = planner.local
    local.allow_finger_target = True  # v10 permits finger/target separation; palm/world remain checked.
    executor.ensure_joint_margin("retreat", gripper=-1)
    retreat = np.asarray(environment.robot_state()["world_T_eef"])
    retreat[2, 3] += .10
    reply = planner.contact_plan(retreat, label="retreat")
    outcome = {"task_goal_reached": False}
    retrace = []
    if reply["success"]:
        outcome = executor.follow_path(reply["path_file"], phase="retreat", gripper=-1, stop_when=stable_goal)
        # RoboCasa may additionally require the gripper to be far from the
        # released object. Continue away along the validated incoming transport.
        if not outcome.get("task_goal_reached") and not environment.check_native_success():
            retrace = list(reversed(placed["container_reposition_paths"])) + [placed["transfer_path"]]
    elif can_retrace_after_ik_failure(reply):
        retrace = [placed["placement_path"], *reversed(placed["container_reposition_paths"]), placed["transfer_path"]]
    audit = {"primary_retreat": reply, "retrace_paths": retrace, "executed_retrace": [],
             "collision_exemptions_added": []}
    for index, source in enumerate(retrace):
        directory = environment.output / "plans" / ("v10_reverse_" + str(index))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "PATH.npz"
        with np.load(source, allow_pickle=False) as data:
            np.savez_compressed(path, q_arm=data["q_arm"][::-1].copy(), world_T_eef=data["world_T_eef"][::-1].copy())
        before = environment.steps
        try:
            outcome = executor.follow_path(path, phase="reverse_" + str(index), gripper=-1, stop_when=stable_goal)
        except EpisodeFailure as exc:
            audit["retrace_failure"] = {"code": exc.code, "details": exc.details}
            if environment.steps != before or exc.code not in {"PATH_PREFLIGHT_REJECTED", "JOINT_BRANCH_DEVIATION"}:
                raise
            break
        audit["executed_retrace"].append(str(path))
        if outcome.get("task_goal_reached"):
            break
    if not outcome.get("task_goal_reached") and not environment.check_native_success():
        reply = planner.global_joint_plan(local.target, carried["initial_q_arm"], label="release_return_home")
        audit["native_far_goal_plan"] = reply
        if not reply["success"]:
            write_json(environment.output / "V10_WITHDRAWAL.json", audit)
            raise EpisodeFailure(reply.get("reason", "SEARCH_BUDGET_EXHAUSTED"), "No checked withdrawal path", details=reply)
        outcome = executor.follow_path(reply["path_file"], phase="release_return_home", gripper=-1, stop_when=stable_goal)
    audit["outcome"] = outcome
    write_json(environment.output / "V10_WITHDRAWAL.json", audit)
    local.allow_finger_target = False
    return outcome
