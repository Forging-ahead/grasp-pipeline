"""RoboCasa extensions around the unchanged LIBERO cuRobo worker."""
import argparse
import copy
import time
from pathlib import Path

import numpy as np

from graspgen_libero_bridge.common.rpc import serve
from graspgen_libero_bridge.planning.planner_worker import Handler, arm_array, densify, write_json, ARM_NAMES


class GraphOnlyMotionGen:
    """Use the vendor's public graph mode while retaining the original checks."""
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def plan_single(self, start, goal, plan_config):
        config = copy.copy(plan_config)
        config.enable_graph = True
        config.enable_opt = False
        config.enable_graph_attempt = None
        config.enable_finetune_trajopt = False
        return self.wrapped.plan_single(start, goal, config)

    def plan_single_js(self, start, goal, plan_config):
        config = copy.copy(plan_config)
        config.enable_graph = True
        config.enable_opt = False
        config.enable_graph_attempt = None
        config.enable_finetune_trajopt = False
        return self.wrapped.plan_single_js(start, goal, config)


def plan_joint(planner, request):
    q_goal = arm_array(request["q_goal"])
    limits = np.asarray(planner.robot.arm_limits)
    if np.any(q_goal < limits[:, 0]) or np.any(q_goal > limits[:, 1]):
        raise ValueError("Joint target outside robot limits")
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    scene = request["scene"]
    planner.sync(scene, bool(request.get("allow_target_contact", False)))
    directory = Path(request["out_dir"]); directory.mkdir(parents=True, exist_ok=True)
    for name, value in [("WORLD", planner.world_data), ("SCENE", scene), ("CONTACT_POLICY", planner.world_report),
                        ("JOINT_GOAL", {"joint_names": ARM_NAMES, "q_arm": q_goal.tolist()})]:
        write_json(directory / (name + ".json"), value)
    start = arm_array(scene["q_arm"])
    def state(q):
        return JointState.from_position(planner.torch.as_tensor(q[None], device="cuda", dtype=planner.torch.float32), joint_names=ARM_NAMES)
    current, target = state(start), state(q_goal)
    planner._enable_collision_costs()
    try:
        valid, status = planner.motion_gen.check_start_state(current)
    finally:
        planner._enable_collision_costs()
    if not valid:
        return {"success": False, "reason": "invalid_start_state", "status": str(status)}
    started = time.monotonic()
    try:
        result = planner.motion_gen.plan_single_js(current, target,
            MotionGenPlanConfig(max_attempts=2, timeout=float(request["timeout_s"]),
                                enable_graph_attempt=1, check_start_validity=False))
    finally:
        planner._enable_collision_costs()
    planner.torch.cuda.synchronize()
    reply = {"success": bool(result.success.item()), "status": str(result.status),
             "wall_s": time.monotonic() - started, "contact_policy": planner.world_report,
             "goal_kind": "joint", "algorithm": "curobo_joint_trajopt"}
    if reply["success"]:
        q = densify(result.get_interpolated_plan().position.detach().cpu().numpy())
        if np.max(np.abs(q[0] - start)) > .001:
            q = densify(np.concatenate([start[None], q]))
        checked = planner.validate_path(q)
        error = float(np.max(np.abs(q[-1] - q_goal)))
        reply.update(path_check=checked, final_joint_error_rad=error)
        if not checked["valid"] or error > .005:
            reply.update(success=False, reason="interpolated_path_validation_failed")
        else:
            path = directory / "PATH.npz"
            np.savez_compressed(path, q_arm=q, world_T_eef=planner.fk_world(q))
            reply.update(path_file=str(path), joint_names=ARM_NAMES)
    else:
        reply["reason"] = "curobo_plan_failed"
    write_json(directory / "RESULT.json", reply)
    return reply


class RoboCasaHandler(Handler):
    def __call__(self, request):
        if request.get("command") in {"plan_pose_graph", "plan_joint_graph"}:
            if self.planner is None:
                raise RuntimeError("configure the planner before planning")
            # The RPC server is synchronous. Replace only the bridge-owned
            # reference for this request; the vendor model is never patched.
            original = self.planner.motion_gen
            self.planner.motion_gen = GraphOnlyMotionGen(original)
            try:
                joint = request["command"] == "plan_joint_graph"
                reply = plan_joint(self.planner, request) if joint else self.planner.plan_pose(request)
                return dict(reply, algorithm="curobo_joint_graph_only" if joint else "curobo_graph_only")
            finally:
                self.planner.motion_gen = original
        if request.get("command") == "plan_joint":
            if self.planner is None:
                raise RuntimeError("configure the planner before planning")
            return plan_joint(self.planner, request)
        if request.get("command") != "check_joint_path":
            return super().__call__(request)
        if self.planner is None:
            raise RuntimeError("configure the planner before checking a path")
        joints = np.asarray(request["q_arm"], dtype=float)
        if joints.ndim != 2 or joints.shape[1] != 7 or not len(joints) or not np.isfinite(joints).all():
            raise ValueError("Expected a finite Nx7 joint path")
        self.planner.sync(request["scene"], bool(request.get("allow_target_contact", False)))
        return dict(self.planner.validate_path(joints), backend="curobo_collision_constraints",
                    contact_policy=self.planner.world_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--robosuite-assets", type=Path, required=True)
    args = parser.parse_args()
    serve(RoboCasaHandler(assets_dir=args.robosuite_assets), args.ready)


if __name__ == "__main__":
    main()
