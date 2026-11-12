"""Stop-and-go base navigation, separate from the seven-joint cuRobo planner."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.adapters.robocasa.actions import physical_to_input
from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
from graspgen_libero_bridge.adapters.robocasa.native import write_json
from graspgen_libero_bridge.planning.robocasa_navigation import mobile_pose, base_grid_path, preview_base_path


def execute_navigation(environment, executor, local, recorder, target_position, target_orientation):
    env = environment
    if not env.config["policy"]["base_motion_enabled"]:
        raise EpisodeFailure("UNSUPPORTED_SKILL", "Base navigation is disabled pending native validation")
    target_position, target_orientation = np.asarray(target_position, float), np.asarray(target_orientation, float)
    if target_position.shape != (3,) or target_orientation.shape != (3,):
        raise ValueError("Native navigation goal must contain position and Euler orientation")
    points = base_grid_path(env, target_position)
    preview = preview_base_path(env, local, points, float(target_orientation[2]))
    write_json(env.output / "NAVIGATION_PREVIEW.json", preview)
    if not preview["valid"]:
        raise EpisodeFailure("NAVIGATION_FAILED", "Full-robot navigation preview rejected the base route", details=preview)
    initial_base = mobile_pose(env)
    base_T_tcp = np.linalg.inv(initial_base) @ np.asarray(env.robot_state()["world_T_eef"])
    policy = env.config["policy"]
    controller = env.robot.composite_controller.part_controllers["base"]
    recorder.phase = "navigate"
    recorder.decision("navigate_base", {"world_position": target_position,
                    "world_euler": target_orientation, "backend": "grid_astar_and_native_base_velocity",
                    "waypoints": points})
    # The preview covers translation at current yaw, then a rotation at the endpoint.
    initial_yaw = Rotation.from_matrix(initial_base[:3, :3]).as_euler("xyz")[2]
    targets = [(p, initial_yaw, False) for p in points[1:]] + [(points[-1], float(target_orientation[2]), True)]
    for point, yaw, final in targets:
        deadline = env.clock() + min(policy["move_timeout_s"], env.remaining_seconds())
        while True:
            env.check_budget()
            base = mobile_pose(env)
            current_yaw = Rotation.from_matrix(base[:3, :3]).as_euler("xyz")[2]
            error = point - base[:2, 3]
            angular_error = float(np.arctan2(np.sin(yaw - current_yaw), np.cos(yaw - current_yaw)))
            tolerance = policy["navigation_position_tolerance_m"] if final else .04
            if np.linalg.norm(error) < tolerance and abs(angular_error) < policy["navigation_rotation_tolerance_rad"]:
                break
            if env.clock() >= deadline:
                raise EpisodeFailure("NAVIGATION_FAILED", "Base waypoint stopped progressing")
            world_velocity = np.clip(1.5 * error, -policy["base_max_speed_m_s"], policy["base_max_speed_m_s"])
            local_velocity = base[:2, :2].T @ world_velocity
            speed = np.r_[local_velocity, np.clip(1.5 * angular_error, -policy["base_max_yaw_rate_rad_s"], policy["base_max_yaw_rate_rad_s"])]
            base_action = physical_to_input(speed, controller)
            state = env.robot_state()
            goal = base @ base_T_tcp
            action = env.layout.pose_action(goal, state["world_T_eef"], state["world_T_control_base"],
                                           executor.gripper_command, base_velocity=base_action, gain=policy["pose_gain"])
            executor._step(action)
    executor.hold(.3, phase="navigate/stop")
    final_base = mobile_pose(env)
    return {"skill": "navigate", "subgoal_success": True, "backend": "grid_astar_and_native_base_velocity",
            "final_base_pose": final_base.tolist(), "physical_steps": env.steps}
