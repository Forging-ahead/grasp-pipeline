"""Compare the measured PandaOmron arm with the exact Panda planner model."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from graspgen_libero_bridge.planning.planning_geometry import PandaMJCF


def self_collision_pairs(robot, q, fingers):
    config = robot.robot_config(fingers)["kinematics"]
    ignored = config["self_collision_ignore"]
    transforms = robot.fk(q, fingers)
    groups = {}
    for name, spheres in robot.spheres.items():
        points = np.asarray([s["center"] for s in spheres])
        radii = np.asarray([s["radius"] for s in spheres]) + config["collision_sphere_buffer"]
        transform = transforms[name]
        groups[name] = (points @ transform[:3, :3].T + transform[:3, 3], radii)
    pairs = []
    names = list(groups)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if second in ignored.get(first, []) or first in ignored.get(second, []):
                continue
            a, ra = groups[first]; b, rb = groups[second]
            gaps = np.linalg.norm(a[:, None] - b[None], axis=-1) - ra[:, None] - rb[None]
            if gaps.min() < 0:
                pairs.append({"first_link": first, "second_link": second,
                              "minimum_clearance_m": float(gaps.min())})
    return pairs


def prepare_start_state(environment, executor, local):
    """Physically unfold an invalid sphere-proxy start, after exact-mesh preview.

    The retract posture comes from the robot model, shared across all tasks.
    No joint state is overwritten, and no collision constraint is disabled.
    """
    from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
    from graspgen_libero_bridge.adapters.robocasa.native import write_json
    state = environment.robot_state()
    robot = PandaMJCF(environment.output / "robot_start_model",
                      assets_dir=Path(environment.config["paths"]["robosuite_assets"]))
    initial_pairs = self_collision_pairs(robot, state["q_arm"], state["q_fingers"])
    if not initial_pairs:
        return {"performed": False, "initial_proxy_collisions": []}
    if local.check():
        raise EpisodeFailure("START_INVALID", "Real forbidden contacts exist at initialization", details={"contacts": local.check()})
    retract = np.asarray(robot.robot_config(state["q_fingers"])["kinematics"]["cspace"]["retract_config"][:7])
    if self_collision_pairs(robot, retract, state["q_fingers"]):
        raise EpisodeFailure("START_INVALID", "Robot retract posture also violates its planning proxy")
    start = np.asarray(state["q_arm"])
    count = max(2, int(np.ceil(np.max(np.abs(retract - start)) / .025)) + 1)
    joints = np.asarray([(1 - u) * start + u * retract for u in np.linspace(0., 1., count)])
    preview = local.validate_path(joints)
    report = {"performed": False, "initial_proxy_collisions": initial_pairs, "preview": preview,
              "target_source": "robot model retract configuration, not a task or seed-specific target"}
    write_json(environment.output / "START_RECOVERY.json", report)
    if not preview["valid"]:
        raise EpisodeFailure("START_INVALID", "Physical recovery path failed full MuJoCo collision preview", details=report)
    poses = []
    for q in joints:
        local.set_q(q); poses.append(local.tcp())
    path = environment.output / "START_RECOVERY_PATH.npz"
    np.savez_compressed(path, q_arm=joints, world_T_eef=np.asarray(poses))
    executor.follow_path(path, phase="prepare_robot_from_proxy_self_collision", gripper=-1)
    final = environment.robot_state()
    remaining = self_collision_pairs(robot, final["q_arm"], final["q_fingers"])
    report.update(performed=True, final_state=final, final_proxy_collisions=remaining)
    write_json(environment.output / "START_RECOVERY.json", report)
    if remaining:
        raise EpisodeFailure("START_INVALID", "Physical recovery did not clear the planning proxy", details=report)
    return report


def check_robot_model(environment, samples):
    state = environment.robot_state()
    robot = PandaMJCF(environment.output / "robot_model_cpu",
                      assets_dir=Path(environment.config["paths"]["robosuite_assets"]))
    base = np.asarray(state["world_T_base"])
    fingers = np.asarray(state["q_fingers"])
    results = []
    for sample in samples:
        actual = base @ robot.fk(sample["q_arm"], fingers)["control_site"]
        expected = np.asarray(sample["world_T_eef"])
        results.append({"position_error_m": float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])),
                        "rotation_error_rad": float(Rotation.from_matrix(actual[:3, :3].T @ expected[:3, :3]).magnitude())})
    return {"passed": all(r["position_error_m"] <= .002 and r["rotation_error_rad"] <= np.deg2rad(1) for r in results),
            "samples": results, "planner_model": "PandaMJCF from selected robosuite assets",
            "native_arm_base_body": state["arm_base_body"], "native_grip_site": state["eef_site"],
            "world_T_arm_base": base.tolist(), "robot_config_assets": environment.config["paths"]["robosuite_assets"]}
