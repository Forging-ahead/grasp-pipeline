#!/usr/bin/env python3
"""Create a contract reference from existing LIBERO records. Never runs an env.

Raw RGB-D/calibration not saved in the old delivery remains explicitly missing.
Mesh paths are preserved as external references, not asserted to exist locally.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.contracts.validation import success_fields

KNOWN_ORDER_ISSUE = "libero_90_LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray"


def read(path):
    return json.loads(path.read_text())


def artifact(path, media_type, description):
    path = path.resolve()
    return dict(uri=str(path), media_type=media_type, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                availability="local", description=description)


def source(types, evidence, description):
    return dict(types=types, evidence=evidence, description=description)


def payload(kind, **kwargs):
    return dict(schema_version="1.0", payload_type=kind, **kwargs)


def pose_from_wxyz(values):
    x, y, z, w, a, b, c = map(float, values)
    norm = math.sqrt(w*w+a*a+b*b+c*c)
    w, a, b, c = [v/norm for v in [w, a, b, c]]
    return [[1-2*(b*b+c*c), 2*(a*b-c*w), 2*(a*c+b*w), x],
            [2*(a*b+c*w), 1-2*(a*a+c*c), 2*(b*c-a*w), y],
            [2*(a*c-b*w), 2*(b*c+a*w), 1-2*(a*a+b*b), z], [0., 0., 0., 1.]]


def outcome(case):
    result = read(case / "RESULT.json")
    key = result["key"]
    failure = None
    if result["status"] != "completed":
        raw = result.get("failure") or {}
        failure = dict(domain="unknown", code=str(raw.get("code") or result["status"]),
                       phase=raw.get("phase"), message=str(result.get("reason") or result["status"]),
                       details={"native_status": result["status"], "original_failure": raw})
    return payload("TaskOutcome", episode_id=key, pipeline_complete=result["status"] == "completed",
                   native_success=result.get("official_goal_at_end"), steps_executed=int(result.get("steps") or 0),
                   instruction_order="violated" if key == KNOWN_ORDER_ISSUE else "not_checked",
                   trajectory_quality="not_checked", observation_records_complete=False, failure=failure,
                   source_record=artifact(case / "RESULT.json", "application/json", "Original final v10 result; unchanged"))


def export(case):
    import numpy as np
    states = read(case / "states.json")
    state = states[0]
    env = read(case / "environment/ENVIRONMENT.json")
    cfg = read(case / "stage_00/config.json")
    original_scene = read(case / "stage_00/initial_scene.json")
    result = read(case / "RESULT.json")
    key = result["key"]
    xml = ET.parse(case / "environment/merged_model.xml").getroot()
    joints = {j.get("name"): j for j in xml.iter("joint") if j.get("name")}
    config_source = source(["configuration", "simulator_truth"], [str(case / "environment/merged_model.xml")], "Recorded simulator model and configuration")
    truth = source(["simulator_truth"], [str(case / "stage_00/initial_scene.json")], "Original exported compiled collision world and entity poses")

    def joint_spec(name):
        joint = joints[name]
        kind = "prismatic" if joint.get("type") == "slide" else "revolute"
        limits = list(map(float, joint.get("range").split()))
        return dict(name=name, kind=kind, units="m" if kind == "prismatic" else "rad", lower=limits[0], upper=limits[1])

    tcp_shift = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., cfg["grasp_control_offset_m"]], [0., 0., 0., 1.]]
    robot = payload("RobotSpec", robot_id="robot0", robot_model="robosuite_panda", model_ref=artifact(case / "environment/merged_model.xml", "application/xml", "Recorded merged MJCF; asset dependencies are external"),
                    model_format="mujoco_mjcf", base_frame=state["base_body"], tcp_frame=state["eef_site"],
                    arm_joints=[joint_spec(n) for n in state["joint_names"]], finger_joints=[joint_spec(n) for n in state["finger_joint_names"]],
                    collision_geometry_ids=[g.get("name") for g in xml.iter("geom") if (g.get("name") or "").startswith(("robot0_", "gripper0_"))],
                    gripper_model="franka_panda", grasp_T_tcp=tcp_shift, action_encoding="osc_pose_normalized_delta_xyz_rotvec_gripper",
                    action_dimension=7, control_hz=env["control_freq"], source=config_source)
    robot_state = dict(robot_id="robot0", joint_names=state["joint_names"], q=state["q_arm"], qvel=state["qvel_arm"],
                       finger_joint_names=state["finger_joint_names"], finger_q=state["q_fingers"],
                       world_T_base=state["world_T_base"], world_T_tcp=state["world_T_eef"],
                       source=source(["robot_state"], [str(case / "states.json") + "#/0"], "Recorded measured initial robot state"))
    with np.load(case / "stage_00/input_points.npz", allow_pickle=False) as data:
        points = data["points_world"].tolist()
    cloud = dict(target_entity_id=cfg["target_object"], frame="world", units="m", points=points,
                 sample_count=len(points), partial=True, capture_state_index=None,
                 source=source(["sensor", "simulator_truth"], [str(case / "stage_00/perception.json"), str(case / "stage_00/input_points.npz")],
                               "Visible RGB-D target cloud; instance mask is simulator truth; old metadata has no exact acquisition step"))
    cameras = [dict(camera_id=name, rgb=None, depth_m=None, instance_mask=None, intrinsic=None, world_T_camera=None,
                    convention="unknown", capture_state_index=None,
                    unavailable_reason="The old delivery retained composite video, not per-step raw RGB-D, masks and calibration",
                    source=source(["sensor"], [str(case / "environment/ENVIRONMENT.json")], "Camera names only; raw observation payload unavailable"))
               for name in env["camera_names"]]
    observation = payload("Observation", episode_id=key, observation_id=key + ":state0", state_index=0,
                          sim_time_s=state["sim_time"], observation_mode="teacher", robot_state=robot_state, cameras=cameras, target_cloud=cloud)
    goals = result["original_goals"]
    entities = sorted(state["objects"])
    regions = sorted({str(arg) for goal in goals for arg in goal[1:] if arg not in entities})
    task = payload("TaskSpec", benchmark="libero", task_id=key, language=env["language"],
                   definition_ref=artifact(case / "RESULT.json", "application/json", "Original goal list is in original_goals; native BDDL path recorded in ENVIRONMENT.json"),
                   entity_ids=entities, region_ids=regions, goal_logic="all",
                   goals=[dict(goal_id="goal_" + str(i), predicate=str(g[0]), arguments=list(g[1:]),
                               source=source(["task_definition"], [str(case / "RESULT.json") + "#/original_goals/" + str(i)], "Original unmodified native goal")) for i, g in enumerate(goals)],
                   required_order=[], instruction_constraints_complete=False)
    geometries = []
    for geom in original_scene["obstacles"] + original_scene["target"]["geometries"]:
        params = {}
        if geom["type"] == "box": params["dims_m"] = geom["dims"]
        if geom["type"] in ("sphere", "cylinder"): params["radius_m"] = geom["radius"]
        if geom["type"] == "cylinder": params["length_m"] = geom["length"]
        if geom["type"] == "mesh":
            params.update(mesh=dict(uri=geom["mesh_path"], media_type="model/obj", sha256=None, availability="external",
                                    description="Original collision mesh path; delivery does not bundle the referenced file"), scale=geom["scale"])
        geometries.append(dict(geometry_id=geom["name"], owner_entity_id=cfg["target_object"] if geom in original_scene["target"]["geometries"] else None,
                               world_T_geometry=pose_from_wxyz(geom["pose"]), shape=geom["type"], parameters=params,
                               representation=geom.get("representation", "original_exported_primitive"), source=truth))
    gravity_text = xml.find("option").get("gravity")
    scene = payload("SceneSnapshot", episode_id=key, scene_id=key + ":initial_scene", state_index=original_scene.get("step_count"),
                    gravity_world_m_s2=list(map(float, gravity_text.split())) if gravity_text else None,
                    coverage="complete_simulator_collision_world", unknown_space_policy="not_applicable", geometries=geometries,
                    entity_poses=[dict(entity_id=n, world_T_object=p, source=truth) for n, p in original_scene["objects"].items()], regions=[], articulations=[], source=truth)
    features = {k: True for k in ["target_cloud", "robot_state", "robot_model", "joint_limits", "object_pose", "collision_geometry", "goal_region",
                                  "local_ik", "contacts", "preview_restore", "tcp_tracking", "gripper_control", "native_success", "articulation_state"]}
    capabilities = payload("Capabilities", benchmark="libero", binding_status="reference_mapping_only", features=features,
                           available_skills=["pick_place", "articulate", "push"],
                           description="Capabilities available in existing LIBERO implementation, not a new live port binding")
    with np.load(case / "trajectory.npz", allow_pickle=False) as trajectory:
        action_count, phase_count = len(trajectory["actions"]), len(trajectory["phases"])
    recording = payload("TrajectoryRecord", episode_id=key, state_count=len(states), action_count=action_count, phase_count=phase_count,
                        states_ref=artifact(case / "states.json", "application/json", "N+1 measured states"),
                        trajectory_ref=artifact(case / "trajectory.npz", "application/x-npz", "N actions and phase labels"),
                        video_ref=artifact(case / "video.mp4", "video/mp4", "Composite report video includes final-frame hold"),
                        keypoints=[dict(action_index=p["action_index"], state_index=p["action_index"] + 1, phase=p["phase"], world_T_tcp=p["world_T_eef"],
                                        pose_source="measured_post_action", label_source="code_policy", decision_record=None) for p in read(case / "KEYPOINTS.json")],
                        raw_observations_complete=False, action_label_source="code_policy", notes="Existing recording is not a complete raw-observation or VLM-decision SFT dataset")
    return payload("ReferenceBundle", observation=observation, robot=robot, task=task, scene=scene, capabilities=capabilities,
                   outcome=outcome(case), recording=recording,
                   limitations=["Offline reference only; no live ports registered", "Raw camera arrays and calibration not retained", "Point-cloud capture state index unknown",
                                "Goal-region geometry not exported here", "Gravity was not explicit in saved XML; default not fabricated",
                                "Instruction-order completeness and training quality not certified"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=ROOT.parent / "outputs/libero-final130-four4090-20261111/delivery")
    parser.add_argument("--output", type=Path, default=ROOT / "contracts/examples")
    args = parser.parse_args()
    case = args.delivery / "cases/libero_object_pick_up_the_orange_juice_and_place_it_in_the_basket"
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = export(case)
    (args.output / "libero_v10_reference.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    results = [outcome(p) for p in sorted((args.delivery / "cases").iterdir()) if (p / "RESULT.json").exists()]
    (args.output / "libero_v10_outcomes.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    counts = dict(tasks=len(results), native_success=sum(r["native_success"] is True for r in results),
                  full_pipeline_success=sum(success_fields(r)["full_pipeline_success"] for r in results),
                  candidate_success=sum(success_fields(r)["candidate_success"] for r in results),
                  sft_eligible=sum(success_fields(r)["sft_eligible"] for r in results), source="Unmodified existing v10 records; no new execution")
    (args.output / "EXPECTED_COUNTS.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
