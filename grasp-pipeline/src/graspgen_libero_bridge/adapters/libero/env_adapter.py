"""Isolated access to an official LIBERO environment.

Importing this module does not import LIBERO, robosuite, MuJoCo, or PyTorch.
Those imports happen only in ``make_env`` / ``reset`` in the simulation process.
Control poses always refer to robosuite's actual grip site, not its wrist body.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
from graspgen_libero_bridge.common.errors import ExecutionError


DEFAULT_TASK = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"


def _pose(position, rotation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def body_subtree_ids(model, root_id):
    """Return a body's descendants without depending on body ID ordering."""
    root_id = int(root_id)
    result = {root_id}
    parents = np.asarray(model.body_parentid, dtype=np.int64)
    for body in range(int(model.nbody)):
        current = body
        visited = set()
        while current not in visited:
            if current == root_id:
                result.add(body)
                break
            visited.add(current)
            parent = int(parents[current])
            if parent == current:
                break
            current = parent
    return result


class EnvAdapter:
    def __init__(self, config: dict, run_dir: Path):
        self.config = dict(config)
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.repo = Path(self.config["libero_repo"]).expanduser().resolve()
        self.target_object = self.config.get("target_object", "akita_black_bowl_1")
        self.destination = self.config.get("destination", "plate_1")
        self.env = None
        self.task_env = None
        self.robot = None
        self.obs = None
        self.task = None
        self.suite = None
        self.task_id = None
        self.step_count = 0
        self.success_seen = False
        self.settle_steps_performed = 0
        self.max_steps = int(self.config.get("max_steps", 1500))
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.config_dir = self.run_dir / "libero_config"
        self.merged_xml_path = self.run_dir / "merged_model.xml"
        self.provenance = {}

    def _configure_imports(self):
        core_path = self.repo / "libero/libero/__init__.py"
        if not core_path.is_file():
            raise FileNotFoundError("Not an official LIBERO source checkout: " + str(self.repo))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = self.run_dir / "unused_datasets"
        dataset_dir.mkdir(exist_ok=True)
        base = self.repo / "libero/libero"
        paths = {
            "benchmark_root": str(base), "bddl_files": str(base / "bddl_files"),
            "init_states": str(base / "init_files"), "assets": str(base / "assets"),
            "datasets": str(dataset_dir),
        }
        # JSON is valid YAML. Write before import to avoid LIBERO's input() prompt.
        (self.config_dir / "config.yaml").write_text(json.dumps(paths, indent=2) + "\n")
        os.environ["LIBERO_CONFIG_PATH"] = str(self.config_dir)
        os.environ.setdefault("MUJOCO_GL", str(self.config.get("mujoco_gl", "egl")))
        for name, module in tuple(sys.modules.items()):
            if name == "libero.libero" or name.startswith("libero.libero."):
                origin = getattr(module, "__file__", None)
                if origin and not Path(origin).resolve().is_relative_to(self.repo):
                    raise RuntimeError("A different LIBERO is already imported; use an isolated simulation process: " + str(origin))
        repo_string = str(self.repo)
        if repo_string in sys.path:
            sys.path.remove(repo_string)
        sys.path.insert(0, repo_string)
        return paths

    def make_env(self):
        if self.env is not None:
            return self.env
        paths = self._configure_imports()
        from libero.libero import benchmark
        import libero.libero as core
        from libero.libero.envs.env_wrapper import SegmentationRenderEnv
        import libero.libero.envs.env_wrapper as wrapper_module
        import mujoco
        import robosuite

        if Path(core.__file__).resolve() != (self.repo / "libero/libero/__init__.py").resolve():
            raise RuntimeError("LIBERO import did not resolve to the selected checkout")
        if Path(core.config_file).resolve() != (self.config_dir / "config.yaml").resolve():
            raise RuntimeError("LIBERO was imported before its isolated configuration was selected")
        suite_name = str(self.config.get("task_suite", "libero_spatial")).lower()
        suites = benchmark.get_benchmark_dict()
        if suite_name not in suites:
            raise KeyError("Unknown LIBERO suite: " + suite_name)
        self.suite = suites[suite_name](task_order_index=int(self.config.get("task_order_index", 0)))
        task_name = self.config.get("task_name", DEFAULT_TASK)
        if task_name is None and "task_id" in self.config:
            self.task_id = int(self.config["task_id"])
        else:
            matches = [i for i in range(self.suite.n_tasks)
                       if self.suite.get_task(i).name == task_name]
            if len(matches) != 1:
                raise ValueError("Task name must match exactly one task in " + suite_name + ": " + str(task_name))
            self.task_id = matches[0]
        self.task = self.suite.get_task(self.task_id)
        bddl_path = Path(self.suite.get_task_bddl_file_path(self.task_id)).resolve()
        size = self.config.get("camera_size", 256)
        height, width = (int(size), int(size)) if np.isscalar(size) else tuple(map(int, size))
        if height < 8 or width < 8:
            raise ValueError("Camera dimensions must be at least 8 pixels")
        cameras = list(self.config.get("camera_names", ["agentview", "robot0_eye_in_hand"]))
        if not cameras:
            raise ValueError("At least one camera is required")
        self.env = SegmentationRenderEnv(
            bddl_file_name=str(bddl_path), robots=["Panda"], controller="OSC_POSE",
            use_camera_obs=True, camera_names=cameras,
            camera_heights=height, camera_widths=width,
            camera_depths=True, camera_segmentations="instance",
            control_freq=int(self.config.get("control_freq", 20)),
            horizon=self.max_steps + int(self.config.get("settle_steps", 5)) + 1,
            render_gpu_device_id=int(self.config.get("render_gpu_device_id", -1)),
            ignore_done=False,
        )
        self.task_env = self.env.env
        if len(self.task_env.robots) != 1:
            self.env.close()
            self.env = None
            raise ValueError("This bridge currently supports one Panda robot")
        self.robot = self.task_env.robots[0]
        self.env.seed(int(self.config.get("seed", 0)))
        self.merged_xml_path.write_text(self.task_env.model.get_xml())
        self.provenance = dict(
            libero_repo=str(self.repo), libero_core=str(core.__file__),
            env_wrapper=str(wrapper_module.__file__), config_path=str(self.config_dir / "config.yaml"),
            robosuite_version=str(robosuite.__version__), mujoco_version=str(mujoco.__version__),
            task_suite=suite_name, task_id=int(self.task_id), task_name=self.task.name,
            bddl_file=str(bddl_path), language=self.task.language,
            controller="OSC_POSE", control_freq=int(self.task_env.control_freq),
            camera_names=cameras, camera_shape=[height, width],
            depth=True, segmentation="simulator_ground_truth_instance",
            merged_model_xml=str(self.merged_xml_path), resource_paths=paths,
        )
        (self.run_dir / "ENVIRONMENT.json").write_text(json.dumps(self.provenance, indent=2) + "\n")
        return self.env

    def reset(self):
        self.make_env()
        self.env.seed(int(self.config.get("seed", 0)))
        self.obs = self.env.reset()
        # hard_reset recreates robots/controllers and invalidates old references.
        self.task_env = self.env.env
        if len(self.task_env.robots) != 1:
            raise ValueError("Expected one active robot after reset")
        self.robot = self.task_env.robots[0]
        import torch
        from libero.libero import get_libero_path

        path = Path(get_libero_path("init_states")) / self.task.problem_folder / self.task.init_states_file
        # Official trusted initial-state assets contain NumPy objects. PyTorch
        # 2.6's weights-only default cannot deserialize them.
        kwargs = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
        states = torch.load(str(path), **kwargs)
        state_id = int(self.config.get("init_state_id", 0))
        if not 0 <= state_id < len(states):
            raise IndexError(f"init_state_id {state_id} outside 0..{len(states)-1}")
        self.obs = self.env.set_init_state(np.asarray(states[state_id]))
        self.settle_steps_performed = int(self.config.get("settle_steps", 5))
        if self.settle_steps_performed < 0:
            raise ValueError("settle_steps cannot be negative")
        for _ in range(self.settle_steps_performed):
            self.obs, _, _, _ = self.env.step(np.zeros(7, dtype=np.float64))
        self.step_count = 0
        self.success_seen = bool(self.env.check_success())
        missing = []
        for name in (self.target_object, self.destination):
            try:
                self.body_id(name)
            except KeyError:
                missing.append(name)
        if missing:
            raise KeyError("Configured target/destination absent from task: " + ", ".join(missing))
        from graspgen_libero_bridge.adapters.libero.model_snapshot import save_reset_model
        save_reset_model(self.env.sim, self.run_dir)
        self.provenance.update(init_state_file=str(path), init_state_id=state_id,
                               initial_states_count=len(states), settle_steps=self.settle_steps_performed,
                               reset_model_binary=str(self.run_dir/'reset_model.mjb'),
                               reset_model_manifest=str(self.run_dir/'MODEL_SNAPSHOT.json'))
        (self.run_dir / "ENVIRONMENT.json").write_text(json.dumps(self.provenance, indent=2) + "\n")
        return self.obs

    def _require_reset(self):
        if self.env is None or self.obs is None:
            raise RuntimeError("Call reset() before accessing the LIBERO state")

    def body_id(self, name):
        """Resolve an object or the support body of its configured goal region.

        Arenas such as tables have valid MuJoCo bodies but are not entries in
        LIBERO's movable-object registry. Resolve only the configured region's
        actual support; never substitute another object or omit its geometry.
        """
        self._require_reset()
        if name in self.task_env.obj_body_id:
            return int(self.task_env.obj_body_id[name])
        region = self.config.get("goal_destination")
        if name == self.destination and region in self.task_env.object_sites_dict:
            model = self.env.sim.model
            site = int(model.site_name2id(region))
            body = int(model.site_bodyid[site])
            definition = self.task_env.parsed_problem.get("regions", {}).get(region, {})
            if definition.get("target") == name and body > 0:
                return body
        raise KeyError("Unknown task object or goal-region support: " + str(name))

    def target_pose(self, name):
        self._require_reset()
        body_id = self.body_id(name)
        data = self.env.sim.data
        return _pose(data.body_xpos[int(body_id)], data.body_xmat[int(body_id)])

    def state(self):
        self._require_reset()
        robot = self.task_env.robots[0]
        self.robot = robot
        sim = self.env.sim
        model, data = sim.model, sim.data
        base_id = int(model.body_name2id(robot.robot_model.root_body))
        site_id = int(robot.eef_site_id)
        q_arm = np.asarray(data.qpos[robot._ref_joint_pos_indexes], dtype=np.float64)
        q_fingers = np.asarray(data.qpos[robot._ref_gripper_joint_pos_indexes], dtype=np.float64)
        if q_arm.shape != (7,) or q_fingers.shape != (2,):
            raise ValueError("Expected seven arm joints and two Panda finger joints")
        objects = {name: self.target_pose(name).tolist() for name in self.task_env.obj_body_id}
        native_data = getattr(data, "_data", data)
        return dict(
            q_arm=q_arm.tolist(), q_fingers=q_fingers.tolist(),
            qvel_arm=np.asarray(data.qvel[robot._ref_joint_vel_indexes]).tolist(),
            qvel_fingers=np.asarray(data.qvel[robot._ref_gripper_joint_vel_indexes]).tolist(),
            gripper_width=float(abs(q_fingers[0]) + abs(q_fingers[1])),
            world_T_eef=_pose(data.site_xpos[site_id], data.site_xmat[site_id]).tolist(),
            world_T_base=_pose(data.body_xpos[base_id], data.body_xmat[base_id]).tolist(),
            joint_names=list(robot.robot_model.joints), finger_joint_names=list(robot.gripper.joints),
            object_poses=objects, objects=objects,
            arm_actuator_force=np.asarray(native_data.qfrc_actuator[robot._ref_joint_vel_indexes]).tolist(),
            contact_count=int(data.ncon), sim_time=float(data.time), step_count=self.step_count,
            eef_site=str(robot.gripper.important_sites["grip_site"]),
            base_body=str(robot.robot_model.root_body),
        )

    def calibration_samples(self):
        """Evaluate static FK at >=5 joint states, without stepping physics.

        The integrator state and auxiliary mutable input arrays are restored in
        a finally block. Controllers, observations, and episode counters are not
        advanced. This method is intended before planning an episode.
        """
        self._require_reset()
        sim, robot = self.env.sim, self.robot
        data = getattr(sim.data, "_data", sim.data)
        state = copy.deepcopy(sim.get_state())
        arrays = {}
        for name in ("act", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
                     "mocap_pos", "mocap_quat", "userdata", "eq_active", "plugin_state"):
            value = getattr(data, name, None)
            if value is not None:
                arrays[name] = np.array(value, copy=True)
        arm_indices = robot._ref_joint_pos_indexes
        initial = np.array(sim.data.qpos[arm_indices], copy=True)
        limits = np.asarray([sim.model.jnt_range[sim.model.joint_name2id(name)]
                             for name in robot.robot_model.joints], dtype=float)
        count = max(5, int(self.config.get("calibration_sample_count", 7)))
        configurations = [initial]
        for joint in range(7):
            for sign in (1., -1.):
                q = initial.copy()
                q[joint] = np.clip(q[joint] + sign * .06,
                                   limits[joint, 0] + 1e-4, limits[joint, 1] - 1e-4)
                if min(np.max(np.abs(q - old)) for old in configurations) > 1e-5:
                    configurations.append(q)
        samples = []
        try:
            for q in configurations[:count]:
                sim.data.qpos[arm_indices] = q
                sim.forward()
                current = self.state()
                samples.append({name: current[name] for name in
                                ("q_arm", "q_fingers", "world_T_eef", "world_T_base")})
        finally:
            sim.set_state(state)
            for name, value in arrays.items():
                getattr(data, name)[...] = value
            sim.forward()
            # mj_forward may update warm-start and derived force arrays. Keep
            # the original integration warm-start and external inputs as well.
            for name, value in arrays.items():
                getattr(data, name)[...] = value
        if len(samples) < 5:
            raise RuntimeError("Could not generate five distinct Panda FK samples")
        if not np.allclose(np.asarray(self.state()["q_arm"]), initial, atol=1e-12, rtol=0):
            raise RuntimeError("Calibration failed to restore the robot state")
        (self.run_dir / "MUJOCO_CALIBRATION_SAMPLES.json").write_text(
            json.dumps(samples, indent=2) + "\n")
        return samples

    def step(self, action):
        self._require_reset()
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Default LIBERO OSC_POSE requires seven finite action values")
        if (np.abs(action) > 1.000001).any():
            raise ValueError("OSC action exceeds normalized [-1,1] bounds")
        if self.step_count >= self.max_steps:
            raise ExecutionError("Episode action budget exhausted", code="episode_step_limit",
                                 category="budget", details={"max_steps": self.max_steps})
        self.obs, reward, done, info = self.env.step(np.clip(action, -1, 1))
        self.step_count += 1
        info = dict(info or {})
        success = bool(self.env.check_success())
        self.success_seen = self.success_seen or success
        limit = self.step_count >= self.max_steps
        info.update(task_success=success, time_limit_reached=bool(limit and not success),
                    step_count=self.step_count)
        return self.obs, reward, bool(done or success or limit), info

    def check_success(self):
        self._require_reset()
        return bool(self.env.check_success())

    def close(self):
        if self.env is not None:
            self.env.close()
        self.env = self.task_env = self.robot = self.obs = None
