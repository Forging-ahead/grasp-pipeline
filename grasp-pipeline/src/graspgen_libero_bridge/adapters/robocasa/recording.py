"""Streaming, frame-aligned real observations and actions; no preview frames."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .native import json_value, write_json


class TrajectoryRecorder:
    def __init__(self, environment):
        import h5py
        import imageio.v2 as imageio
        from robosuite import macros

        self.environment = environment
        self.directory = environment.output
        self.phase = "initial"
        self.record_wall_s = 0.
        self.frame_count = 0
        self.datasets = {}
        self.flip = str(macros.IMAGE_CONVENTION).lower() == "opengl"
        self.h5 = h5py.File(self.directory / "TRAJECTORY.h5", "x")
        self.h5.attrs["schema_version"] = "robocasa-trajectory-1"
        self.h5.attrs["observation_action_alignment"] = "observation[t], action[t], observation[t+1]"
        self.h5.attrs["teacher_provenance"] = environment.config["input_mode"]
        self.events = (self.directory / "STEPS.jsonl").open("x")
        self.decisions = (self.directory / "DECISIONS.jsonl").open("x")
        self.video = imageio.get_writer(str(self.directory / "VIDEO.mp4"),
                                       fps=float(environment.env.control_freq), codec="libx264",
                                       macro_block_size=1, pixelformat="yuv420p")
        self.closed = False
        self._observation()
        environment.observers.append(self.record_step)

    def _append(self, name, value, *, compress=False):
        value = np.asarray(value)
        if name not in self.datasets:
            kwargs = {"compression": "gzip", "compression_opts": 1} if compress else {}
            self.datasets[name] = self.h5.create_dataset(name, shape=(0, *value.shape),
                    maxshape=(None, *value.shape), chunks=(1, *value.shape), dtype=value.dtype, **kwargs)
        ds = self.datasets[name]
        ds.resize(len(ds) + 1, axis=0)
        ds[-1] = value

    def _observation(self):
        env = self.environment
        self._append("states/qpos", env.env.sim.data.qpos)
        self._append("states/qvel", env.env.sim.data.qvel)
        self._append("states/sim_time", np.asarray(env.env.sim.data.time))
        self._append("states/world_T_tcp", env.robot_state()["world_T_eef"])
        views = []
        for camera in env.config["policy"]["camera_names"]:
            for suffix in ("_image", "_depth", "_segmentation_instance"):
                array = np.asarray(env.obs[camera + suffix])
                array = np.flip(array, axis=0) if self.flip else array
                self._append("observations/" + camera + suffix, array, compress=True)
                if suffix == "_image":
                    views.append(array.astype(np.uint8))
        self.video.append_data(np.concatenate(views, axis=1))
        self.frame_count += 1

    def record_step(self, before, action, result):
        started = time.perf_counter()
        self._append("actions", action)
        self._observation()
        env = self.environment
        self.events.write(json.dumps(json_value({"step": env.steps, "phase": self.phase,
                         "before": before, "after": result["observation"]["robot_state"],
                         "action": action, "native_success": result["native_success"],
                         "native_done": result["terminated"], "runner_truncated": result["truncated"],
                         "sim_step_wall_s": result["step_wall_s"]}), allow_nan=False) + "\n")
        self.record_wall_s += time.perf_counter() - started

    def decision(self, tool_name, arguments, *, expected_subgoal=None):
        record = {"observation_index": self.environment.steps,
                  "episode_id": self.environment.episode_id,
                  "revision": self.environment.revision,
                  "sim_time": float(self.environment.env.sim.data.time),
                  "phase": self.phase, "teacher": "code_policy_teacher",
                  "tool_name": tool_name, "arguments": arguments,
                  "expected_subgoal": expected_subgoal,
                  "observation_ref": "TRAJECTORY.h5:observations[" + str(self.environment.steps) + "]"}
        self.decisions.write(json.dumps(json_value(record), allow_nan=False) + "\n")
        self.decisions.flush()
        self.h5.flush()
        return record

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.environment.observers.remove(self.record_step)
        self.video.close()
        self.h5.attrs["physical_actions"] = self.environment.steps
        self.h5.attrs["observation_frames"] = self.frame_count
        self.h5.close()
        self.events.close()
        self.decisions.close()
        write_json(self.directory / "RECORDING.json", {
            "physical_actions": self.environment.steps, "observation_frames": self.frame_count,
            "alignment_valid": self.frame_count == self.environment.steps + 1,
            "fps": float(self.environment.env.control_freq), "record_wall_s": self.record_wall_s,
            "no_action_episode": self.environment.steps == 0,
            "video_available": (self.directory / "VIDEO.mp4").is_file()})
