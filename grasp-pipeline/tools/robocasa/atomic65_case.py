#!/usr/bin/env python3
"""Run one frozen task configuration with the unchanged actor and a hard wall budget."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.runtime.robocasa_supervision import run_supervised
from graspgen_libero_bridge.common.layout import source_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["run", "probe"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    env = os.environ.copy()
    gpu = str(config["gpu"])
    env.update(PYTHONPATH=str(source_root()), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", CUDA_VISIBLE_DEVICES=gpu,
               MUJOCO_EGL_DEVICE_ID=gpu, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    temp = args.output / "tmp"; temp.mkdir()
    env["TMPDIR"] = str(temp)
    env["NUMBA_CACHE_DIR"] = str(args.output / "numba-cache")
    command = [config["paths"]["simulation_python"], str(Path(__file__).parent / "atomic65_driver.py"), args.mode,
               "--config", str(args.config), "--output", str(args.output)]
    with (args.output / "RUN.log").open("x") as log:
        supervisor = run_supervised(command, env=env, log=log, seconds=config["policy"]["episode_wall_seconds_atomic"])
    (args.output / "SUPERVISOR.json").write_text(json.dumps(supervisor, indent=2) + "\n")
    return supervisor["return_code"]


if __name__ == "__main__":
    raise SystemExit(main())
