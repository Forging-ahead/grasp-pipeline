"""Standalone RoboCasa launcher; help, list and config require only stdlib."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from graspgen_libero_bridge.adapters.robocasa.config import catalog, load_config
from graspgen_libero_bridge.common.layout import project_root, source_root
from graspgen_libero_bridge.runtime.robocasa_supervision import run_supervised


def main(argv=None):
    parser = argparse.ArgumentParser(description="RoboCasa GraspGen/cuRobo pipeline and native validation")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List the declared RoboCasa task bindings")
    listing.add_argument("--json", action="store_true")
    for name in ("config", "probe", "validate", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("task_id")
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--split", choices=["pretrain", "target"], default="pretrain")
        sub.add_argument("--gpu", type=int, default=0)
        sub.add_argument("--paths", type=Path)
        sub.add_argument("--policy", type=Path)
        sub.add_argument("--budget-profile", choices=["native", "libero-v10-teacher"], default="native",
                         help="Native horizon by default; explicit 5000-step v10 data-collection diagnostic otherwise")
        sub.add_argument("--output", type=Path)
        sub.add_argument("--dry-run", action="store_true")
        if name == "validate":
            sub.add_argument("--execute", action="store_true", help="Actually perform the unloaded control checks")
        if name == "run":
            sub.add_argument("--replay-grasps-from", type=Path, help="Paired diagnostic: verify and reuse a recorded identical-input candidate pool")
    args = parser.parse_args(argv)
    if args.command == "list":
        rows = catalog()
        print(json.dumps(rows, ensure_ascii=False, indent=2) if args.json else
              "\n".join(row["task_id"] + " [" + row["role"] + "]" for row in rows))
        return 0
    try:
        config = load_config(args.task_id, seed=args.seed, split=args.split, gpu=args.gpu,
                             paths_file=args.paths, policy_file=args.policy, budget_profile=args.budget_profile)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if getattr(args, "replay_grasps_from", None) is not None:
        config["candidate_replay_case"] = str(args.replay_grasps_from.resolve())
    if args.command == "config":
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required; every attempt needs a new output directory")
    output = args.output.resolve()
    if output.exists():
        parser.error("Refusing to reuse an existing output directory: " + str(output))
    if not config["paths"] and not args.dry_run:
        parser.error("--paths is required for actual execution")
    command = [(config["paths"] or {}).get("simulation_python", "<simulation_python>"),
               str(project_root() / "apps/robocasa/run_task.py"), args.command,
               "--config", str(output / "CONFIG.json"), "--output", str(output)]
    if getattr(args, "execute", False):
        command.append("--execute")
    if args.dry_run:
        print(json.dumps({"command": command, "config": config, "simulation_started": False}, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    (output / "CONFIG.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source_root()) + os.pathsep + env.get("PYTHONPATH", ""),
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", CUDA_VISIBLE_DEVICES=str(args.gpu),
               MUJOCO_EGL_DEVICE_ID=str(args.gpu), PYTHONUNBUFFERED="1")
    env.setdefault("NUMBA_CACHE_DIR", str(output / "numba-cache"))
    temporary = output / "tmp"; temporary.mkdir()
    env.setdefault("TMPDIR", str(temporary))
    seconds = config["policy"]["episode_wall_seconds_composite" if config["task"]["category"] == "composite"
                               else "episode_wall_seconds_atomic"]
    with (output / "RUN.log").open("x") as log:
        supervision = run_supervised(command, env=env, log=log, seconds=seconds)
    (output / "SUPERVISOR.json").write_text(json.dumps(supervision, indent=2) + "\n")
    return supervision["return_code"]


if __name__ == "__main__":
    raise SystemExit(main())
