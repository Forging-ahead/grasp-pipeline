"""Lightweight checkout CLI. Listing/configuration never load robotics code."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from graspgen_libero_bridge.common.layout import project_root, source_root
from graspgen_libero_bridge.runtime.config import load_paths, load_task, task_catalog


def _task_options(parser):
    parser.add_argument("key")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--paths", type=Path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Organized GraspGen/cuRobo LIBERO pipeline, frozen v10 policy")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List the frozen 130 task keys")
    listing.add_argument("--json", action="store_true")
    config = commands.add_parser("config", help="Print the effective launch config; no model or environment")
    _task_options(config)
    run = commands.add_parser("run", help="Run a task using the configured simulation interpreter")
    _task_options(run)
    run.add_argument("--tag", default="dev1")
    run.add_argument("--output-root", type=Path)
    run.add_argument("--inspect", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="Print command and config without starting a child")
    legacy = commands.add_parser("legacy", help="Original explicit single-task doctor/run interface")
    legacy.add_argument("--paths", type=Path)
    legacy.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "list":
        rows = task_catalog()
        print(json.dumps([r["key"] for r in rows], indent=2) if args.json else "\n".join(r["key"] for r in rows))
        return 0
    if args.command == "config":
        _, cfg = load_task(args.key, gpu=args.gpu, init=args.init, paths_file=args.paths)
        print(json.dumps(cfg, indent=2))
        return 0
    paths = load_paths(args.paths)
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_root()) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    if args.command == "legacy":
        if not args.arguments:
            parser.error("legacy requires doctor or run and its original arguments")
        command = [paths["simulation_python"], "-m", "graspgen_libero_bridge.runtime.legacy_cli", *args.arguments]
        if args.paths is not None:
            command.extend(["--paths", str(args.paths.resolve())])
    else:
        _, cfg = load_task(args.key, gpu=args.gpu, init=args.init, paths_file=args.paths)
        command = [paths["simulation_python"], str(project_root() / "apps/libero/run_task.py"), args.key,
                   "--gpu", str(args.gpu), "--init", str(args.init), "--tag", args.tag]
        if args.paths is not None:
            command.extend(["--paths", str(args.paths.resolve())])
        if args.output_root is not None:
            command.extend(["--output-root", str(args.output_root.resolve())])
        if args.inspect:
            command.append("--inspect")
        if args.dry_run:
            print(json.dumps({"command": command, "config": cfg,
                              "output": str((args.output_root or project_root() / "runs") / args.tag / args.key),
                              "simulation_started": False}, indent=2))
            return 0
    return subprocess.call(command, env=environment)


if __name__ == "__main__":
    sys.exit(main())
