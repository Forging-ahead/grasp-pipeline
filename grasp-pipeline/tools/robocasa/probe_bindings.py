#!/usr/bin/env python3
"""Check native task entities/handles/targets without executing any task skill."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["pretrain", "target"], default="pretrain")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    from graspgen_libero_bridge.adapters.robocasa.config import load_config
    from graspgen_libero_bridge.adapters.robocasa.native import RoboCasaEnvironment, write_json
    from graspgen_libero_bridge.adapters.robocasa.articulation import native_joints
    from graspgen_libero_bridge.runtime.robocasa_provenance import source_identity
    config = load_config(args.task_id, paths_file=args.paths, split=args.split)
    env = RoboCasaEnvironment(config, args.output)
    report = {"scope": "native binding/lifecycle inventory, no task rollout", "task_id": args.task_id,
              "passed": False, "physical_actions": 0, "task_success_claimed": False,
              "source_digest": source_identity()["source_digest"], "bindings": []}
    try:
        write_json(args.output / "CONFIG.json", config)
        env.reset()
        for stage in config["task"]["stages"]:
            if stage["skill"] == "pick_place":
                report["bindings"].append({"skill": stage["skill"], "target": stage["target"],
                    "target_body_id": env.entity_body(stage["target"]), "destination": stage["destination"],
                    "destination_body_id": env.entity_body(stage["destination"]),
                    "contents_body_ids": [env.entity_body(n) for n in stage.get("carried_contents", [])]})
            elif stage["skill"] == "articulation":
                report["bindings"].append({"skill": stage["skill"], "fixture": stage["fixture"],
                                           "native_joints": native_joints(env, stage)})
            elif stage["skill"] == "navigate":
                report["bindings"].append({"skill": stage["skill"],
                    "native_position": getattr(env.env, stage["native_target"]),
                    "native_orientation": getattr(env.env, stage["native_orientation"])})
            else:
                raise ValueError("Unknown stage skill")
        report["passed"] = True
    except Exception as exc:
        report.update(error=str(exc), failure_code=getattr(exc, "code", "BINDING_ERROR"), traceback=traceback.format_exc())
    finally:
        report["physical_actions"] = env.steps
        env.close()
        write_json(args.output / "BINDING_PROBE.json", report)
        print(json.dumps({"task_id": args.task_id, "passed": report["passed"], "physical_actions": env.steps,
                          "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    main()
