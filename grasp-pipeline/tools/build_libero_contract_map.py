#!/usr/bin/env python3
"""Record current source anchors and gaps without executing the old driver."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = "src/graspgen_libero_bridge/"
ROWS = [
    ("EnvironmentPort", "reset/observe/step/close", P+"adapters/libero/env_adapter.py", "EnvAdapter.step", "Existing step returns a fused 4-tuple done; v1 requires separate native terminated/truncated and stable episode/state IDs"),
    ("TaskPort", "describe_task/check_native_success/check_subgoal", "apps/libero/run_task.py", "switch_target", "Original goals and private predicate calls must be wrapped; full language order is not inferred by the current scheduler"),
    ("TaskPort", "goal scheduling reference", P+"policy/task_policy.py", "schedule", "BDDL predicates are current input; other benchmarks can supply normalized goals, preserving native evaluation"),
    ("RobotPort", "describe_robot/robot_state", P+"adapters/libero/env_adapter.py", "EnvAdapter.state", "Name-ordered state exists; separate privileged object_poses from proprioception and publish robot/TCP model explicitly"),
    ("ScenePort", "scene_snapshot", P+"adapters/libero/scene.py", "export_scene", "Currently uses MuJoCo truth; add scene revision, asset resolution and explicit region/attachment records"),
    ("PlanningPort", "plan", P+"planning/planner_worker.py", "BridgePlanner.plan_pose", "Bind request/scene/state IDs and normalize backend result; cuRobo does not choose goals or skills"),
    ("PreviewPort", "validate_and_restore", P+"control/controller.py", "ContactMonitor.validate_path", "Native state snapshot/contact checking is engine-specific; preserve no-step and full restoration guarantees"),
    ("PreviewPort", "local IK backend", P+"planning/cartesian.py", "plan_cartesian", "Jacobian and saved-state access remain MuJoCo-specific; new engines need an equivalent backend"),
    ("ExecutionPort", "execute", P+"control/controller.py", "Controller.follow_path", "Resolve actual action encoding, monitor feedback, gripper state and episode end; reached pose is not grasp success"),
    ("RecordingPort", "record_step/finalize", P+"control/controller.py", "Recorder.capture", "Current state/action/video recorder does not save complete per-step raw RGB-D or VLM decisions"),
    ("CapabilityPort", "capabilities/check_requirements reference", P+"contracts/validation.py", "readiness", "Offline declared-data checker only; no registered live adapter or runtime gate yet"),
]


def main():
    rows = []
    for port, method, file, symbol, gap in ROWS:
        path = ROOT / file
        tree = ast.parse(path.read_text())
        pieces = symbol.split(".")
        nodes = tree.body
        for piece in pieces:
            node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == piece)
            nodes = node.body
        rows.append(dict(port=port, methods=method, source=file, symbol=symbol, line=node.lineno,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(), remaining_work=gap))
    document = dict(contract_version="1.0", binding_status="reference_mapping_only", ports=rows,
                    not_implemented=["A live universal adapter/facade", "Rewiring v10 runtime through ports", "RoboTwin/RoboCasa adapters", "Generic robot/gripper replacement"],
                    remaining_engine_coupling=["planning/cartesian.py", "planning/grasp_contact.py", "control/controller.py", "policy/evaluate.py", "policy/skills.py", "adapters/libero/"])
    (ROOT / "contracts/LIBERO_V10_BINDINGS.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"source_anchors": len(rows), "ports": len({r['port'] for r in rows})}))


if __name__ == "__main__":
    main()
