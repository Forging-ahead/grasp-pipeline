"""CPU-only comparison of copied original and organized sources in a sandbox.

Place next to original-v10/ and organized/. No GPU actors are launched.
"""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
common = dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none", PYTHONDONTWRITEBYTECODE="1",
              PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
assets = importlib.util.find_spec("robosuite")
common["ROBOSUITE_ASSETS"] = str(Path(assets.origin).parent / "models/assets")
commands = []


def run(label, arguments, cwd, pythonpath):
    log = root / (label + ".log")
    command = [sys.executable, *arguments]
    environment = dict(common, PYTHONPATH=str(pythonpath))
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    commands.append(dict(label=label, command=command, cwd=str(cwd), log=str(log), returncode=result.returncode))
    print(json.dumps(dict(label=label, returncode=result.returncode, tail=log.read_text()[-600:])), flush=True)


run("baseline", ["-m", "pytest", "-q", "tests", "-p", "no:cacheprovider", "--junitxml=" + str(root / "baseline.xml")],
    root / "original-v10/bridge", root / "original-v10/bridge")
run("organized", ["-m", "pytest", "-q", "tests", "-p", "no:cacheprovider", "--junitxml=" + str(root / "organized.xml")],
    root / "organized", root / "organized/src")
run("preservation", [str(root / "organized/tools/verify_preservation.py"), "--baseline", str(root / "original-v10"),
                     "--report", str(root / "preservation.json")], root / "organized", root / "organized/src")
for label, module in [("model_worker_help", "graspgen_libero_bridge.models.graspgen_worker"),
                      ("planner_worker_help", "graspgen_libero_bridge.planning.planner_worker"),
                      ("package_cli_help", "graspgen_libero_bridge")]:
    run(label, ["-m", module, "--help"], root / "organized", root / "organized/src")


def cases(path):
    rows = {}
    for node in ET.parse(path).iter("testcase"):
        key = node.attrib.get("classname", "").split(".")[-1] + "::" + node.attrib["name"]
        status = "error" if node.find("error") is not None else "failed" if node.find("failure") is not None else "skipped" if node.find("skipped") is not None else "passed"
        if key in rows:
            raise RuntimeError("Ambiguous test identity: " + key)
        rows[key] = status
    return rows


original, organized = cases(root / "baseline.xml"), cases(root / "organized.xml")
differences = {key: {"baseline": status, "organized": organized.get(key)} for key, status in original.items()
               if organized.get(key) != status}
report = dict(passed=all(c["returncode"] == 0 for c in commands) and not differences
              and all(s == "passed" for s in organized.values()), checked_at_utc=datetime.now(timezone.utc).isoformat(),
              python=sys.version, executable=sys.executable, remote_root=str(root),
              gpu_visible=common["CUDA_VISIBLE_DEVICES"], full_task_rollouts_started=0,
              original_tests=len(original), organized_tests=len(organized),
              original_statuses={s: list(original.values()).count(s) for s in set(original.values())},
              organized_statuses={s: list(organized.values()).count(s) for s in set(organized.values())},
              baseline_case_differences=differences, commands=commands,
              scope="Original CPU tests, organized CPU tests, source/config preservation and worker help entry points; no GPU inference or rollout")
(root / "VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("commands", "baseline_case_differences")}, indent=2), flush=True)
sys.exit(0 if report["passed"] else 1)
