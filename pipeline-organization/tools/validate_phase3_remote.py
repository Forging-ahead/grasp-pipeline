"""CPU-only acceptance for phase3 contracts and unchanged v10 behavior."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
project = root / "organized"
environment = dict(os.environ, PYTHONPATH=str(project / "src"), CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none",
                   PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
assets = importlib.util.find_spec("robosuite")
environment["ROBOSUITE_ASSETS"] = str(Path(assets.origin).parent / "models/assets")
results = []
for label, arguments in [
    ("contracts", [str(project / "tools/check_contracts.py"), "--report", str(root / "CONTRACT_VALIDATION.json")]),
    ("tests", ["-m", "pytest", "-q", "tests", "-p", "no:cacheprovider", "--junitxml=" + str(root / "tests.xml")]),
    ("preservation", [str(project / "tools/verify_preservation.py"), "--baseline", str(root / "original-v10"), "--report", str(root / "PRESERVATION.json")]),
]:
    with (root / (label + ".log")).open("w") as log:
        process = subprocess.run([sys.executable, *arguments], cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=240)
    results.append(dict(label=label, returncode=process.returncode))
    print(json.dumps({**results[-1], "tail": (root / (label + ".log")).read_text()[-1000:]}), flush=True)
cases = list(ET.parse(root / "tests.xml").iter("testcase"))
counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
for case in cases:
    status = next((s for s in ("failure", "error", "skipped") if case.find(s) is not None), "passed")
    counts["failed" if status == "failure" else status] += 1
files = [p for base in ["src", "apps", "configs", "contracts", "tests", "tools"] for p in (project / base).rglob("*")
         if p.is_file() and p.suffix in (".py", ".json")]
files.extend([project / "run.py", project / "pyproject.toml"])
fingerprints = {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
(root / "TESTED_SOURCE_SHA256.json").write_text(json.dumps(fingerprints, indent=2) + "\n")
report = dict(passed=all(r["returncode"] == 0 for r in results) and counts["passed"] == len(cases),
              checked_at_utc=datetime.now(timezone.utc).isoformat(), remote_root=str(root), python=sys.version,
              commands=results, tests=counts, tested_source_and_contract_files=len(fingerprints),
              full_task_rollouts_started=0, gpu_visible=environment["CUDA_VISIBLE_DEVICES"],
              scope="Contract/schema/reference acceptance, all current CPU tests and unchanged v10 configuration/algorithm preservation")
(root / "VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
sys.exit(0 if report["passed"] else 1)
