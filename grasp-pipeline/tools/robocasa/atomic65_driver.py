#!/usr/bin/env python3
"""Give the frozen driver a cleanup window so deadline failures keep recordings."""
import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


class RecordingDeadline(BaseException):
    """Bypass nested planning exception handlers, then run the driver's finally."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "probe"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    budget = config["policy"]["episode_wall_seconds_atomic"]
    def deadline(_signum, _frame):
        raise RecordingDeadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(1., budget - 10.))
    try:
        from graspgen_libero_bridge.runtime.robocasa_driver import main as driver_main
        return driver_main([args.mode, "--config", str(args.config), "--output", str(args.output)])
    except RecordingDeadline:
        path = args.output / "RESULT.json"
        result = json.loads(path.read_text()) if path.is_file() else {}
        result.update(status="failed", pipeline_success=False, training_candidate=False, native_budget_success=False,
                      failure_code="WALL_TIMEOUT", failure_stage="recording_deadline",
                      error="890-second work deadline reached; 10 seconds of the 900-second cap reserved for recording cleanup",
                      recording_cleanup_grace_s=10, work_budget_s=budget - 10.)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return 124
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)


if __name__ == "__main__":
    raise SystemExit(main())
