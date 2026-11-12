import os
import subprocess
import sys

from graspgen_libero_bridge.runtime.robocasa_supervision import run_supervised


def test_blocked_initialization_is_bounded_and_its_worker_is_killed(tmp_path):
    script = tmp_path / "parent.py"
    pid_path = tmp_path / "worker.pid"
    script.write_text("import subprocess,sys,time\nfrom pathlib import Path\n"
                      "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                      "Path(sys.argv[1]).write_text(str(p.pid))\ntime.sleep(60)\n")
    with (tmp_path / "log").open("w") as log:
        result = run_supervised([sys.executable, str(script), str(pid_path)],
                                env=os.environ.copy(), log=log, seconds=1)
    assert result["timed_out"] and result["return_code"] == 124
    assert result["wall_s"] < 6
    # A child can remain a zombie briefly; it must not be a running GPU worker.
    pid = int(pid_path.read_text())
    completed = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    assert not completed.stdout.strip() or completed.stdout.strip().startswith("Z")


def test_normal_return_code_is_preserved(tmp_path):
    with (tmp_path / "log").open("w") as log:
        result = run_supervised([sys.executable, "-c", "raise SystemExit(7)"],
                                env=os.environ.copy(), log=log, seconds=3)
    assert result["return_code"] == 7 and not result["timed_out"]
