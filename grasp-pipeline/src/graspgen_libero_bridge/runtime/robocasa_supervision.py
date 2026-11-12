"""Bound simulator initialization and clean up its entire worker process group."""
from __future__ import annotations

import os
import signal
import subprocess
import time


def _terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # Workers can survive their parent, so clean up the group even if it exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_supervised(command, *, env, log, seconds):
    if seconds <= 0:
        raise ValueError("A positive wall-clock budget is required")
    started = time.monotonic()
    received = []
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    previous = {}

    def interrupted(signum, _frame):
        received.append(signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        while process.poll() is None and not received and time.monotonic() - started < seconds:
            try:
                process.wait(timeout=min(.2, max(.001, seconds - (time.monotonic() - started))))
            except subprocess.TimeoutExpired:
                pass
        timed_out = process.poll() is None and not received
        if timed_out or received:
            _terminate_group(process)
        return {"return_code": 128 + received[0] if received else (124 if timed_out else process.returncode),
                "child_return_code": process.returncode, "timed_out": timed_out,
                "interrupted_signal": received[0] if received else None,
                "wall_s": time.monotonic() - started, "budget_s": seconds}
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        _terminate_group(process)
