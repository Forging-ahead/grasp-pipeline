#!/usr/bin/env python3
"""Simulation-process driver; lifecycle belongs to this one continuous episode."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graspgen_libero_bridge.runtime.robocasa_driver import main

if __name__ == "__main__":
    raise SystemExit(main())
