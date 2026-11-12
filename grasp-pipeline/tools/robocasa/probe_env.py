#!/usr/bin/env python3
"""Launch the native RoboCasa lifecycle/observation probe in its own interpreter."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graspgen_libero_bridge.runtime.robocasa_cli import main

if __name__ == "__main__":
    raise SystemExit(main(["probe", *sys.argv[1:]]))
