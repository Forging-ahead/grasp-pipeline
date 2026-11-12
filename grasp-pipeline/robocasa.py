#!/usr/bin/env python3
"""Explicit RoboCasa entry; the existing run.py remains the LIBERO launcher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from graspgen_libero_bridge.runtime.robocasa_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
