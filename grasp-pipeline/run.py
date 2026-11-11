#!/usr/bin/env python3
"""Checkout entry point: no installation or model imports required for --help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from graspgen_libero_bridge.runtime.cli import main

if __name__ == "__main__":
    sys.exit(main())
