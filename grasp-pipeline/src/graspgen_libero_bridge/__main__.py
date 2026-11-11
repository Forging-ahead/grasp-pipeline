"""Use the organized checkout CLI; legacy doctor/run remains explicit."""
import sys
from graspgen_libero_bridge.runtime.cli import main

if __name__ == "__main__":
    sys.exit(main())
