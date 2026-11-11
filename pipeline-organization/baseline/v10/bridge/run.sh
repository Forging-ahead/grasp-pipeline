#!/usr/bin/env bash
set -euo pipefail
BRIDGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${BRIDGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1
exec "${LIBERO_PYTHON:-/opt/venv/openpi/bin/python}" -m graspgen_libero_bridge "$@"
