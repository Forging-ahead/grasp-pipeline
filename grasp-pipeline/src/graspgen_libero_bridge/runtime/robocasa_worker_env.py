"""Keep generated model/planner caches on the selected experiment filesystem."""
import os
from pathlib import Path


def worker_environment(output):
    cache = output / "cache"
    defaults = {"WARP_CACHE_PATH": cache / "warp", "PYTORCH_KERNEL_CACHE_PATH": cache / "torch-kernels",
                "TORCH_EXTENSIONS_DIR": cache / "torch-extensions"}
    result = {}
    for key, path in defaults.items():
        result[key] = os.environ.get(key, str(path))
        Path(result[key]).mkdir(parents=True, exist_ok=True)
    return result
