"""Load exactly the frozen v10 task parameters plus deployment-only paths.

The task catalog is byte-identical to v10. The 15 literal driver overrides are
extracted without changing their values. GPU and init retain their CLI binding.
"""
import json
from pathlib import Path

from graspgen_libero_bridge.common.layout import project_root


PATH_KEYS = ("libero_repo", "grasp_repo", "graspgen_python", "graspgen_config", "planner_python")
RUNTIME_KEYS = (*PATH_KEYS, "simulation_python")


def _read(path):
    return json.loads(Path(path).read_text())


def default_paths_file():
    return project_root() / "configs/runtime/paths.server.json"


def load_paths(paths_file=None):
    source = Path(paths_file) if paths_file is not None else default_paths_file()
    paths = _read(source)
    if not isinstance(paths, dict) or set(paths) != set(RUNTIME_KEYS):
        raise ValueError("Runtime paths must contain exactly: " + ", ".join(RUNTIME_KEYS)
                         + "; policy parameters belong to the frozen policy config")
    for key, value in paths.items():
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(key + " must be a nonempty absolute path")
    return paths


def task_catalog():
    return _read(project_root() / "configs/libero/tasks.v10.json")


def load_task(key, *, gpu=0, init=0, paths_file=None):
    row = next((row for row in task_catalog() if row["key"] == key), None)
    if row is None:
        raise KeyError("Unknown task key: " + key)
    cfg = dict(row["config"], simulation_gpu=gpu, model_gpu=gpu, planner_gpu=gpu, init_state_id=init)
    cfg.update(_read(project_root() / "configs/policy/v10_overrides.json"))
    paths = load_paths(paths_file)
    cfg.update({key: paths[key] for key in PATH_KEYS})
    # Bootstrap only; each stage still resolves its true destination as in v10.
    cfg["destination"] = cfg["target_object"]
    return row, cfg
