"""Verify a rerun's native scene and full saved initial physics before action."""
import json
from pathlib import Path

import numpy as np

from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure


def verify_initial_state(config, current_dir, baseline_dir):
    current_dir, baseline_dir = Path(current_dir), Path(baseline_dir)
    old_result = json.loads((baseline_dir / "RESULT.json").read_text())
    old_lock = json.loads((baseline_dir / "ENVIRONMENT_LOCK.json").read_text())
    new_lock = json.loads((current_dir / "ENVIRONMENT_LOCK.json").read_text())
    report = {"baseline_case": str(baseline_dir), "matched": False, "arrays": {}, "errors": []}
    for key, expected in (("task_id", config["task"]["task_id"]), ("seed", config["seed"]), ("split", config["split"])):
        if old_result.get(key) != expected:
            report["errors"].append("episode_identity:" + key)
    for key in ("scene_xml_sha256", "success_source_sha256", "action_spec"):
        if key not in old_lock or key not in new_lock or old_lock[key] != new_lock[key]:
            report["errors"].append("native_environment:" + key)
    with np.load(baseline_dir / "INITIAL_STATE.npz", allow_pickle=False) as old, np.load(current_dir / "INITIAL_STATE.npz", allow_pickle=False) as new:
        if set(old.files) != set(new.files):
            report["errors"].append("initial_physics_field_set")
        for key in sorted(set(old.files) & set(new.files)):
            a, b = old[key], new[key]
            same_shape = a.shape == b.shape
            matched = bool(same_shape and np.allclose(a, b, atol=1e-12, rtol=0., equal_nan=False))
            error = float(np.max(np.abs(a - b))) if same_shape and a.size else (0. if same_shape else None)
            if error is not None and not np.isfinite(error):
                error = None
            report["arrays"][key] = {"matched": matched, "max_abs_error": error}
            if not matched:
                report["errors"].append("initial_physics:" + key)
    report["matched"] = not report["errors"]
    (current_dir / "INITIAL_COMPARISON.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["matched"]:
        raise EpisodeFailure("INITIAL_STATE_MISMATCH", "Rerun differs from its frozen baseline initial state", details=report)
    return report
