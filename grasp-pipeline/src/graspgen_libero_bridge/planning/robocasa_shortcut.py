"""Bounded collision-checked shortcuts for long graph paths, preserving endpoints."""
from __future__ import annotations

import numpy as np


def joint_length(path):
    return float(np.max(np.abs(np.diff(path, axis=0)), axis=1).sum()) if len(path) > 1 else 0.


def shortcut_path(path, validate_segment, *, max_checks, remaining_seconds):
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 7 or not len(path) or not np.isfinite(path).all():
        raise ValueError("Expected finite Nx7 joint path")
    output, index, checks, accepted = [path[0].copy()], 0, [], 0
    while index < len(path) - 1 and len(checks) < max_checks and remaining_seconds() > .2:
        target = len(path) - 1
        found = False
        while target > index + 1 and len(checks) < max_checks and remaining_seconds() > .2:
            count = max(1, int(np.ceil(np.abs(path[target] - path[index]).max() / .025)))
            segment = np.linspace(path[index], path[target], count + 1)
            if joint_length(segment) >= joint_length(path[index:target + 1]) - 1e-6:
                target = index + (target - index) // 2
                continue
            valid = bool(validate_segment(segment))
            checks.append({"first_index": index, "last_index": target, "valid": valid})
            if valid:
                output.extend(segment[1:]); index = target
                accepted += 1; found = True
                break
            target = index + (target - index) // 2
        if not found:
            break
    output.extend(path[index + 1:])
    result = np.asarray(output)
    return result, {"checks": checks, "accepted_shortcuts": accepted,
                    "original_samples": len(path), "result_samples": len(result),
                    "original_joint_length": joint_length(path), "result_joint_length": joint_length(result)}
