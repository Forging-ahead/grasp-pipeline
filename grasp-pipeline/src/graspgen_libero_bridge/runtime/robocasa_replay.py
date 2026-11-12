"""Replay an identical recorded candidate pool for paired migration diagnostics."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from graspgen_libero_bridge.adapters.robocasa.errors import EpisodeFailure
from graspgen_libero_bridge.adapters.robocasa.native import write_json


def replay_candidates(environment, directory, cloud):
    source = Path(environment.config["candidate_replay_case"])
    record = json.loads((source / "RESULT.json").read_text())
    config = environment.config
    for key, expected in (("task_id", config["task"]["task_id"]), ("seed", config["seed"]), ("split", config["split"])):
        if record.get(key) != expected:
            raise EpisodeFailure("PAIRED_INPUT_MISMATCH", "Candidate replay episode differs: " + key)
    current_lock = json.loads((environment.output / "ENVIRONMENT_LOCK.json").read_text())
    source_lock = json.loads((source / "ENVIRONMENT_LOCK.json").read_text())
    if current_lock["scene_xml_sha256"] != source_lock["scene_xml_sha256"]:
        raise EpisodeFailure("PAIRED_INPUT_MISMATCH", "Candidate replay scene XML differs")
    with np.load(source / "INITIAL_STATE.npz", allow_pickle=False) as old, np.load(environment.output / "INITIAL_STATE.npz", allow_pickle=False) as new:
        if not np.array_equal(old["qpos"], new["qpos"]):
            raise EpisodeFailure("PAIRED_INPUT_MISMATCH", "Candidate replay initial qpos differs")
    pool = source / directory.relative_to(environment.output)
    with np.load(pool / "CLOUD.npz", allow_pickle=False) as old:
        if not np.array_equal(old["points_world"], cloud):
            raise EpisodeFailure("PAIRED_INPUT_MISMATCH", "Observed target cloud differs; refusing a different-input replay")
    original = pool / "CANDIDATES.npz"
    shutil.copyfile(original, directory / "CANDIDATES.npz")
    metadata = {"backend": "recorded_identical_graspgen_pool", "diagnostic_only": True,
                "source_case": str(source.resolve()), "source_pool": str(original.resolve()),
                "pool_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                "source_digest": record["source_digest"], "source_seed": record["seed"],
                "scene_xml_exact_match": True, "initial_qpos_exact_match": True, "cloud_exact_match": True,
                "generation_s": 0., "generation_calls": 0,
                "limitation": "Reuses prior model inference; not a fresh-model inference timing measurement"}
    write_json(directory / "GENERATOR.json", metadata)
    return metadata
