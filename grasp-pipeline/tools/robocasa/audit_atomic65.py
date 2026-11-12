#!/usr/bin/env python3
"""Verify completed recordings without modifying any task result."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import imageio_ffmpeg
from report_atomic65 import summarize, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing audit")
    summary = summarize(args.root)
    if args.require_complete and not summary["complete"]:
        parser.error("The 65-task sweep has not finished")
    policy = read(args.root / "POLICY.json")
    checks = []
    for row in summary["rows"]:
        if not row["completed"]:
            continue
        directory = args.root / "cases" / row["case"]
        video = directory / "VIDEO.mp4"
        audit_file = directory / "VIDEO_AUDIT.json"
        current_sha = hashlib.sha256(video.read_bytes()).hexdigest() if video.is_file() else None
        cached = read(audit_file)
        if cached and cached.get("sha256") == current_sha:
            media = cached
        else:
            result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video), "-f", "null", "-"],
                                    capture_output=True, text=True, timeout=90) if video.is_file() else None
            media = {"decoded": bool(result is not None and result.returncode == 0), "sha256": current_sha,
                     "bytes": video.stat().st_size if video.is_file() else 0,
                     "decode_error": None if result is None or result.returncode == 0 else result.stderr[-1200:]}
            audit_file.write_text(json.dumps(media, indent=2) + "\n")
        record = read(directory / "RECORDING.json") or {}
        length_errors = []
        try:
            with h5py.File(directory / "TRAJECTORY.h5", "r") as trajectory:
                expected = row["physical_steps"]
                actions = len(trajectory["actions"]) if "actions" in trajectory else 0
                if actions != expected:
                    length_errors.append("action count mismatch")
                for camera in policy["camera_names"]:
                    for suffix in ("_image", "_depth", "_segmentation_instance"):
                        key = "observations/" + camera + suffix
                        if key not in trajectory or len(trajectory[key]) != expected + 1:
                            length_errors.append(key + " length mismatch")
        except (OSError, ValueError) as exc:
            length_errors.append(str(exc))
        generators = [read(p) for p in directory.glob("stages/*/grasp_*/GENERATOR.json")]
        calls = sum(int(p.get("generation_calls", 0)) for p in generators if p)
        cached_candidates = any(p and p.get("backend") == "recorded_identical_graspgen_pool" for p in generators)
        checks.append({"task_id": row["task_id"], "status": row["status"], "identity_valid": row["identity_valid"],
                       "video": media, "physical_steps": row["physical_steps"],
                       "alignment_valid": bool(record.get("alignment_valid") and not length_errors),
                       "recording_errors": length_errors, "fresh_model_inference_calls": calls,
                       "used_cached_candidates": cached_candidates})
    audit = {"complete": summary["complete"], "checked_cases": len(checks), "source_digest": summary["source_digest"],
             "task_bindings_digest": summary["task_bindings_digest"], "all_identities_valid": all(c["identity_valid"] for c in checks),
             "decoded_videos": sum(c["video"]["decoded"] for c in checks),
             "aligned_recordings": sum(c["alignment_valid"] for c in checks),
             "used_cached_candidates": any(c["used_cached_candidates"] for c in checks),
             "fresh_model_inference_calls": sum(c["fresh_model_inference_calls"] for c in checks), "cases": checks}
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ["complete", "checked_cases", "all_identities_valid", "decoded_videos", "aligned_recordings", "fresh_model_inference_calls", "used_cached_candidates"]}))
    return 0 if audit["all_identities_valid"] and audit["decoded_videos"] == len(checks) and audit["aligned_recordings"] == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
