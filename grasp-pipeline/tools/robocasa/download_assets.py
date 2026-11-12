#!/usr/bin/env python3
"""Install pinned public RoboCasa/NVIDIA assets with SHA256 and safe ZIP checks.

Requires huggingface_hub, not robocasa. No simulator or model is imported.
Existing different files are preserved and reported as conflicts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

REPOSITORIES = {
    "robocasa/robocasa-assets": "1b92c3d02ca4354984fec961357db0bff7b32166",
    "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF": "420a04af939c34873e6839a586b70844baf28aab",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_checked(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    inventory = {}
    with zipfile.ZipFile(archive) as z:
        total = sum(info.file_size for info in z.infolist())
        if total > shutil.disk_usage(destination).free - 2 * 1024 ** 3:
            raise OSError("Insufficient space for the expanded archive plus reserve")
        for info in z.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Unsafe archive member: " + info.filename)
            target = destination / relative
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Archive escapes its asset directory")
            if info.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".asset-", delete=False) as out:
                temporary = Path(out.name)
                try:
                    with z.open(info) as source:
                        shutil.copyfileobj(source, out, length=1024 * 1024)
                    out.flush()
                    fingerprint = sha256(temporary)
                    if target.exists():
                        if not target.is_file() or sha256(target) != fingerprint:
                            raise FileExistsError("Asset conflicts with an existing file: " + str(target))
                    else:
                        os.link(temporary, target)
                    inventory[relative.as_posix()] = {"bytes": info.file_size, "sha256": fingerprint}
                finally:
                    temporary.unlink(missing_ok=True)
    if not inventory:
        raise ValueError("Archive contains no asset files")
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocasa-repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--endpoint", choices=["https://huggingface.co", "https://hf-mirror.com"], default="https://huggingface.co")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in [1, 8]")
    root = args.robocasa_repo / "robocasa/models/assets"
    if not (root / "box_links/box_links_assets.json").is_file():
        parser.error("Selected checkout has no native asset registry")
    args.cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "45")
    os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(endpoint=args.endpoint, token=False)
    jobs = []
    for repo, revision in REPOSITORIES.items():
        metadata = api.dataset_info(repo, revision=revision, files_metadata=True, timeout=25)
        if metadata.sha != revision:
            raise ValueError("Asset repository revision differs from the fixed source")
        for item in metadata.siblings:
            filename = item.rfilename
            if repo.startswith("robocasa/"):
                if filename not in {"textures.zip", "generative_textures.zip", "objaverse.zip", "aigen_objs.zip"}:
                    continue
                destination = root / "objects" if filename in {"objaverse.zip", "aigen_objs.zip"} else root
            else:
                if not filename.endswith(".zip") or not filename.startswith(("fixtures_lightwheel/", "objects_lightwheel/")):
                    continue
                destination = root / ("fixtures" if filename.startswith("fixtures_lightwheel/") else "objects/lightwheel")
            lfs = item.lfs
            checksum = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            if not checksum:
                raise ValueError("Archive has no published LFS SHA256: " + filename)
            jobs.append({"repo": repo, "revision": revision, "filename": filename,
                         "sha256": checksum, "bytes": item.size, "destination": str(destination.resolve())})
    report_path = args.cache / "ASSET_LOCK.json"
    prior = json.loads(report_path.read_text()) if report_path.exists() else {"archives": {}}
    report = {"schema_version": "robocasa-assets-1", "endpoint": args.endpoint,
              "repositories": REPOSITORIES, "archives": dict(prior["archives"]), "complete": False}

    def install(job):
        key = job["repo"] + "/" + job["filename"]
        old = report["archives"].get(key)
        if old and old.get("sha256") == job["sha256"] and old.get("status") == "verified":
            inventory_path = Path(old["inventory_file"])
            if inventory_path.is_file():
                inventory = json.loads(inventory_path.read_text())
                if all((Path(job["destination"]) / p).is_file() and
                       sha256(Path(job["destination"]) / p) == record["sha256"] for p, record in inventory.items()):
                    return key, old
        archive = Path(hf_hub_download(repo_id=job["repo"], repo_type="dataset", revision=job["revision"],
                            filename=job["filename"], local_dir=args.cache / job["repo"].replace("/", "--"),
                            token=False, endpoint=args.endpoint))
        if sha256(archive) != job["sha256"]:
            raise ValueError("Downloaded archive SHA256 mismatch: " + job["filename"])
        inventory = extract_checked(archive, job["destination"])
        record_name = hashlib.sha256(key.encode()).hexdigest() + ".inventory.json"
        inventory_path = args.cache / record_name
        inventory_path.write_text(json.dumps(inventory, sort_keys=True) + "\n")
        return key, dict(job, status="verified", files=len(inventory), inventory_file=str(inventory_path.resolve()))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(install, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            key = job["repo"] + "/" + job["filename"]
            try:
                key, record = future.result()
            except Exception as exc:
                record = dict(job, status="failed", error_type=type(exc).__name__, error=str(exc))
            report["archives"][key] = record
            temporary = report_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n"); temporary.replace(report_path)
            print(json.dumps({"archive": job["filename"], "status": record["status"], "error_type": record.get("error_type")}), flush=True)
    report["complete"] = bool(jobs) and all(report["archives"].get(j["repo"] + "/" + j["filename"], {}).get("status") == "verified" for j in jobs)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "archives": len(jobs), "lock": str(report_path)}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
