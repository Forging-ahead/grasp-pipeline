"""Content identities for one frozen adapter version; no simulator imports."""
import hashlib
import json
from pathlib import Path

from graspgen_libero_bridge.common.layout import project_root


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_identity():
    root = project_root()
    files = [p for name in ("src", "apps", "configs") for p in (root / name).rglob("*")
             if p.is_file() and p.suffix in {".py", ".json"} and "__pycache__" not in p.parts
             and not p.name.endswith(".local.json")]
    files += [root / "robocasa.py"]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    return {"source_digest": digest_json(hashes), "files": hashes}
