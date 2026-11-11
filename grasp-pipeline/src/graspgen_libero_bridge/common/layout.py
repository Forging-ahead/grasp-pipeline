"""Checkout paths and recursive source provenance; no robotics dependencies."""
import hashlib
from pathlib import Path


def package_root():
    return Path(__file__).resolve().parents[1]


def source_root():
    return package_root().parent


def project_root():
    return source_root().parent


def source_fingerprints():
    root = package_root()
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}
