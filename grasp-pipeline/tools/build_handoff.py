#!/usr/bin/env python3
"""Build a source handoff ZIP with a frozen baseline and content manifest.

Only explicit source/document roots are included. Outputs, local config,
third-party checkouts, weights, environments and caches are excluded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('src', 'apps', 'configs', 'contracts', 'tests', 'tools', 'docs', 'migration', 'deploy')
ROOT_FILES = ('README.md', 'run.py', 'pyproject.toml', '.gitignore')
OMIT_PARTS = {'__pycache__', '.pytest_cache', '.git', '.venv', 'runs'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def eligible(path):
    relative = path.relative_to(ROOT)
    return (not any(part in OMIT_PARTS for part in relative.parts)
            and path.suffix not in {'.pyc', '.zip', '.tar', '.gz'}
            and not path.name.endswith('.local.json')
            and path.name not in {'PACKAGE_QA.json', 'REMOTE_PACKAGE_QA.json', 'PACKAGE_SOURCE_SHA256.json'})


def verify_manifest(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / 'MANIFEST.json').read_text())
    expected = manifest['files']
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual != set(expected) | {'MANIFEST.json'}:
        raise ValueError('Manifest file set differs')
    for name, record in expected.items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError('Invalid manifest path: ' + name)
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Manifest content differs: ' + name)
    return {'passed': True, 'files_verified': len(expected), 'release_id': manifest['release_id']}


def build(output_root, release_id, baseline):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}', release_id):
        raise ValueError('release-id must be a plain directory name')
    destination = output_root.resolve() / release_id
    archive = output_root.resolve() / (release_id + '.zip')
    if destination.exists() or archive.exists():
        raise FileExistsError('Refusing to reuse delivery output: ' + str(destination))
    source_files = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    source_files += [p for name in SOURCE_DIRS for p in (ROOT / name).rglob('*') if p.is_file() and eligible(p)]
    baseline_files = [p for p in baseline.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    if not baseline_files or not (baseline / 'FROZEN_SOURCE.json').is_file():
        raise ValueError('Missing frozen v10 baseline')
    for path in source_files + baseline_files:
        if path.is_symlink():
            raise ValueError('Do not package symlink: ' + str(path))
    destination.mkdir(parents=True)
    for path in source_files:
        target = destination / 'grasp-pipeline' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for path in baseline_files:
        target = destination / 'pipeline-organization/baseline/v10' / path.relative_to(baseline)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(ROOT / 'deploy/HANDOFF_README.md', destination / 'README.md')
    files = {p.relative_to(destination).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(destination.rglob('*')) if p.is_file()}
    manifest = {'release_id': release_id, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'frozen_policy_version': 'v10', 'files': files, 'frozen_baseline_files': len(baseline_files),
                'other_benchmark_adapters_implemented': False, 'weights_or_environments_bundled': False,
                'new_task_success_rate_claimed': False}
    (destination / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    verification = verify_manifest(destination)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                handle.write(path, str(Path(release_id) / path.relative_to(destination)))
    fingerprint = sha(archive)
    archive.with_suffix('.zip.sha256').write_text(fingerprint + '  ' + archive.name + '\n')
    return dict(verification, archive=str(archive), archive_bytes=archive.stat().st_size,
                archive_sha256=fingerprint, directory=str(destination),
                baseline_files=len(baseline_files), packaged_python_files=sum(name.endswith('.py') for name in files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--release-id', default='grasp-pipeline-v10-handoff-20261112')
    parser.add_argument('--baseline', type=Path, default=ROOT.parent / 'pipeline-organization/baseline/v10')
    parser.add_argument('--verify', type=Path, help='Verify an extracted package; no file writes')
    args = parser.parse_args()
    if not args.verify and not args.output_root:
        parser.error('--output-root is required for a build')
    try:
        report = verify_manifest(args.verify) if args.verify else build(args.output_root, args.release_id, args.baseline)
    except (OSError, ValueError) as exc:
        print(json.dumps({'passed': False, 'error': str(exc)}))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
