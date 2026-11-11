"""Verify the exact final ZIP without relying on the surrounding work tree."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[2]
archive = ROOT / 'deliverables/grasp-pipeline-v10-handoff-20261112.zip'
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert archive.with_suffix('.zip.sha256').read_text().split()[0] == digest
commands = []
tested = json.loads((ROOT / 'grasp-pipeline/migration/phase5/remote/TESTED_SOURCE_SHA256.json').read_text())
with tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='phase5-final-extract-') as temporary:
    target = Path(temporary)
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        names = handle.namelist()
        tops = {Path(name).parts[0] for name in names}
        assert len(tops) == 1
        assert all((target / name).resolve().is_relative_to(target) for name in names)
        assert not any(any(p in {'__pycache__', '.git', '.venv', 'runs'} for p in Path(name).parts)
                       or name.endswith(('.pth', '.mp4', '.local.json', '.pyc')) for name in names)
        handle.extractall(target)
    bundle = target / tops.pop()
    project = bundle / 'grasp-pipeline'
    manifest = json.loads((bundle / 'MANIFEST.json').read_text())
    for name, expected in tested.items():
        assert hashlib.sha256((project / name).read_bytes()).hexdigest() == expected, name
    template = project / 'deploy/HANDOFF_README.md'
    assert template.read_bytes() == (bundle / 'README.md').read_bytes()
    docs = [path for path in bundle.rglob('*.md')
            if '/pipeline-organization/baseline/' not in str(path) and path != template]
    links = 0
    for path in docs:
        for reference in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if '://' in reference or reference.startswith('#'):
                continue
            assert (path.parent / unquote(reference.split('#')[0])).exists(), (path, reference)
            links += 1
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(project / 'src'),
                       CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
    entries = [
        ('manifest_before', [sys.executable, 'tools/build_handoff.py', '--verify', str(bundle)]),
        ('preservation', [sys.executable, 'tools/verify_preservation.py']),
        ('documentation', [sys.executable, 'tools/check_integration_docs.py']),
        ('runtime_config_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_runtime_config.py', '-v']),
        ('deployment_tool_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_deployment_tools.py', '-v']),
        ('manifest_after', [sys.executable, 'tools/build_handoff.py', '--verify', str(bundle)]),
    ]
    for name, argv in entries:
        result = subprocess.run(argv, cwd=project, env=environment, capture_output=True, text=True, timeout=60)
        commands.append({'name': name, 'returncode': result.returncode})
        assert result.returncode == 0, (name, result.stdout, result.stderr)
    report = {'passed': True, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'archive': archive.name, 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
              'manifest_files_verified': len(manifest['files']), 'frozen_baseline_files': manifest['frozen_baseline_files'],
              'final_code_config_references_match_cpu_tested_files': len(tested), 'independent_temporary_extraction': True,
              'commands': commands, 'test_source_path': 'Extracted package src; no original workspace path',
              'markdown_files_checked': len(docs), 'local_document_links_checked': links,
              'readme_template_verified_at_archive_root': True, 'unwanted_files_found': 0,
              'cpu_tests_on_matching_candidate': 121, 'full_task_rollouts_started': 0, 'gpu_egl_validated': False,
              'scope': 'Exact final ZIP CRC/SHA/manifest, independent extraction, package documentation links, CPU-tested source identity and 12 lightweight tests. 121-test CPU acceptance used byte-identical candidate source, preserved in package.'}
    archive.with_suffix('.validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'grasp-pipeline/migration/phase5/PACKAGE_QA.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
