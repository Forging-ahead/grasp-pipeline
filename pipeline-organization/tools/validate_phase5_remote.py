"""Verify an independently extracted handoff candidate, using existing CPU deps."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(args.archive) as archive:
    if archive.testzip() is not None:
        raise ValueError('Archive CRC mismatch')
    roots = {Path(name).parts[0] for name in archive.namelist()}
    if len(roots) != 1 or any(not (output / name).resolve().is_relative_to(output) for name in archive.namelist()):
        raise ValueError('Invalid archive paths')
    archive.extractall(output)
bundle = output / roots.pop()
project = bundle / 'grasp-pipeline'
environment = dict(os.environ, PYTHONPATH=str(project / 'src'), CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                   PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                   OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
spec = importlib.util.find_spec('robosuite')
environment['ROBOSUITE_ASSETS'] = str(Path(spec.origin).parent / 'models/assets')
paths = json.loads((project / 'configs/runtime/paths.server.json').read_text())
commands = [
    ('manifest_before', [sys.executable, str(project / 'tools/build_handoff.py'), '--verify', str(bundle)]),
    ('preservation', [sys.executable, 'tools/verify_preservation.py', '--report', str(output / 'PRESERVATION.json')]),
    ('contracts', [sys.executable, 'tools/check_contracts.py', '--report', str(output / 'CONTRACT_VALIDATION.json')]),
    ('integration_docs', [sys.executable, 'tools/check_integration_docs.py', '--schema', '--report', str(output / 'DOCUMENTATION_QA.json')]),
    ('tests', [sys.executable, '-m', 'pytest', '-q', 'tests', '-p', 'no:cacheprovider', '--junitxml=' + str(output / 'tests.xml')]),
    ('simulation_help', [paths['simulation_python'], str(project / 'run.py'), '--help']),
    ('model_help', [paths['graspgen_python'], '-m', 'graspgen_libero_bridge.models.graspgen_worker', '--help']),
    ('planner_help', [paths['planner_python'], '-m', 'graspgen_libero_bridge.planning.planner_worker', '--help']),
    ('manifest_after', [sys.executable, str(project / 'tools/build_handoff.py'), '--verify', str(bundle)]),
]
results = []
for label, argv in commands:
    with (output / (label + '.log')).open('w') as log:
        try:
            result = subprocess.run(argv, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=240)
            record = {'label': label, 'returncode': result.returncode}
        except subprocess.TimeoutExpired:
            record = {'label': label, 'returncode': None, 'error': 'timeout'}
    results.append(record)
    print(json.dumps({**record, 'tail': (output / (label + '.log')).read_text()[-700:]}), flush=True)
counts = {'passed': 0, 'failed': 0, 'error': 0, 'skipped': 0}
for case in ET.parse(output / 'tests.xml').iter('testcase'):
    status = next((s for s in ('failure', 'error', 'skipped') if case.find(s) is not None), 'passed')
    counts['failed' if status == 'failure' else status] += 1
files = [p for base in ('src', 'apps', 'configs', 'contracts', 'tests', 'tools', 'deploy')
         for p in (project / base).rglob('*') if p.is_file() and p.suffix in ('.py', '.json')]
files.extend([project / 'run.py', project / 'pyproject.toml'])
fingerprints = {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(output / 'TESTED_SOURCE_SHA256.json').write_text(json.dumps(fingerprints, indent=2) + '\n')
report = {'passed': all(r['returncode'] == 0 for r in results) and counts == {'passed': 121, 'failed': 0, 'error': 0, 'skipped': 0},
          'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
          'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
          'independent_extraction_root': str(bundle), 'commands': results, 'tests': counts,
          'tested_source_config_reference_files': len(fingerprints), 'gpu_visible': '',
          'full_task_rollouts_started': 0, 'model_inference_started': False,
          'scope': 'New extracted source package with existing server CPU dependencies; not a fresh-install or GPU/trajectory validation'}
(output / 'CPU_VALIDATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
raise SystemExit(0 if report['passed'] else 1)
