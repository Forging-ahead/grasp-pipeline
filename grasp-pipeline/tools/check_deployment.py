#!/usr/bin/env python3
"""Read-only deployment preflight; never creates an env or loads model weights.

--probe-runtime imports CPU modules through the three configured interpreters.
CUDA is hidden in every probe. Passing is not GPU/EGL or task acceptance.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graspgen_libero_bridge.runtime.config import load_paths

PROBE = r'''
import hashlib, importlib, importlib.metadata, json, os, sys
from pathlib import Path
options = json.loads(sys.argv[1])
role = options['role']
result = {'role': role, 'python': sys.version, 'executable': sys.executable,
          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'modules': {}, 'errors': []}
for name in options['modules']:
    try:
        module = importlib.import_module(name)
        result['modules'][name] = {'version': str(getattr(module, '__version__', 'unknown')),
                                  'path': getattr(module, '__file__', None)}
    except Exception as exc:
        result['errors'].append(name + ': ' + type(exc).__name__ + ': ' + str(exc))
torch = sys.modules.get('torch')
result['cuda_initialized'] = bool(torch and torch.cuda.is_initialized())
result['distributions'] = sorted(
    [{'name': d.metadata.get('Name', 'unknown'), 'version': d.version} for d in importlib.metadata.distributions()],
    key=lambda d: (d['name'].lower(), d['version']))
if role == 'graspgen':
    try:
        from omegaconf import OmegaConf
        config_path = Path(options['model_config'])
        cfg = OmegaConf.load(config_path)
        names = [str(cfg.data.gripper_name), str(cfg.diffusion.gripper_name), str(cfg.discriminator.gripper_name)]
        if names != ['franka_panda'] * 3 or int(cfg.data.num_points) != 2048:
            raise ValueError('Expected the official 2048-point Panda configuration')
        result['model_config'] = {'grippers': names, 'input_points': int(cfg.data.num_points)}
        result['checkpoints'] = {}
        for label, name in [('generator', cfg.eval.checkpoint), ('discriminator', cfg.discriminator.checkpoint)]:
            path = config_path.parent / str(name)
            record = {'path': str(path), 'exists': path.is_file()}
            if not path.is_file():
                raise FileNotFoundError(str(path))
            if options['hash_weights']:
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(chunk)
                record['sha256'] = digest.hexdigest()
                if record['sha256'] != options['expected_weights'][label]:
                    raise ValueError('Baseline weight hash differs: ' + label)
            result['checkpoints'][label] = record
    except Exception as exc:
        result['errors'].append('model_config: ' + type(exc).__name__ + ': ' + str(exc))
if result['cuda_initialized']:
    result['errors'].append('Unexpected CUDA initialization in CPU-only preflight')
result['passed'] = not result['errors']
print('GRASP_DEPLOY_JSON=' + json.dumps(result))
'''


def check_deployment(paths_file, *, probe_runtime=False, hash_weights=False, timeout=60):
    paths = load_paths(paths_file)
    checks = []

    def check(name, passed, detail):
        checks.append(dict(name=name, passed=bool(passed), detail=str(detail)))

    for name in ['simulation_python', 'graspgen_python', 'planner_python']:
        path = Path(paths[name])
        check(name, path.is_file() and os.access(path, os.X_OK), path)
    for name, suffix in [
        ('libero_repo', 'libero/libero/__init__.py'),
        ('libero_repo', 'libero/libero/bddl_files'),
        ('libero_repo', 'libero/libero/init_files'),
        ('libero_repo', 'libero/libero/assets'),
        ('grasp_repo', 'third_party/GraspGen/grasp_gen/grasp_server.py'),
    ]:
        path = Path(paths[name]) / suffix
        check(name + ':' + suffix, path.exists(), path)
    check('graspgen_config', Path(paths['graspgen_config']).is_file(), paths['graspgen_config'])
    reference = json.loads((ROOT / 'deploy/reference/dependencies.v10.json').read_text())
    weights = reference['weight_sha256']
    runtime = {}
    if probe_runtime:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                           PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        roles = {'simulation': 'simulation_python', 'graspgen': 'graspgen_python', 'curobo': 'planner_python'}
        for role, key in roles.items():
            options = dict(role=role, modules=list(reference['roles'][role]['modules']),
                           model_config=paths['graspgen_config'], hash_weights=hash_weights, expected_weights=weights)
            try:
                proc = subprocess.run([paths[key], '-c', PROBE, json.dumps(options)], cwd=ROOT,
                                      env=environment, capture_output=True, text=True, timeout=timeout)
                lines = [line for line in proc.stdout.splitlines() if line.startswith('GRASP_DEPLOY_JSON=')]
                if proc.returncode or len(lines) != 1:
                    raise RuntimeError('exit=' + str(proc.returncode) + '; ' + (proc.stderr or proc.stdout)[-2000:])
                result = json.loads(lines[0].split('=', 1)[1])
                runtime[role] = result
                check(role + ':cpu_imports', result['passed'], '; '.join(result['errors']) or 'CPU imports/config passed')
                differences = {name: {'expected': expected, 'actual': result['modules'].get(name, {}).get('version')}
                               for name, expected in reference['roles'][role]['modules'].items()
                               if result['modules'].get(name, {}).get('version') != expected}
                check(role + ':baseline_core_versions', not differences, json.dumps(differences))
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                runtime[role] = {'passed': False, 'error': type(exc).__name__ + ': ' + str(exc)}
                check(role + ':runtime_probe', False, runtime[role]['error'])
    return dict(passed=all(item['passed'] for item in checks), checked_at_utc=datetime.now(timezone.utc).isoformat(),
                paths_file=str(Path(paths_file).resolve()), paths=paths, checks=checks, runtime=runtime,
                probe_runtime=probe_runtime, weight_hashes_checked=hash_weights and probe_runtime,
                level='paths+cpu_imports+weight_identity' if hash_weights else ('paths+cpu_imports' if probe_runtime else 'paths'),
                gpu_egl_validated=False, model_inference_started=False, environment_created=False,
                full_task_rollouts_started=0,
                limitations=['Directory presence is not full asset or initial-state verification.',
                             'CPU imports do not validate CUDA kernels, full GraspGen imports, EGL rendering, or successful tasks.',
                             'Core version comparison is not a complete install lockfile.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, default=ROOT / 'configs/runtime/paths.server.json')
    parser.add_argument('--probe-runtime', action='store_true')
    parser.add_argument('--hash-weights', action='store_true')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.hash_weights and not args.probe_runtime:
        parser.error('--hash-weights requires --probe-runtime to resolve the active model config')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    try:
        report = check_deployment(args.paths, probe_runtime=args.probe_runtime,
                                  hash_weights=args.hash_weights, timeout=args.timeout)
    except (OSError, ValueError) as exc:
        report = {'passed': False, 'error': type(exc).__name__ + ': ' + str(exc),
                  'environment_created': False, 'model_inference_started': False, 'full_task_rollouts_started': 0}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
