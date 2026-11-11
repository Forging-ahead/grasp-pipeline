"""Deployment checks must fail honestly and handoff manifests detect mutation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


deployment = module('check_deployment')
handoff = module('build_handoff')


class DeploymentToolsTests(unittest.TestCase):
    def paths(self, root, create=False):
        paths = dict(libero_repo=str(root / 'libero'), grasp_repo=str(root / 'fork'),
                     graspgen_config=str(root / 'models/panda.yml'),
                     simulation_python=sys.executable, graspgen_python=sys.executable, planner_python=sys.executable)
        if create:
            for name in ['libero/libero/__init__.py', 'third_party/GraspGen/grasp_gen/grasp_server.py']:
                base = Path(paths['libero_repo'] if name.startswith('libero/') else paths['grasp_repo'])
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('')
            for name in ['bddl_files', 'init_files', 'assets']:
                (Path(paths['libero_repo']) / 'libero/libero' / name).mkdir()
            path = Path(paths['graspgen_config']); path.parent.mkdir(); path.write_text('')
        path = root / 'paths.json'; path.write_text(json.dumps(paths))
        return path

    def test_missing_dependencies_are_failure_without_starting_subprocess(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(deployment.subprocess, 'run') as start:
            result = deployment.check_deployment(self.paths(Path(directory)))
            self.assertFalse(result['passed'])
            self.assertFalse(result['environment_created'])
            start.assert_not_called()

    def test_path_check_pass_does_not_claim_model_or_gpu_acceptance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(deployment.subprocess, 'run') as start:
            result = deployment.check_deployment(self.paths(Path(directory), create=True))
            self.assertTrue(result['passed'])
            self.assertFalse(result['gpu_egl_validated'])
            self.assertFalse(result['model_inference_started'])
            self.assertEqual(result['level'], 'paths')
            start.assert_not_called()

    def test_policy_setting_in_paths_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.paths(Path(directory)); data = json.loads(path.read_text())
            data['lift_height_m'] = 0.5; path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): deployment.check_deployment(path)

    def test_runtime_timeout_is_failure_and_does_not_claim_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.paths(Path(directory), create=True)
            with patch.object(deployment.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 0.1)):
                result = deployment.check_deployment(path, probe_runtime=True)
            self.assertFalse(result['passed'])
            self.assertTrue(all(not record['passed'] for record in result['runtime'].values()))

    def bundle(self, root):
        path = root / 'code.txt'; path.write_text('original')
        (root / 'MANIFEST.json').write_text(json.dumps({'release_id': 'test', 'files': {
            'code.txt': {'bytes': path.stat().st_size, 'sha256': handoff.sha(path)}}}))
        self.assertTrue(handoff.verify_manifest(root)['passed'])
        return path

    def test_manifest_rejects_same_size_content_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.bundle(root).write_text('mutation')
            with self.assertRaises(ValueError): handoff.verify_manifest(root)

    def test_manifest_rejects_unrecorded_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.bundle(root); (root / 'extra').write_text('extra')
            with self.assertRaises(ValueError): handoff.verify_manifest(root)

    def test_builder_refuses_to_overwrite_previous_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'existing').mkdir()
            with self.assertRaises(FileExistsError): handoff.build(root, 'existing', root / 'missing-baseline')

    def test_builder_rejects_path_as_release_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError): handoff.build(root, '../other', root)


if __name__ == '__main__':
    unittest.main()
