"""Check the new deployment/launch boundary without robotics dependencies."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from graspgen_libero_bridge.common.layout import package_root, source_fingerprints, source_root
from graspgen_libero_bridge.runtime.cli import main
from graspgen_libero_bridge.runtime.config import load_paths, load_task, task_catalog


class RuntimeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.key = task_catalog()[0]["key"]

    def test_deployment_file_cannot_override_policy_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.json"
            path.write_text(json.dumps({**load_paths(), "lift_height_m": 0.3}))
            with self.assertRaisesRegex(ValueError, "policy parameters"):
                load_task(self.key, paths_file=path)

    def test_custom_deployment_changes_only_five_config_paths(self):
        _, original = load_task(self.key, gpu=2, init=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.json"
            paths = {key: "/another-machine/" + key for key in load_paths()}
            path.write_text(json.dumps(paths))
            _, actual = load_task(self.key, gpu=2, init=1, paths_file=path)
        changed = {key for key in actual if actual[key] != original[key]}
        self.assertEqual(changed, {"libero_repo", "grasp_repo", "graspgen_python", "graspgen_config", "planner_python"})
        self.assertNotIn("simulation_python", actual)
        self.assertEqual(actual["model_gpu"], 2)
        self.assertEqual(actual["init_state_id"], 1)

    def test_dry_run_never_starts_simulation_and_forwards_launch_options(self):
        output = io.StringIO()
        with patch("subprocess.call", side_effect=AssertionError("must not launch")), contextlib.redirect_stdout(output):
            code = main(["run", self.key, "--gpu", "3", "--init", "1", "--tag", "layout-test", "--dry-run"])
        data = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertFalse(data["simulation_started"])
        self.assertIn("layout-test", data["command"])
        self.assertEqual(data["config"]["planner_gpu"], 3)
        self.assertEqual(data["config"]["init_state_id"], 1)

    def test_recursive_provenance_covers_relocated_workers_and_adapter(self):
        fingerprints = source_fingerprints()
        self.assertIn("models/graspgen_worker.py", fingerprints)
        self.assertIn("planning/planner_worker.py", fingerprints)
        self.assertIn("adapters/libero/env_adapter.py", fingerprints)
        self.assertEqual(source_root() / "graspgen_libero_bridge", package_root())


if __name__ == "__main__":
    unittest.main()
