"""Real localhost / subprocess tests; no robotics, NumPy, or GPU dependencies."""
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import textwrap
import unittest

from graspgen_libero_bridge.rpc import WorkerClient


_WORKER = textwrap.dedent("""\
    import argparse
    import os
    from graspgen_libero_bridge.rpc import serve

    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--ready', required=True)
    args = parser.parse_args()

    def handle(request):
        command = request.get('command')
        if command == 'echo':
            return {'value': request['value'], 'pid': os.getpid()}
        if command == 'fail':
            raise ValueError('deliberate RPC test failure')
        raise KeyError('unknown test command')

    serve(handle, args.ready)
""")


class LocalhostRPCContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="libero_rpc_test_")
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        module = folder / "bridge_rpc_test_worker.py"
        module.write_text(_WORKER)
        project = Path(__file__).resolve().parents[1]
        search_path = os.pathsep.join([str(folder), str(project)])
        self.worker = WorkerClient(
            sys.executable, "bridge_rpc_test_worker", folder / "runtime",
            environment={"PYTHONPATH": search_path}, timeout=10,
        )
        self.addCleanup(self.worker.close)

    def test_authenticated_request_runs_in_real_child(self):
        reply = self.worker.request({"command": "echo", "value": {"hello": [1, 2, 3]}}, timeout=5)
        self.assertEqual(reply["value"], {"hello": [1, 2, 3]})
        self.assertEqual(reply["pid"], self.worker.process.pid)
        self.assertNotEqual(reply["pid"], os.getpid())
        self.assertEqual(self.worker.connection["host"], "127.0.0.1")
        mode = stat.S_IMODE(self.worker.ready.stat().st_mode)
        self.assertEqual(mode, 0o600)

    def test_invalid_token_rejected_and_worker_remains_usable(self):
        address = (self.worker.connection["host"], self.worker.connection["port"])
        with socket.create_connection(address, timeout=5) as connection:
            message = {"token": "not-the-token", "request": {"command": "echo", "value": "forbidden"}}
            connection.sendall((json.dumps(message) + "\n").encode())
            with connection.makefile("rb") as reader:
                response = json.loads(reader.readline())
        self.assertFalse(response["ok"])
        self.assertIn("Invalid worker token", response["error"])
        self.assertNotIn("result", response)
        reply = self.worker.request({"command": "echo", "value": "still-running"}, timeout=5)
        self.assertEqual(reply["value"], "still-running")

    def test_handler_exception_propagates_without_killing_worker(self):
        with self.assertRaisesRegex(RuntimeError, "deliberate RPC test failure") as caught:
            self.worker.request({"command": "fail"}, timeout=5)
        self.assertIn("ValueError", str(caught.exception))
        reply = self.worker.request({"command": "echo", "value": "recovered"}, timeout=5)
        self.assertEqual(reply["value"], "recovered")

    def test_shutdown_exits_child_and_removes_ready_file(self):
        reply = self.worker.request({"command": "shutdown"}, timeout=5)
        self.assertTrue(reply["success"])
        self.assertTrue(reply["shutdown"])
        self.assertEqual(self.worker.process.wait(timeout=5), 0)
        self.assertFalse(self.worker.ready.exists())
        # Cleanup remains safe after explicit shutdown and can be called twice.
        self.worker.close()
        self.worker.close()


if __name__ == "__main__":
    unittest.main()
