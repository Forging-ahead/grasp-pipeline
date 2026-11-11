"""Authenticated localhost JSON RPC, with each dependency in its own process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import subprocess
import time
import traceback


def serve(handler, ready_path):
    token = secrets.token_hex(24)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                raw = self.rfile.readline(16 * 1024 * 1024)
                message = json.loads(raw)
                if not secrets.compare_digest(message.pop("token", ""), token):
                    raise PermissionError("Invalid worker token")
                request = message.get("request", message)
                if request.get("command") == "shutdown":
                    result = {"success": True, "shutdown": True}
                    self.server.stop_requested = True
                else:
                    result = handler(request)
                response = {"ok": True, "result": result}
            except Exception as exc:
                traceback.print_exc()
                response = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
            self.wfile.write((json.dumps(response, allow_nan=False) + "\n").encode())

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        server.timeout = 1
        server.stop_requested = False
        ready = Path(ready_path)
        ready.parent.mkdir(parents=True, exist_ok=True)
        temp = ready.with_suffix(".tmp")
        temp.write_text(json.dumps({"host": "127.0.0.1", "port": server.server_address[1], "token": token, "pid": os.getpid()}))
        os.chmod(temp, 0o600)
        temp.replace(ready)
        try:
            while not server.stop_requested:
                server.handle_request()
        finally:
            ready.unlink(missing_ok=True)


class WorkerClient:
    def __init__(self, interpreter, module, directory, *, arguments=(), environment=None, timeout=600):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ready = self.directory / "ready.json"
        self.ready.unlink(missing_ok=True)
        self.log = (self.directory / "worker.log").open("w")
        env = os.environ.copy()
        if environment:
            env.update({k: str(v) for k, v in environment.items()})
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen([interpreter, "-m", module, "--serve", "--ready", str(self.ready), *arguments],
                                        env=env, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + timeout
            while not self.ready.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f"Worker exited {self.process.returncode}: {self.directory / 'worker.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Worker startup timeout: {self.directory}")
                time.sleep(.2)
            self.connection = json.loads(self.ready.read_text())
        except BaseException:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            self.log.close()
            raise

    def request(self, request, timeout=600):
        with socket.create_connection((self.connection["host"], self.connection["port"]), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall((json.dumps({"token": self.connection["token"], "request": request}, allow_nan=False) + "\n").encode())
            with sock.makefile("rb") as handle:
                payload = handle.readline(32 * 1024 * 1024)
            if not payload:
                raise RuntimeError(f"Worker closed RPC connection (exit={self.process.poll()}); see {self.directory / 'worker.log'}")
            response = json.loads(payload)
        if not response["ok"]:
            raise RuntimeError(response.get("error", "Worker failed") + "\n" + response.get("traceback", ""))
        return response["result"]

    def close(self):
        if self.process.poll() is None:
            try:
                self.request({"command": "shutdown"}, timeout=5)
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        self.log.close()
        self.ready.unlink(missing_ok=True)
