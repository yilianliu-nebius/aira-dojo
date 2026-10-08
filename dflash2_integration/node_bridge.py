"""Node-local HTTP endpoint with a durable relay to the real local OMP bridge.

Only public research prompts and model completions enter the queue. UUID file
names define the transport scope. Jobs enter a locked, fsynced event spool.
No model, search policy, credentials, teacher settings, or GPU workload changes.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import uuid

from dflash2_integration.llm_bridge import DEFAULT_MODEL, JOB_HOST, JOB_TASK, MAX_REQUEST_BYTES, validate_job

MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def atomic_json(path: Path, body: dict) -> None:
    encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    temporary = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_job(spool: Path, request: dict) -> dict:
    event = validate_job(request) | {"time": time.time(), "host": JOB_HOST, "task": JOB_TASK}
    encoded = (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    spool.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(spool, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "ab") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    directory = os.open(spool.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"status": "recorded", "id": event["id"], "host": JOB_HOST, "task": JOB_TASK, "time": event["time"]}


class NodeServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], args: argparse.Namespace):
        super().__init__(address, NodeHandler)
        self.options = args
        self.reports = args.root / "reports"


class NodeHandler(BaseHTTPRequestHandler):
    server: NodeServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ready", "model": DEFAULT_MODEL, "transport": "durable relay"})
        else:
            self.send_json(404, {"error": {"message": "Unknown endpoint"}})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/jobs"}:
            self.send_json(404, {"error": {"message": "Unknown endpoint"}})
            return
        self.connection.settimeout(30)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("The request must be a JSON object")
            if self.path == "/jobs":
                self.send_json(200, append_job(self.server.reports / "job-events.jsonl", request))
                return
            if request.get("model", DEFAULT_MODEL) != DEFAULT_MODEL or request.get("stream"):
                raise ValueError("Invalid model or response mode")
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("A nonempty message list is required")
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
                    raise ValueError("Each message needs a role and plain text content")
            timeout = request.get("timeout_seconds", self.server.options.timeout_seconds)
            if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Invalid timeout")
            timeout = min(timeout, self.server.options.timeout_seconds)
            identifier = uuid.uuid4().hex
            created = time.time()
            envelope = {
                "id": identifier, "created_at": created, "expires_at": created + timeout,
                "body": {"model": DEFAULT_MODEL, "messages": [{"role": item["role"], "content": item["content"]} for item in messages], "timeout_seconds": timeout},
            }
            atomic_json(self.server.reports / "llm_requests" / f"{identifier}.json", envelope)
            response_path = self.server.reports / "llm_responses" / f"{identifier}.json"
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if response_path.is_file():
                    with response_path.open("rb") as stream:
                        raw = stream.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise RuntimeError("The relay response exceeds its size limit")
                    result = json.loads(raw)
                    if not isinstance(result, dict) or result.get("id") != identifier or result.get("status") not in {200, 400, 502, 504} or not isinstance(result.get("body"), dict):
                        raise RuntimeError("The relay returned an invalid response")
                    if result["status"] == 200:
                        try:
                            text = result["body"]["choices"][0]["message"]["content"]
                        except (KeyError, IndexError, TypeError) as error:
                            raise RuntimeError("The relay returned no real model content") from error
                        if not isinstance(text, str) or not text.strip():
                            raise RuntimeError("The relay returned no real model content")
                    self.send_json(result["status"], result["body"])
                    return
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
            self.send_json(504, {"error": {"message": "The durable relay reached its request deadline"}})
        except ValueError:
            self.send_json(400, {"error": {"message": "Invalid node bridge request"}})
        except (OSError, RuntimeError):
            self.send_json(502, {"error": {"message": "The durable relay failed; no substitute model response exists"}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18768)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    if not args.root.is_dir():
        parser.error("--root must be an existing authorized experiment directory")
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 600:
        parser.error("--timeout-seconds must be finite and between 0 and 600")
    with NodeServer(("127.0.0.1", args.port), args) as server:
        print(f"Node relay: http://127.0.0.1:{args.port}/v1/chat/completions", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
