"""Local OpenAI-compatible bridge to the authenticated OMP CLI.

OMP owns authentication. The bridge never reads credentials. It invokes the
fixed research model without tools, extensions, skills, rules, or a workspace.
The node relay transports public prompts and real completions through files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import uuid

DEFAULT_MODEL = "openai-codex/gpt-6.1-sol"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
JOB_HOST = "89.169.100.246"
JOB_TASK = "build-and-start-the-autoresearch"


def validate_job(request: dict) -> dict:
    """Validate actual launched job IDs for the node spool and local registry."""
    if set(request) != {"id", "name", "gpu", "kind", "note"}:
        raise ValueError("A job needs id, name, gpu, kind, and note")
    kind, identifier = request["kind"], request["id"]
    if kind == "docker":
        if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-fA-F]{64}", identifier) is None:
            raise ValueError("A Docker job needs its complete 64-hex ID")
        identifier = identifier.lower()
    elif kind == "process":
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
            raise ValueError("A process job needs a positive PID")
        identifier = str(identifier)
        if re.fullmatch(r"[1-9][0-9]{0,19}", identifier) is None:
            raise ValueError("A process job needs a positive PID")
    else:
        raise ValueError("A job kind must be docker or process")
    gpu = request["gpu"]
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu not in range(2, 8):
        raise ValueError("A job GPU must be an integer from 2 through 7")
    name, note = request["name"], request["note"]
    if not isinstance(name, str) or not 1 <= len(name) <= 200 or "\x00" in name:
        raise ValueError("A job needs a name of at most 200 characters")
    if not isinstance(note, str) or len(note) > 4000 or "\x00" in note:
        raise ValueError("A job note must be at most 4000 characters")
    return {"id": identifier, "name": name, "gpu": gpu, "kind": kind, "note": note}


def assistant_content(stdout: str) -> tuple[str, dict[str, Any]]:
    """Read assistant text from OMP JSON events, never from logs or thoughts."""
    last_message = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        messages = []
        if event.get("type") == "message_end":
            messages = [event.get("message")]
        elif event.get("type") == "agent_end":
            messages = event.get("messages", [])
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                last_message = message
    if last_message is None or last_message.get("stopReason") in {"error", "aborted"}:
        raise RuntimeError("OMP returned no complete assistant message")
    content = last_message.get("content", [])
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str))
    else:
        text = ""
    if not text.strip():
        raise RuntimeError("OMP returned no assistant text")
    raw_usage = last_message.get("usage") or {}
    usage = {}
    if isinstance(raw_usage, dict):
        for source, target in (("input", "prompt_tokens"), ("output", "completion_tokens"), ("totalTokens", "total_tokens")):
            value = raw_usage.get(source)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[target] = value
    return text, usage


def invoke_omp(messages: list[dict[str, str]], executable: str, timeout: float) -> tuple[str, dict]:
    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    conversation = [message for message in messages if message["role"] != "system"]
    prompt = (
        "Answer the last request in this conversation. Preserve its requested output format.\n" + json.dumps(conversation, ensure_ascii=False)
        if conversation else "Produce the proposal requested by the system message now."
    )
    with tempfile.TemporaryDirectory(prefix="aira-omp-") as temporary:
        prompt_path = Path(temporary) / "request.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        process = subprocess.Popen(
            [executable, "-p", "--mode", "json", "--no-session", "--no-tools", "--no-extensions", "--no-skills", "--no-rules", "--no-title",
             "--model", DEFAULT_MODEL, "--thinking", "high", "--max-time", str(max(1, int(timeout))), "--cwd", temporary,
             "--system-prompt", system or "Answer the supplied research request. Do not use tools.", "@" + str(prompt_path)],
            cwd=temporary, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            try:
                stdout, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("OMP reached its time limit") from error
            if process.returncode != 0:
                raise RuntimeError("OMP exited without a completion")
            return assistant_content(stdout)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()


class BridgeServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], args: argparse.Namespace):
        super().__init__(address, BridgeHandler)
        self.options = args


class BridgeHandler(BaseHTTPRequestHandler):
    server: BridgeServer

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
            self.send_json(200, {"status": "ready", "model": DEFAULT_MODEL})
        else:
            self.send_json(404, {"error": {"message": "Unknown endpoint"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "Unknown endpoint"}})
            return
        self.connection.settimeout(30)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict) or request.get("model", DEFAULT_MODEL) != DEFAULT_MODEL or request.get("stream"):
                raise ValueError("Invalid model or response mode")
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("A nonempty message list is required")
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
                    raise ValueError("Each message needs a role and plain text content")
            timeout = request.get("timeout_seconds", self.server.options.timeout_seconds)
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Invalid timeout")
            text, usage = invoke_omp(messages, self.server.options.omp, min(timeout, self.server.options.timeout_seconds))
            response = {
                "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()), "model": DEFAULT_MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
            if usage:
                response["usage"] = usage
            self.send_json(200, response)
        except ValueError:
            self.send_json(400, {"error": {"message": "Invalid bridge request"}})
        except TimeoutError:
            self.send_json(504, {"error": {"message": "OMP reached its time limit"}})
        except (RuntimeError, OSError):
            self.send_json(502, {"error": {"message": "OMP failed; no substitute response exists"}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--omp", default="omp")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    with BridgeServer(("127.0.0.1", args.port), args) as server:
        print(f"OMP bridge: http://127.0.0.1:{args.port}/v1/chat/completions", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
