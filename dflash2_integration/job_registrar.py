"""Relay real OMP responses and register actual jobs through short SSH calls.

The fixed remote report paths hold public prompts, completions, and job events.
Two workers call the real local OMP HTTP bridge. SSH reads and response writes
have 15-second limits. No credential copy, long tunnel, model substitution, or
research policy exists here. The work registry receives only actual job IDs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

from dflash2_integration.llm_bridge import DEFAULT_MODEL, JOB_HOST, JOB_TASK, MAX_REQUEST_BYTES, validate_job

SSH_HOST = "autoresearch-h200"
LOCAL_OMP_URL = "http://127.0.0.1:18765/v1/chat/completions"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
UUID_PATTERN = r"[0-9a-f]{32}"

# Fixed code, validated root, decimal cursor, bounded count, and UUID-only
# exclusions are the entire remote read interface. No event supplies a path.
SPOOL_READER = """import fcntl,json,os,pathlib,re,sys
root=pathlib.Path(sys.argv[1])
offset=int(sys.argv[2]); count=int(sys.argv[3]); excluded=set(sys.argv[4].split(','))
reports=root/'reports'; lines=[]
try:
    stream=open(reports/'job-events.jsonl','rb')
except FileNotFoundError:
    pass
else:
    with stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_SH)
        if offset>os.fstat(stream.fileno()).st_size: offset=0
        stream.seek(offset); data=stream.read(1048576)
        complete=data.rfind(b'\\n')+1
        if data and not complete and len(data)==1048576: raise ValueError('oversize event')
        lines=data[:complete].decode('utf-8').splitlines(); offset+=complete
requests=[]
for path in sorted((reports/'llm_requests').glob('*.json')):
    if len(requests)>=count: break
    identifier=path.stem
    if not re.fullmatch('[0-9a-f]{32}',identifier) or identifier in excluded: continue
    if path.is_symlink() or (reports/'llm_responses'/(identifier+'.json')).exists(): continue
    with path.open('rb') as stream: raw=stream.read(2101249)
    if len(raw)>2101248: raise ValueError('oversize request')
    requests.append({'id':identifier,'request':json.loads(raw)})
print(json.dumps({'offset':offset,'lines':lines,'requests':requests}))
"""

# SSH stdin carries a JSON envelope, not shell text. The UUID determines the
# only writable response file. Its directory lies within the fixed reports root.
RESPONSE_WRITER = """import json,os,pathlib,re,sys,tempfile
root=pathlib.Path(sys.argv[1]); identifier=sys.argv[2]
if not re.fullmatch('[0-9a-f]{32}',identifier): raise ValueError('invalid UUID')
raw=sys.stdin.buffer.read(4194305)
if len(raw)>4194304: raise ValueError('oversize response')
body=json.loads(raw)
if body.get('id')!=identifier or body.get('status') not in (200,400,502,504) or not isinstance(body.get('body'),dict): raise ValueError('invalid response')
directory=root/'reports'/'llm_responses'; directory.mkdir(parents=True,exist_ok=True)
with tempfile.NamedTemporaryFile(mode='wb',dir=directory,delete=False) as stream:
    temporary=pathlib.Path(stream.name); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
os.replace(temporary,directory/(identifier+'.json'))
fd=os.open(directory,os.O_RDONLY)
try: os.fsync(fd)
finally: os.close(fd)
print(json.dumps({'id':identifier,'status':'written'}))
"""


def ssh_call(host: str, script: str, arguments: list[str], timeout: float, payload: str | None = None) -> dict:
    command = shlex.join(["python3", "-c", script, *arguments])
    stdin_options = {"stdin": subprocess.DEVNULL} if payload is None else {"input": payload}
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1", host, command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **stdin_options,
        timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("A short SSH relay request failed")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("The short SSH request returned an invalid envelope")
    return value


def read_spool(host: str, root: str, offset: int, slots: int, active: set[str], timeout: float) -> dict:
    payload = ssh_call(host, SPOOL_READER, [root, str(offset), str(slots), ",".join(sorted(active))], timeout)
    if set(payload) != {"offset", "lines", "requests"}:
        raise ValueError("The spool reader returned invalid fields")
    next_offset, lines, requests = payload["offset"], payload["lines"], payload["requests"]
    if isinstance(next_offset, bool) or not isinstance(next_offset, int) or next_offset < 0:
        raise ValueError("The spool reader returned an invalid offset")
    if not isinstance(lines, list) or any(not isinstance(line, str) for line in lines):
        raise ValueError("The spool reader returned invalid event lines")
    if not isinstance(requests, list) or len(requests) > slots:
        raise ValueError("The spool reader returned too many requests")
    for item in requests:
        if not isinstance(item, dict) or set(item) != {"id", "request"} or not isinstance(item["id"], str) or re.fullmatch(UUID_PATTERN, item["id"]) is None:
            raise ValueError("The spool reader returned an invalid request UUID")
    return payload


def validate_event(line: str) -> dict:
    event = json.loads(line)
    fields = {"id", "name", "gpu", "kind", "note"}
    if not isinstance(event, dict) or set(event) != fields | {"host", "task", "time"}:
        raise ValueError("Invalid job event fields")
    if event["host"] != JOB_HOST or event["task"] != JOB_TASK:
        raise ValueError("Unauthorized job host or task")
    timestamp = event["time"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp <= 0:
        raise ValueError("Invalid job timestamp")
    return validate_job({key: event[key] for key in fields}) | {"time": timestamp}


def atomic_json(path: Path, value: dict) -> None:
    temporary = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_cursor(path: Path, root: str, offset: int, registered: set[str]) -> None:
    atomic_json(path, {"root": root, "offset": offset, "registered": sorted(registered)})


def load_cursor(path: Path, root: str) -> tuple[int, set[str]]:
    if not path.exists():
        return 0, set()
    state = json.loads(path.read_text(encoding="utf-8"))
    offset, ids = state.get("offset"), state.get("registered")
    if state.get("root") != root or isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("The local cursor does not match this experiment")
    if not isinstance(ids, list) or any(not isinstance(identifier, str) or re.fullmatch(r"(?:docker:[0-9a-f]{64}|process:[1-9][0-9]{0,19})", identifier) is None for identifier in ids):
        raise ValueError("The local cursor contains invalid job IDs")
    return offset, set(ids)


def register_event(task: str, event: dict, timeout: float) -> None:
    note = f"{event['name']}; GPU {event['gpu']}; launched_at {event['time']}; {event['note']}"
    result = subprocess.run(
        ["work", "job", "add", task, "--id", event["id"], "--host", JOB_HOST, "--kind", event["kind"], "--note", note],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("The work registry rejected the actual job")


def request_body(identifier: str, request: dict) -> tuple[dict, float]:
    if not isinstance(request, dict) or set(request) != {"id", "created_at", "expires_at", "body"} or request["id"] != identifier:
        raise ValueError("Invalid relay request fields")
    created, expires = request["created_at"], request["expires_at"]
    for number in (created, expires):
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
            raise ValueError("Invalid relay timestamps")
    if not 0 < expires - created <= 600.001:
        raise ValueError("Invalid relay request duration")
    body = request["body"]
    if not isinstance(body, dict) or set(body) != {"model", "messages", "timeout_seconds"} or body["model"] != DEFAULT_MODEL:
        raise ValueError("Unauthorized relay model or fields")
    messages = body["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("A relay request needs messages")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
            raise ValueError("Invalid public relay message")
    timeout = body["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("Invalid relay model timeout")
    if len(json.dumps(body, ensure_ascii=False).encode()) > MAX_REQUEST_BYTES:
        raise ValueError("Oversize public relay request")
    return body, expires


def real_completion(body: dict, timeout: float) -> tuple[int, dict]:
    outgoing = Request(
        LOCAL_OMP_URL, data=json.dumps(body | {"timeout_seconds": timeout}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(outgoing, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
    except HTTPError as error:
        raw, status = error.read(MAX_RESPONSE_BYTES + 1), error.code
    if len(raw) > MAX_RESPONSE_BYTES - 256:
        raise RuntimeError("Oversize real OMP response")
    result = json.loads(raw)
    if not isinstance(result, dict) or status not in {200, 400, 502, 504}:
        raise RuntimeError("Invalid local OMP bridge response")
    if status == 200:
        text = result["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("The real OMP response has no assistant content")
    return status, result


def relay_request(host: str, root: str, identifier: str, request: dict, cache_dir: Path, deadline: float, stop: threading.Event) -> bool:
    cached = cache_dir / f"{identifier}.json"
    if cached.exists():
        envelope = json.loads(cached.read_text(encoding="utf-8"))
    else:
        try:
            body, expires = request_body(identifier, request)
            timeout = min(expires - time.time(), deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("The relay request expired")
            status, result = real_completion(body, timeout)
        except (ValueError, KeyError, IndexError, TypeError):
            status, result = 400, {"error": {"message": "The public relay request or model response was invalid"}}
        except TimeoutError:
            status, result = 504, {"error": {"message": "The real model request reached its deadline"}}
        except (OSError, URLError, RuntimeError):
            status, result = 502, {"error": {"message": "The real OMP bridge request failed; no substitute response exists"}}
        envelope = {"id": identifier, "status": status, "body": result}
        atomic_json(cached, envelope)
    if not isinstance(envelope, dict) or envelope.get("id") != identifier or envelope.get("status") not in {200, 400, 502, 504} or not isinstance(envelope.get("body"), dict):
        raise ValueError("The local response cache is invalid")
    payload = json.dumps(envelope, allow_nan=False)
    while not stop.is_set() and time.monotonic() < deadline:
        try:
            result = ssh_call(host, RESPONSE_WRITER, [root, identifier], min(15, deadline - time.monotonic()), payload)
            if result != {"id": identifier, "status": "written"}:
                raise RuntimeError("The short SSH response write did not acknowledge its UUID")
            logging.info("Relayed actual model response %s with HTTP status %s", identifier, envelope["status"])
            return True
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            logging.warning("Short SSH response transfer failed; the real cached response will retry")
            stop.wait(min(5, max(0, deadline - time.monotonic())))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=SSH_HOST, choices=(SSH_HOST,))
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", default=JOB_TASK, choices=(JOB_TASK,))
    parser.add_argument("--seconds", type=float, default=86400)
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local" / "state" / "work" / "job-registrars")
    args = parser.parse_args()
    if re.fullmatch(r"/home/ubuntu/dflash2-autoresearch-[A-Za-z0-9_-]+", args.root) is None:
        parser.error("--root must name an authorized dflash2-autoresearch experiment directory")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state_dir = args.state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    root_hash = hashlib.sha256(args.root.encode()).hexdigest()[:16]
    state_path = state_dir / f"{args.task}-{root_hash}.json"
    cache_dir = state_dir / f"responses-{root_hash}"
    descriptor = os.open(state_path.with_suffix(".lock"), os.O_CREAT | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("A relay already owns this experiment cursor")
        offset, registered = load_cursor(state_path, args.root)
        deadline = time.monotonic() + args.seconds
        stop = threading.Event()
        new_count, error_count, relay_count = 0, 0, 0
        active = {}
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aira-relay")
        print("AIRA short-SSH relay ready; the model remains openai-codex/gpt-6.1-sol", flush=True)
        try:
            while time.monotonic() < deadline:
                for identifier, future in list(active.items()):
                    if future.done():
                        try:
                            relay_count += int(future.result())
                        except Exception:
                            error_count += 1
                            logging.error("A relay worker failed; its UUID remains pending")
                        del active[identifier]
                try:
                    payload = read_spool(args.host, args.root, offset, 2 - len(active), set(active), min(15, deadline - time.monotonic()))
                    for item in payload["requests"]:
                        if item["id"] not in active:
                            active[item["id"]] = executor.submit(relay_request, args.host, args.root, item["id"], item["request"], cache_dir, deadline, stop)
                    for line in payload["lines"]:
                        try:
                            event = validate_event(line)
                        except (ValueError, TypeError, KeyError):
                            error_count += 1
                            logging.error("Rejected an invalid job event; no job was registered")
                            continue
                        identifier = event["kind"] + ":" + event["id"]
                        if identifier in registered:
                            continue
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("The registrar deadline was reached")
                        register_event(args.task, event, min(15, remaining))
                        registered.add(identifier)
                        new_count += 1
                        save_cursor(state_path, args.root, offset, registered)
                        logging.info("Registered actual %s job %s on GPU %s", event["kind"], event["id"], event["gpu"])
                    offset = payload["offset"]
                    save_cursor(state_path, args.root, offset, registered)
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                    error_count += 1
                    logging.warning("Short SSH job/request tracking failed; the actual spool will retry")
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    stop.wait(min(5, remaining))
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            executor.shutdown(wait=True, cancel_futures=True)
            for future in active.values():
                if not future.cancelled():
                    try:
                        relay_count += int(future.result())
                    except Exception:
                        error_count += 1
            save_cursor(state_path, args.root, offset, registered)
        print(json.dumps({"registered_this_run": new_count, "registered_total": len(registered), "relayed_responses": relay_count, "tracking_errors": error_count, "offset": offset, "cursor": str(state_path)}))


if __name__ == "__main__":
    main()
