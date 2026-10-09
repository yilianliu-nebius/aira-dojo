"""Launch the pinned stock SGLang engine with the trusted draft extension."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(os.environ.get("DFLASH2_ROOT", "/home/ubuntu/dflash2-autoresearch-20261008-architecture")).resolve()
IMAGE = "sha256:6b454775d5e28e8a16d890fa9f3d7b4f442f41c78ed16fd1551c428cc1ccf80c"
ENGINE_COMMIT = "5f55db35e926d50676f75b812640ea2410b0fe0e"
DRAFT_REVISION = "015e795645c74b1a0eeef3b570031fb62e769bc5"
DRAFT = (ROOT / "artifacts" / f"released-{DRAFT_REVISION}").resolve()
TARGET = Path("/home/ubuntu/models/Qwen3.8-27B-FP8")
PREFIX = "aira-dflash2-"
INTEGRATION = Path(__file__).resolve().parent
SERVING_PLUGIN = "dflash2_integration.serving_model"
EXTERNAL_MODEL_PACKAGE = "dflash2_integration.serving_models"


def docker(*args: str) -> str:
    return subprocess.check_output(["docker", *args], text=True).strip()


def engine_arguments(args: argparse.Namespace) -> list[str]:
    command = [
        "python", "-m", "sglang.launch_server",
        "--model-path", "/target",
        "--served-model-name", "Qwen/Qwen3.8-27B",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--tp-size", "1", "--dtype", "bfloat16",
        "--attention-backend", "fa3",
        "--mem-fraction-static", str(args.mem_fraction),
        "--context-length", str(args.context_length),
        "--max-total-tokens", str(args.max_total_tokens),
        "--max-running-requests", str(args.max_running_requests),
        "--cuda-graph-bs-decode", "1", "2", "4", "8",
        "--enable-metrics", "--disable-radix-cache",
    ]
    if not args.target_only:
        command += [
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", "/draft",
            "--speculative-num-draft-tokens", "8",
            "--speculative-draft-attention-backend", "fa3",
        ]
    return command


def wait_ready(port: int, timeout: float, name: str) -> dict:
    deadline = time.monotonic() + timeout
    last_error = "The server has no response."
    while time.monotonic() < deadline:
        state = json.loads(docker("inspect", name))[0]["State"]
        if not state["Running"]:
            raise RuntimeError(f"The server exited: {state}. Use docker logs {name}.")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/get_server_info", timeout=5) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(2)
    raise TimeoutError(f"The server is not ready after {timeout}s: {last_error}")


def start(args: argparse.Namespace) -> dict:
    if args.gpu not in range(2, 8):
        raise ValueError("Teachers reserve GPUs 0 and 1. Use an assigned GPU from 2 through 7.")
    if not args.name.startswith(PREFIX):
        raise ValueError(f"The container name must start with {PREFIX}.")
    root = args.root.resolve()
    if root != ROOT:
        raise ValueError(f"New remote artifacts must stay under {ROOT}.")
    draft = args.draft.resolve()
    if not args.target_only:
        allowed_draft_roots = (root, (root / "artifacts").resolve(), (root / "controls").resolve())
        if not any(draft.is_relative_to(directory) for directory in allowed_draft_roots):
            raise ValueError("The draft must be under the experiment root or a pinned read-only control root.")
        config = json.loads((draft / "config.json").read_text())
        if config.get("architectures") != ["DFlash2DraftModel"]:
            raise ValueError("The draft must use DFlash2DraftModel.")
    bridge = os.environ.get("DFLASH2_JOB_BRIDGE_URL")
    if bridge:
        parsed = urllib.parse.urlparse(bridge)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.port != 18768 or parsed.path != "/jobs" or parsed.query or parsed.fragment):
            raise ValueError("The job bridge must use loopback HTTP port 18768 and the /jobs path.")
    output = args.output.resolve() if args.output else None
    if output and not output.is_relative_to(root):
        raise ValueError("The launch report must be under the experiment root.")
    cache = root / "runtime" / args.name
    cache.mkdir(parents=True, exist_ok=True)
    command = [
        "run", "--detach", "--init", "--name", args.name,
        "--label", "work.task=build-and-start-the-autoresearch",
        "--label", "aira.role=trial" if args.name.startswith(PREFIX + "trial") else "aira.role=baseline",
        "--gpus", f"device={args.gpu}", "--network", "host",
        "--shm-size", "16g", "--memory", "96g", "--cpus", "16",
        "-v", f"{args.target.resolve()}:/target:ro",
        "-v", f"{cache}:/runtime",
        "-e", "HF_HUB_OFFLINE=1", "-e", "HF_HOME=/runtime/hf",
        "-e", "TORCHINDUCTOR_CACHE_DIR=/runtime/torchinductor",
        "-e", "TRITON_CACHE_DIR=/runtime/triton",
        "--entrypoint", "python",
    ]
    if args.name.startswith(PREFIX + "trial") and output:
        command += ["--label", f"aira.trial_dir={output.parent}"]
    if not args.target_only:
        command += [
            "-v", f"{draft}:/draft:ro",
            "-v", f"{INTEGRATION}:/integration/dflash2_integration:ro",
            "-e", "PYTHONPATH=/integration",
            "-e", f"SGLANG_EXTERNAL_MODEL_PACKAGE={EXTERNAL_MODEL_PACKAGE}",
        ]
    engine = engine_arguments(args)
    command += [IMAGE, *engine[1:]]
    container_id = docker(*command)
    state = json.loads(docker("inspect", args.name))[0]["State"]
    record = {
        "container_id": container_id, "pid": state["Pid"], "name": args.name,
        "gpu": args.gpu, "endpoint": f"http://127.0.0.1:{args.port}",
        "image": IMAGE, "engine_commit": ENGINE_COMMIT,
        "draft_revision": DRAFT_REVISION if not args.target_only and draft == DRAFT else None,
        "draft_path": None if args.target_only else str(draft),
        "target_path": str(args.target.resolve()), "target_only": args.target_only,
        "engine_arguments": engine,
        "serving_plugin": None if args.target_only else SERVING_PLUGIN,
        "external_model_package": None if args.target_only else EXTERNAL_MODEL_PACKAGE,
        "integration_path": None if args.target_only else str(INTEGRATION),
    }
    (cache / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if bridge:
        job = {"id": container_id, "name": args.name, "gpu": args.gpu, "kind": "docker",
               "note": f"Stock SGLang {ENGINE_COMMIT}; launch metadata {cache / 'launch.json'}; docker logs {args.name}"}
        req = urllib.request.Request(bridge, data=json.dumps(job).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"The job bridge returned HTTP {response.status}, not HTTP 200.")
            record["job_registration"] = json.load(response)
    if args.wait_seconds:
        record["server_info"] = wait_ready(args.port, args.wait_seconds, args.name)
        (cache / "ready.json").write_text(json.dumps(record, indent=2) + "\n")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    launch = subparsers.add_parser("start")
    launch.add_argument("--draft", type=Path, default=DRAFT)
    launch.add_argument("--target", type=Path, default=TARGET)
    launch.add_argument("--root", type=Path, default=ROOT)
    launch.add_argument("--name", default=PREFIX + "released-baseline")
    launch.add_argument("--port", type=int, default=28522)
    launch.add_argument("--gpu", type=int, default=2)
    launch.add_argument("--target-only", action="store_true")
    launch.add_argument("--mem-fraction", type=float, default=0.65)
    launch.add_argument("--context-length", type=int, default=8192)
    launch.add_argument("--max-total-tokens", type=int, default=16384)
    launch.add_argument("--max-running-requests", type=int, default=8)
    launch.add_argument("--wait-seconds", type=float, default=900)
    launch.add_argument("--output", type=Path)
    stop = subparsers.add_parser("stop")
    stop.add_argument("--name", required=True)
    argv = sys.argv[1:]
    if argv and argv[0].startswith("--"):
        argv.insert(0, "start")
    args = parser.parse_args(argv)
    if args.action == "start":
        print(json.dumps(start(args), indent=2))
    else:
        if not args.name.startswith(PREFIX):
            parser.error(f"The container name must start with {PREFIX}.")
        labels = json.loads(docker("inspect", args.name))[0]["Config"]["Labels"]
        if labels.get("work.task") != "build-and-start-the-autoresearch":
            parser.error("The container does not belong to this work task.")
        print(docker("stop", args.name))


if __name__ == "__main__":
    main()
