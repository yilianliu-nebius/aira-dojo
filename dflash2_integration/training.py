"""Host API for bounded, released-weight SpecForge continuation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import uuid

from dflash2_integration.candidate import validate_candidate

ROOT = Path(os.environ.get("DFLASH2_ROOT", "/home/ubuntu/dflash2-autoresearch-20261008-architecture"))
IMAGE = "sha256:6b454775d5e28e8a16d890fa9f3d7b4f442f41c78ed16fd1551c428cc1ccf80c"
RELEASE = "released-015e795645c74b1a0eeef3b570031fb62e769bc5"


def train(candidate: dict | str, gpu: int, output_dir: str, *, seconds: int = 300, steps: int = 32) -> dict:
    candidate = validate_candidate(json.loads(Path(candidate).read_text()) if isinstance(candidate, str) else candidate, allow_baseline=True)
    if gpu not in (3, 4, 5, 6) or min(seconds, steps) < 2:
        raise ValueError("The trainer requires GPU3/4/5/6 and a nonempty fixed budget")
    for prerequisite in (ROOT / "artifacts/feature-cache-supervised/manifest.json", ROOT / "artifacts/teacher-head-contract.json"):
        if not prerequisite.is_file():
            raise RuntimeError(f"Missing frozen trainer prerequisite: {prerequisite}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("A previous trial directory cannot be overwritten")
    (output / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    name = "aira-dflash2-train-" + uuid.uuid4().hex[:12]
    readonly_mounts = []
    for directory_name in ("artifacts", "deps"):
        directory = (ROOT / directory_name).resolve()
        readonly_mounts.extend(("-v", f"{directory}:{directory}:ro"))
    command = ["sudo", "-n", "docker", "run", "-d", "--init", "--name", name,
               "--label", "work.task=build-and-start-the-autoresearch", "--label", f"aira.trial_dir={output.parent}",
               "--gpus", f"device={gpu}", "--network=host", "--ipc=host", "--ulimit", "memlock=-1:-1",
               "-v", f"{ROOT}:{ROOT}", *readonly_mounts, "-v", "/home/ubuntu/models:/home/ubuntu/models:ro",
               "-e", f"PYTHONPATH={ROOT}/integration:{ROOT}/deps/SpecForge", "-e", "WANDB_MODE=disabled",
               "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONUNBUFFERED=1", "-e", "SPECFORGE_DFLASH_FUSED_HEAD=0",
               IMAGE, "python3", "-m", "dflash2_integration.trainer.runtime", "--root", str(ROOT),
               "--candidate", str(output / "candidate.json"), "--output-dir", str(output),
               "--seconds", str(seconds), "--steps", str(steps)]
    previous = {}
    container = None
    def terminate(signum, frame):
        raise InterruptedError(f"Trainer interrupted by signal {signum}")
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, terminate)
        container = subprocess.check_output(command, text=True).strip()
        receipt = {"id": container, "name": name, "gpu": gpu, "kind": "docker",
                   "note": f"Released DFlash2 continuation; artifacts {output}"}
        (output / "job.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"event": "trainer_started", **receipt}), flush=True)
        bridge = os.environ.get("DFLASH2_JOB_BRIDGE_URL")
        if bridge:
            request = urllib.request.Request(bridge, data=json.dumps(receipt).encode(), headers={"Content-Type": "application/json"})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=15) as response:
                if response.status != 200:
                    raise RuntimeError("Trainer registration failed")
        # Startup has its own cap. The measured update budget is enforced inside the runtime.
        deadline = time.monotonic() + 1800 + seconds
        readiness_reported = False
        while True:
            info = json.loads(subprocess.check_output(["sudo", "-n", "docker", "inspect", container], text=True))[0]
            readiness_path = output / "readiness.json"
            if not readiness_reported and readiness_path.is_file():
                print(json.dumps({"event": "optimizer_readiness", "id": container,
                                  **json.loads(readiness_path.read_text())}), flush=True)
                readiness_reported = True
            if not info["State"]["Running"]:
                if info["State"]["ExitCode"]:
                    raise RuntimeError(f"Trainer exited {info['State']['ExitCode']}; see {output}/training.log")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Trainer startup/update deadline expired")
            time.sleep(2)
        return json.loads((output / "result.json").read_text())
    finally:
        if container:
            subprocess.run(["sudo", "-n", "docker", "stop", "-t", "45", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with (output / "training.log").open("w") as log:
                subprocess.run(["sudo", "-n", "docker", "logs", container], stdout=log, stderr=subprocess.STDOUT)
        for sig, handler in previous.items():
            signal.signal(sig, handler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(train(args.candidate, args.gpu, args.output_dir, seconds=args.seconds, steps=args.steps)))

if __name__ == "__main__":
    main()
