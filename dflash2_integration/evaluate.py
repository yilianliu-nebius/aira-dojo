"""Trusted DFlash2 trial evaluation for the upstream AIRA search policy."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ.get("DFLASH2_ROOT", "/home/ubuntu/dflash2-autoresearch-20261007"))
FIELDS = {"kind", "hypothesis", "predicted_metric", "selector_rank", "learning_rate", "selector_loss_weight", "seed"}
DIAGNOSTIC_RATIOS = {
    "token_accuracy": "acc",
    "token_ce_loss": "ce_loss",
    "unary_top1_accuracy": "dflash/hard_label/unary_top1_accuracy",
    "unary_top16_recall": "dflash/hard_label/unary_top16_recall",
    "unary_top16_mass": "dflash/hard_label/unary_top16_mass",
    "teacher_overlap": "dflash/teacher/unary_distribution_overlap",
    "teacher_top1_agreement": "dflash/teacher/unary_top1_agreement",
    "teacher_top16_mass": "dflash/teacher/unary_top16_mass",
    "teacher_overlap_chain_length": "dflash/teacher/unary_overlap_chain_length",
    "selector_self_conditioned_accuracy": "dflash2/selector/self_conditioned_marginal_accuracy",
    "selector_covered_accuracy": "dflash2/selector/teacher_forced_covered_accuracy",
    "selector_greedy_accepted_length": "dflash2/selector/greedy_accepted_length",
    "selector_teacher_agreement": "dflash2/selector/self_conditioned_teacher_argmax_agreement",
    "selector_gold_probability": "dflash2/selector/teacher_forced_covered_gold_probability",
    "selector_loss": "dflash2/selector/loss",
}
POSITION_RATIOS = ("teacher_overlap", "teacher_top1_agreement", "teacher_top16_mass",
                   "unary_top16_recall", "selector_self_conditioned_accuracy",
                   "selector_covered_accuracy", "selector_teacher_agreement", "selector_loss")


def validate_candidate(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError(f"Candidate must have exactly these fields: {sorted(FIELDS)}")
    if value["kind"] not in {"architecture", "recipe"}:
        raise ValueError("kind must be architecture or recipe")
    for name in ("hypothesis", "predicted_metric"):
        if not isinstance(value[name], str) or not 1 <= len(value[name]) <= 4096:
            raise ValueError(f"{name} must be a nonempty bounded string")
    if type(value["selector_rank"]) is not int or value["selector_rank"] not in (256, 384, 512):
        raise ValueError("selector_rank must be 256, 384, or 512")
    if type(value["seed"]) is not int or value["seed"] != 42:
        raise ValueError("The initial paired screen fixes seed=42")
    for name, low, high in (("learning_rate", 1e-5, 1e-4), ("selector_loss_weight", 0.1, 1.0)):
        number = value[name]
        if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{name} is outside its declared range")
    if value["kind"] == "architecture":
        if value["selector_rank"] != 384:
            raise ValueError("Only rank384 is a new architecture within the five-percent growth gate")
        if value["learning_rate"] != 3e-5 or value["selector_loss_weight"] != 0.1:
            raise ValueError("Architecture arms must keep the matched training recipe")
    elif value["selector_rank"] != 256:
        raise ValueError("Recipe arms must keep the released architecture")
    return value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_protected(manifest: dict) -> None:
    for name, expected in manifest.items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f"Protected artifact changed: {name}")


def cleanup_trial(output_dir: Path) -> None:
    trial = str(output_dir.resolve())
    identifiers = subprocess.check_output(
        ["docker", "ps", "--filter", "label=work.task=build-and-start-the-autoresearch",
         "--filter", f"label=aira.trial_dir={trial}", "--format", "{{.ID}}"],
        text=True, timeout=15,
    ).split()
    for identifier in identifiers:
        name = subprocess.check_output(["docker", "inspect", "--format", "{{.Name}}", identifier],
                                       text=True, timeout=15).strip().lstrip("/")
        if not name.startswith("aira-dflash2-"):
            raise RuntimeError(f"Refuse to stop an unexpected trial container: {name}")
        subprocess.run(["docker", "stop", "--timeout", "30", identifier],
                       capture_output=True, text=True, timeout=40, check=True)


def cancel_trial(signum: int, _frame: object) -> None:
    raise RuntimeError(f"Trial cancelled by signal {signum}")


@contextlib.contextmanager
def acquire_gpu(gpus: list[int], deadline: float, output_dir: Path):
    handles = []
    try:
        for gpu in gpus:
            handles.append((gpu, (ROOT / "locks" / f"gpu-{gpu}.lock").open("a+")))
        while time.monotonic() < deadline:
            for gpu, handle in handles:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    yield gpu
                finally:
                    cleanup_trial(output_dir)
                    fcntl.flock(handle, fcntl.LOCK_UN)
                return
            time.sleep(0.5)
        raise TimeoutError("No assigned trial GPU became available")
    finally:
        for _, handle in handles:
            handle.close()


def run(command: list[str], output_dir: Path, name: str, timeout: float) -> subprocess.CompletedProcess:
    with (output_dir / f"{name}.stdout.log").open("w") as out, (output_dir / f"{name}.stderr.log").open("w") as err:
        result = subprocess.run(command, stdout=out, stderr=err, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}; see {output_dir / (name + '.stderr.log')}")
    return result


def last_json(path: Path) -> dict:
    for line in reversed(path.read_text().splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"No JSON result in {path}")


def teacher_matching(trained: dict, train_dir: Path) -> dict:
    """Pool only declared native metric numerators and denominators."""
    result = {}
    for phase in ("before", "after"):
        path = Path(trained["diagnostics"][phase]).resolve(strict=True)
        if not path.is_relative_to(train_dir.resolve()):
            raise RuntimeError("Diagnostic path is outside this trial")
        records = json.loads(path.read_text())
        if not isinstance(records, list) or not 1 <= len(records) <= 64:
            raise RuntimeError("Diagnostic records are invalid")

        def pooled(name: str):
            numerator = denominator = 0.0
            for record in records:
                pair = record["ratio_metrics"].get(name)
                if pair is None:
                    return None
                if (not isinstance(pair, list) or len(pair) != 2
                        or any(type(value) not in (int, float) or not math.isfinite(value) for value in pair)
                        or pair[1] < 0):
                    raise RuntimeError("Native diagnostic ratio is invalid")
                numerator += pair[0]
                denominator += pair[1]
            return numerator / denominator if denominator > 0 else None

        ratios = {alias: pooled(name) for alias, name in DIAGNOSTIC_RATIOS.items()}
        ratios = {alias: value for alias, value in ratios.items() if value is not None}
        positions = {}
        for alias in POSITION_RATIOS:
            family = DIAGNOSTIC_RATIOS[alias].split("/", 1)[1]
            values = [pooled(f"position_{position}/{family}") for position in range(1, 8)]
            if all(value is not None for value in values):
                positions[alias] = values
        result[phase] = {"pooled_ratios": ratios, "position_bins": positions}
    return result


def evaluate(candidate_path: Path, output_dir: Path) -> dict:
    started = time.monotonic()
    candidate = validate_candidate(read_json(candidate_path))
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(ROOT / "runs"):
        raise ValueError("Trial output must be under the assigned runs directory")
    if (output_dir / "result.json").exists() or (output_dir / "training").exists():
        raise ValueError("Refuse to overwrite an earlier trial")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = read_json(ROOT / "protected" / "policy.json")
    manifest = read_json(ROOT / "protected" / "manifest.json")
    verify_protected(manifest)
    save_json(output_dir / "candidate.json", candidate)
    save_json(output_dir / "policy.json", policy)
    result = {"status": "invalid", "useful_output_tokens_per_second": None, "acceptance_length": None,
              "candidate": candidate, "artifacts": {"trial_dir": str(output_dir)}}
    gpus = policy["trial_gpus"]
    if not gpus or any(type(gpu) is not int or gpu not in (3, 4, 5, 6) for gpu in gpus):
        raise ValueError("Policy must assign only authorized trial GPUs 3–6")
    with acquire_gpu(gpus, started + policy["trial_timeout_seconds"], output_dir) as gpu:
        result["gpu"] = gpu
        train_dir = output_dir / "training"
        run([sys.executable, "-m", "dflash2_integration.training", "--candidate", str(output_dir / "candidate.json"),
             "--gpu", str(gpu), "--output-dir", str(train_dir), "--seconds", str(policy["training_seconds"]),
             "--steps", str(policy["training_steps"])], output_dir, "training", policy["training_timeout_seconds"])
        trained = last_json(output_dir / "training.stdout.log")
        if type(trained.get("updates")) is not int or trained["updates"] < 2:
            raise RuntimeError("The trial did not complete two real optimizer updates")
        checkpoint = Path(trained["checkpoint"]).resolve(strict=True)
        if not checkpoint.is_relative_to(output_dir.resolve()):
            raise RuntimeError("Trainer checkpoint is outside this trial")
        config = read_json(checkpoint / "config.json")
        if config.get("architectures") != ["DFlash2DraftModel"]:
            raise RuntimeError("Export is not DFlash2DraftModel")
        if config.get("dflash_config", {}).get("selector_rank") != candidate["selector_rank"]:
            raise RuntimeError("Export selector rank differs from the candidate")
        result["training"] = trained
        result["teacher_matching"] = teacher_matching(trained, train_dir)
        name = "aira-dflash2-trial-" + hashlib.sha256(str(output_dir.resolve()).encode()).hexdigest()[:12]
        port = 28600 + gpu
        ready_path = output_dir / "server.json"
        try:
            run([sys.executable, "-m", "dflash2_integration.serve", "--draft", str(checkpoint), "--gpu", str(gpu),
                 "--port", str(port), "--name", name, "--output", str(ready_path)], output_dir, "serve",
                policy["serve_timeout_seconds"])
            metrics_path = output_dir / "benchmark.json"
            run([sys.executable, "-m", "dflash2_integration.benchmark", "--url", f"http://127.0.0.1:{port}",
                 "--dataset", policy["development_dataset"], "--output", str(metrics_path),
                 "--max-tokens", str(policy["max_output_tokens"])], output_dir, "benchmark",
                policy["benchmark_timeout_seconds"])
            metrics = read_json(metrics_path)["totals"]
            if metrics.get("error_count") != 0 or metrics.get("output_tokens", 0) <= 0:
                raise RuntimeError("Benchmark has errors or no useful output")
            throughput = metrics["tokens_per_second"]
            if not isinstance(throughput, (int, float)) or not math.isfinite(throughput) or throughput <= 0:
                raise RuntimeError("Benchmark throughput is invalid")
            result.update(status="valid", useful_output_tokens_per_second=throughput,
                          acceptance_length=metrics["acceptance_length"], benchmark=metrics)
            baseline = read_json(ROOT / "protected" / "baseline-result.json")
            matched_rate = baseline["useful_output_tokens_per_second"]
            released_rate = baseline["released_reference"]["tokens_per_second"]
            result["comparison"] = {
                "matched_baseline": True,
                "baseline_useful_output_tokens_per_second": matched_rate,
                "delta_useful_output_tokens_per_second": throughput - matched_rate,
                "released_baseline_useful_output_tokens_per_second": released_rate,
                "delta_released_useful_output_tokens_per_second": throughput - released_rate,
            }
            result["artifacts"].update(checkpoint=str(checkpoint), benchmark=str(metrics_path), server=str(ready_path))
        finally:
            stopped = subprocess.run(["docker", "stop", "--timeout", "30", name], capture_output=True,
                                     text=True, timeout=40, check=False)
            result["server_stop_returncode"] = stopped.returncode
    verify_protected(manifest)
    result["wall_seconds"] = time.monotonic() - started
    result["evidence_scope"] = "development screen; not a hidden-test win"
    save_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, cancel_trial)
    signal.signal(signal.SIGINT, cancel_trial)
    try:
        result = evaluate(args.candidate, args.output_dir)
    except Exception as error:
        result = {"status": "invalid", "useful_output_tokens_per_second": None, "acceptance_length": None,
                  "error": f"{type(error).__name__}: {error}", "infrastructure_error": not isinstance(error, ValueError),
                  "artifacts": {"trial_dir": str(args.output_dir)}}
        if args.output_dir.is_dir() and not (args.output_dir / "result.json").exists():
            save_json(args.output_dir / "result.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
