"""Measure stock SGLang output rate, latency, and speculative acceptance."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "ignore_eos": False}


def get_json(endpoint: str, path: str, timeout: float) -> dict:
    with urllib.request.urlopen(endpoint.rstrip("/") + path, timeout=timeout) as response:
        return json.load(response)


def request(endpoint: str, prompt: dict, max_new_tokens: int, timeout: float) -> dict:
    payload = {"sampling_params": {**SAMPLING, "max_new_tokens": max_new_tokens}, "stream": False}
    if "input_ids" in prompt:
        payload["input_ids"] = prompt["input_ids"]
    else:
        payload["text"] = prompt["text"]
    started = time.perf_counter()
    record = {"id": prompt["id"], "error": None}
    try:
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.load(response)
        meta = body["meta_info"]
        tokens = meta["completion_tokens"]
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("The engine returned an invalid completion_tokens count.")
        finish_reason = meta.get("finish_reason", {})
        if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
            raise RuntimeError(f"The engine aborted the request: {finish_reason}")
        rounds = meta.get("spec_verify_ct")
        accepted = meta.get("spec_num_correct_drafts")
        if rounds is not None and (not isinstance(rounds, int) or rounds <= 0):
            raise ValueError("The engine returned an invalid verification count.")
        if accepted is not None and (not isinstance(accepted, int) or accepted < 0):
            raise ValueError("The engine returned an invalid accepted draft count.")
        record.update({
            "output_tokens": tokens, "text": body.get("text", ""),
            "output_ids": body.get("output_ids"), "meta_info": meta,
            "accepted_tokens": accepted, "verification_rounds": rounds,
            "acceptance_length": 1 + accepted / rounds if accepted is not None and rounds else None,
        })
    except urllib.error.HTTPError as error:
        record["error"] = f"HTTP {error.code}: {error.read().decode(errors='replace')[:4096]}"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["latency_seconds"] = time.perf_counter() - started
    record["tokens_per_second"] = record.get("output_tokens", 0) / record["latency_seconds"]
    return record


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def _benchmark(
    endpoint: str, prompts_path: Path, output_path: Path,
    max_new_tokens: int = 2048, concurrency: int = 1, warmup_requests: int = 1,
    timeout: float = 600,
) -> dict:
    """Read the sealed input; write only the report to the output path."""
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("The benchmark endpoint must use loopback.")
    if output_path.resolve() == prompts_path.resolve():
        raise ValueError("The report must not replace the input set.")
    if max_new_tokens < 1 or concurrency < 1 or warmup_requests < 0:
        raise ValueError("Use positive token and concurrency bounds and nonnegative warmup_requests.")
    raw = prompts_path.read_bytes()
    prompts = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not prompts:
        raise ValueError("The input set is empty.")
    for index, prompt in enumerate(prompts):
        prompt.setdefault("id", str(index))
        if ("text" in prompt) == ("input_ids" in prompt):
            raise ValueError("Each prompt needs exactly one text or input_ids field.")
        if "text" in prompt and not isinstance(prompt["text"], str):
            raise ValueError("The text field must be a string.")
        if "input_ids" in prompt and (
            not isinstance(prompt["input_ids"], list)
            or not prompt["input_ids"]
            or any(not isinstance(token, int) or token < 0 for token in prompt["input_ids"])
        ):
            raise ValueError("The input_ids field must be a nonempty list of token IDs.")
    server_info = get_json(endpoint, "/get_server_info", timeout)
    warmups = [request(endpoint, prompts[index % len(prompts)], max_new_tokens, timeout)
               for index in range(warmup_requests)]
    if any(item["error"] for item in warmups):
        raise RuntimeError(f"The warmup request failed: {warmups}")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        requests = list(pool.map(lambda prompt: request(endpoint, prompt, max_new_tokens, timeout), prompts))
    elapsed = time.perf_counter() - started
    successful = [item for item in requests if item["error"] is None]
    tokens = sum(item["output_tokens"] for item in successful)
    natural = [item for item in successful
               if isinstance(item["meta_info"].get("finish_reason"), dict)
               and item["meta_info"]["finish_reason"].get("type") == "stop"]
    useful_tokens = sum(item["output_tokens"] for item in natural)
    speculative = [item for item in successful
                   if item["accepted_tokens"] is not None and item["verification_rounds"] is not None]
    accepted = sum(item["accepted_tokens"] for item in speculative) if speculative else None
    rounds = sum(item["verification_rounds"] for item in speculative) if speculative else None
    latencies = [item["latency_seconds"] for item in successful]
    result = {
        "endpoint": endpoint, "prompts_sha256": hashlib.sha256(raw).hexdigest(),
        "sampling": {**SAMPLING, "max_new_tokens": max_new_tokens},
        "concurrency": concurrency, "warmup_requests": warmup_requests,
        "server_info": server_info, "warmups": warmups,
        "totals": {
            "request_count": len(requests), "successful_requests": len(successful),
            "error_count": len(requests) - len(successful),
            "output_tokens": tokens, "wall_seconds": elapsed,
            "tokens_per_second": tokens / elapsed,
            "useful_output_tokens": useful_tokens,
            "useful_output_tokens_per_second": useful_tokens / elapsed,
            "accepted_tokens": accepted, "verification_rounds": rounds,
            "acceptance_length": 1 + accepted / rounds if accepted is not None and rounds else None,
            "mean_request_acceptance_length": (
                sum(item["acceptance_length"] for item in speculative) / len(speculative)
                if speculative else None
            ),
            "speculative_metrics_requests": len(speculative),
            "latency_p50_seconds": percentile(latencies, 0.5),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "natural_eos_requests": len(natural),
            "capped_requests": sum(
                isinstance(item["meta_info"].get("finish_reason"), dict)
                and item["meta_info"]["finish_reason"].get("type") == "length"
                for item in successful
            ),
        },
        "requests": requests,
        "metric_note": "Output tokens use engine completion_tokens. Useful tokens exclude failed or cap-truncated requests. Acceptance length is 1 + accepted draft tokens / verification rounds. Stock engine verification applies; this report does not establish distribution equivalence.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def benchmark(
    endpoint: str, prompts_path: Path, output_path: Path,
    max_new_tokens: int = 2048, concurrency: int = 1, warmup_requests: int = 1,
    timeout: float = 600,
) -> dict:
    lock = Path("/home/ubuntu/dflash2-autoresearch-20261007/locks/benchmark.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _benchmark(endpoint, prompts_path, output_path, max_new_tokens,
                          concurrency, warmup_requests, timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", "--url", dest="endpoint", required=True)
    parser.add_argument("--prompts", "--dataset", dest="prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", "--max-tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    result = benchmark(args.endpoint, args.prompts, args.output, args.max_new_tokens,
                       args.concurrency, args.warmup_requests, args.timeout)
    print(json.dumps({"output": str(args.output), **result["totals"]}, indent=2))
    if result["totals"]["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
