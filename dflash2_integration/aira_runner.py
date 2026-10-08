"""Run bounded DFlash2 research with the upstream AIRA Greedy policy.

A model proposes one Python literal assignment. The adapter parses the literal
without execution and passes its JSON to an immutable evaluator. Upstream Greedy
owns draft/debug/improve selection, operators, memory, nodes, and the journal.
The evaluator alone owns validity and fitness. Research use follows the upstream
CC BY-NC 4.0 license; retain its attribution and third-party license notices.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
from functools import partial
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler

from jinja2 import Environment, StrictUndefined
from jsonschema import Draft202012Validator, ValidationError

# Upstream config imports require this nonsecret setting even for --help.
# Preserve the deployment's explicit value. This does not create a directory.
os.environ.setdefault("LOGGING_DIR", "/tmp/aira-dflash2-logs")

from dojo.config_dataclasses.operators.memory import MemoryOpConfig
from dojo.config_dataclasses.solver.greedy import GreedySolverConfig
from dojo.core.interpreters.base import ExecutionResult
from dojo.core.solvers.operators.debug import debug_op
from dojo.core.solvers.operators.draft import draft_op
from dojo.core.solvers.operators.improve import improve_op
from dojo.core.solvers.operators.memory import create_memory_op
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
from dojo.core.tasks.constants import (
    AUX_EVAL_INFO, EXECUTION_OUTPUT, TASK_DESCRIPTION, VALID_SOLUTION, VALIDATION_FITNESS,
)
from dojo.solvers.greedy.greedy import Greedy
from dojo.utils.logger import get_logger

ASSETS = Path(__file__).with_name("aira")
MODEL = "openai-codex/gpt-6.1-sol"


class BudgetExpired(RuntimeError):
    """The controller reached the overall wall deadline."""


class Budget:
    def __init__(self, seconds: float):
        self.started = time.monotonic()
        self.deadline = self.started + seconds

    def remaining(self, limit: float | None = None) -> float:
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise BudgetExpired("The research wall deadline was reached")
        return min(seconds, limit) if limit is not None else seconds


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def terminate_evaluator(process: subprocess.Popen) -> tuple[str, str]:
    """Permit evaluator-owned container cleanup before the final kill."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.communicate()
    try:
        return process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()


def stop_on_signal(signum: int, frame: Any) -> None:
    # Do not interrupt the evaluator's 60-second cleanup with a second signal.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise KeyboardInterrupt


def public_comparison(value: Any) -> dict:
    """Expose aggregate comparison evidence, never artifact paths or raw data."""
    result = {
        "matched_baseline": isinstance(value, dict) and value.get("matched_baseline") is True,
        "evidence_scope": "development screen; no isolated paired confidence or multi-seed confirmation",
    }
    if isinstance(value, dict):
        for key in ("baseline_useful_output_tokens_per_second", "delta_useful_output_tokens_per_second",
                    "released_baseline_useful_output_tokens_per_second", "delta_released_useful_output_tokens_per_second"):
            number = value.get(key)
            if not isinstance(number, bool) and isinstance(number, (int, float)) and math.isfinite(number):
                result[key] = number
    return result


def public_teacher_matching(value: Any) -> dict | None:
    """Expose only the declared public aggregates, not arbitrary numeric data."""
    scalar_metrics = {
        "token_accuracy", "token_ce_loss", "unary_top1_accuracy", "unary_top16_recall",
        "unary_top16_mass", "teacher_overlap", "teacher_top1_agreement",
        "teacher_top16_mass", "teacher_overlap_chain_length",
        "selector_self_conditioned_accuracy", "selector_covered_accuracy",
        "selector_greedy_accepted_length", "selector_teacher_agreement",
        "selector_gold_probability", "selector_loss",
    }
    position_metrics = {
        "teacher_overlap", "teacher_top1_agreement", "teacher_top16_mass",
        "unary_top16_recall", "selector_self_conditioned_accuracy",
        "selector_covered_accuracy", "selector_teacher_agreement", "selector_loss",
    }

    def finite_number(number: Any) -> bool:
        return not isinstance(number, bool) and isinstance(number, (int, float)) and math.isfinite(number)

    if not isinstance(value, dict):
        return None
    result = {}
    for phase in ("before", "after"):
        source = value.get(phase)
        if not isinstance(source, dict):
            continue
        public = {}
        ratios = source.get("pooled_ratios")
        if isinstance(ratios, dict):
            ratios = {key: ratios[key] for key in scalar_metrics if key in ratios and finite_number(ratios[key])}
            if ratios:
                public["pooled_ratios"] = ratios
        bins = source.get("position_bins")
        if isinstance(bins, dict):
            bins = {
                key: bins[key] for key in position_metrics if key in bins
                and isinstance(bins[key], list) and 1 <= len(bins[key]) <= 7
                and all(finite_number(number) for number in bins[key])
            }
            if bins:
                public["position_bins"] = bins
        if public:
            result[phase] = public
    return result or None


def candidate_from_code(code: str, validator: Draft202012Validator, kind: str) -> dict:
    """Accept only `candidate = <literal dict>`; never execute model code."""
    if not isinstance(code, str) or len(code) > 65536:
        raise ValueError("The proposal must be at most 65536 characters")
    tree = ast.parse(code, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise ValueError("The proposal must contain exactly one candidate assignment")
    assignment = tree.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        raise ValueError("The proposal must assign one name")
    if assignment.targets[0].id != "candidate":
        raise ValueError("The proposal must assign the name candidate")
    # Only dictionary/list/scalar JSON literals survive this whitelist. In
    # particular, calls, names, imports, comprehensions, sets, and attributes do not.
    allowed = (ast.Dict, ast.List, ast.Constant, ast.UnaryOp, ast.USub, ast.UAdd, ast.Load)
    for node in ast.walk(assignment.value):
        if not isinstance(node, allowed):
            raise ValueError("The candidate must contain JSON-compatible literals only")
        if isinstance(node, ast.Dict):
            keys = [ast.literal_eval(key) for key in node.keys if key is not None]
            if any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
                raise ValueError("The candidate must use unique string keys")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (str, int, float, bool, type(None))):
            raise ValueError("The candidate must contain JSON-compatible scalar values")
    candidate = ast.literal_eval(assignment.value)
    if not isinstance(candidate, dict):
        raise ValueError("The candidate must be a dictionary")
    json.dumps(candidate, allow_nan=False)
    validator.validate(candidate)
    if candidate.get("kind") != kind:
        raise ValueError(f"This research run accepts only {kind} candidates")
    return candidate


class BridgeLLM:
    """Implement the operator LLM interface with a real loopback OMP request."""
    client_content_key = "content"

    def __init__(self, operator: str, bridge_url: str, budget: Budget, timeout: float):
        environment = Environment(undefined=StrictUndefined, autoescape=False)
        self.template = environment.from_string((ASSETS / f"{operator}.jinja").read_text(encoding="utf-8"))
        self.operator = operator
        self.bridge_url = bridge_url
        self.budget = budget
        self.timeout = timeout
        self.calls = 0
        # The bridge is loopback only. Ignore proxy environment variables.
        self.opener = build_opener(ProxyHandler({}))

    def __call__(self, query_data: dict, no_user_message: bool = True) -> tuple[str, dict]:
        timeout = self.budget.remaining(self.timeout)
        messages = [{"role": "system", "content": self.template.render(**query_data)}]
        body = json.dumps({"model": MODEL, "messages": messages, "timeout_seconds": timeout}).encode("utf-8")
        request = Request(self.bridge_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        self.calls += 1
        try:
            with self.opener.open(request, timeout=timeout) as response:
                result = json.loads(response.read(4 * 1024 * 1024))
        except (HTTPError, URLError, TimeoutError) as error:
            if time.monotonic() >= self.budget.deadline:
                raise BudgetExpired("The wall deadline was reached during the model request") from error
            raise RuntimeError("The OMP bridge request failed; no substitute proposal exists") from error
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The OMP bridge returned no model content")
        return content, {
            "usage": result.get("usage", {}) | {"cumulative_num_llm_calls": self.calls, "source": "omp"},
            "prompt_messages": messages, "completion_text": content, "operator": self.operator,
        }


class CandidateTask:
    """Translate the immutable evaluator CLI to upstream `step_task`."""
    def __init__(self, work_dir: Path, command: list[str], kind: str, schema: dict, budget: Budget, max_trials: int, timeout: float, job_bridge_url: str):
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema)
        self.work_dir = work_dir
        self.command = command
        self.kind = kind
        self.budget = budget
        self.max_trials = max_trials
        self.timeout = timeout
        self.trials = 0
        self.consecutive_infrastructure_errors = 0
        self.child_environment = os.environ.copy()
        self.child_environment["DFLASH2_JOB_BRIDGE_URL"] = job_bridge_url

    @staticmethod
    def invalid(reason: str, artifacts: dict, infrastructure_error: bool = True) -> dict:
        return {
            "status": "invalid", "useful_output_tokens_per_second": None,
            "acceptance_length": None, "artifacts": artifacts, "adapter_feedback": reason,
            "infrastructure_error": infrastructure_error,
        }

    def evaluate(self, candidate: dict, trial_dir: Path) -> dict:
        candidate_path = trial_dir / "candidate.json"
        write_json(candidate_path, candidate)
        remaining = self.budget.remaining()
        if remaining <= 60:
            raise BudgetExpired("The wall deadline leaves no evaluator cleanup reserve")
        timeout = min(self.timeout, remaining - 60)
        command = self.command + ["--candidate", str(candidate_path), "--output-dir", str(trial_dir)]
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True, env=self.child_environment,
        )
        try:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                stdout, stderr = terminate_evaluator(process)
                (trial_dir / "evaluator.stdout").write_text(stdout, encoding="utf-8")
                (trial_dir / "evaluator.stderr").write_text(stderr, encoding="utf-8")
                return self.invalid("The evaluator reached its time limit", {"trial_dir": str(trial_dir)})
            (trial_dir / "evaluator.stdout").write_text(stdout, encoding="utf-8")
            (trial_dir / "evaluator.stderr").write_text(stderr, encoding="utf-8")
            if process.returncode != 0:
                return self.invalid(f"The evaluator exited with status {process.returncode}", {"trial_dir": str(trial_dir)})
            try:
                result = json.loads(stdout)
            except json.JSONDecodeError:
                return self.invalid("The evaluator stdout was not one JSON object", {"trial_dir": str(trial_dir)})
            if not isinstance(result, dict) or result.get("status") not in {"valid", "invalid"}:
                return self.invalid("The evaluator returned an invalid status", {"trial_dir": str(trial_dir)})
            if "artifacts" not in result or "acceptance_length" not in result or "useful_output_tokens_per_second" not in result:
                return self.invalid("The evaluator omitted required result fields", {"trial_dir": str(trial_dir)})
            if result["status"] == "valid":
                for key in ("useful_output_tokens_per_second", "acceptance_length"):
                    value = result[key]
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                        return self.invalid(f"The evaluator returned an invalid {key}", {"trial_dir": str(trial_dir)})
                    if key == "useful_output_tokens_per_second" and value == 0:
                        return self.invalid("The evaluator returned zero useful throughput", {"trial_dir": str(trial_dir)})
            return result
        finally:
            if process.poll() is None:
                terminate_evaluator(process)

    def step_task(self, state: dict, code: str) -> tuple[dict, dict]:
        self.budget.remaining()
        if self.trials >= self.max_trials:
            raise BudgetExpired("The maximum number of trials was reached")
        self.trials += 1
        trial_dir = self.work_dir / f"trial-{self.trials:04d}"
        trial_dir.mkdir()
        (trial_dir / "proposal.py").write_text(code or "", encoding="utf-8")
        started = time.monotonic()
        try:
            candidate = candidate_from_code(code, self.validator, self.kind)
        except (ValueError, TypeError, SyntaxError, ValidationError, RecursionError) as error:
            # All error text here comes from public candidate data or its public schema.
            result = self.invalid(f"Candidate rejected: {error}", {"trial_dir": str(trial_dir)}, infrastructure_error=False)
        else:
            result = self.evaluate(candidate, trial_dir)
        infrastructure_error = result["status"] == "invalid" and result.get("infrastructure_error") is True
        self.consecutive_infrastructure_errors = self.consecutive_infrastructure_errors + 1 if infrastructure_error else 0
        write_json(trial_dir / "result.json", result)
        valid = result["status"] == "valid"
        # Do not send raw evaluator logs, artifacts, hidden prompts, or verifier
        # output to the LLM. Memory gets only public aggregate feedback.
        feedback = {key: result.get(key) for key in ("status", "useful_output_tokens_per_second", "acceptance_length")}
        feedback["comparison"] = public_comparison(result.get("comparison"))
        teacher_matching = public_teacher_matching(result.get("teacher_matching"))
        if teacher_matching is not None:
            feedback["teacher_matching"] = teacher_matching
        if "adapter_feedback" in result:
            feedback["adapter_feedback"] = result["adapter_feedback"]
        execution = ExecutionResult(
            term_out=[json.dumps(feedback, allow_nan=False)], exec_time=time.monotonic() - started,
            exit_code=0 if valid else 1,
        )
        return state, {
            EXECUTION_OUTPUT: execution, VALID_SOLUTION: valid,
            VALIDATION_FITNESS: result["useful_output_tokens_per_second"] if valid else None,
            AUX_EVAL_INFO: {
                "acceptance_length": result.get("acceptance_length"), "artifacts": result["artifacts"],
                "trial_dir": str(trial_dir), "comparison": result.get("comparison"),
                "infrastructure_error": infrastructure_error,
                "teacher_matching": teacher_matching,
            },
        }


class DFlashGreedy(Greedy):
    """Keep the upstream search; replace only the task-specific LLM/score I/O."""
    def __init__(self, cfg: GreedySolverConfig, task_info: dict, bridge_url: str, budget: Budget, llm_timeout: float):
        self.bridge_url = bridge_url
        self.budget = budget
        self.llm_timeout = llm_timeout
        super().__init__(cfg, task_info)

    def setup_operators(self) -> None:
        self.memory_op = create_memory_op(self.cfg.memory)
        self.debug_memory_op = create_memory_op(self.cfg.debug_memory)
        self.draft_fn = partial(draft_op, BridgeLLM("draft", self.bridge_url, self.budget, self.llm_timeout), self.cfg, self.memory_op)
        self.improve_fn = partial(improve_op, BridgeLLM("improve", self.bridge_url, self.budget, self.llm_timeout), self.cfg, self.memory_op)
        self.debug_fn = partial(debug_op, BridgeLLM("debug", self.bridge_url, self.budget, self.llm_timeout), self.cfg, self.debug_memory_op)

    def update_data_preview(self, state: dict) -> None:
        # Upstream step calls this even with cfg.data_preview=False. Never use
        # an interpreter or traverse a workspace: hidden data stays inaccessible.
        self.data_preview = "Hidden evaluation data is not available to the research agent."

    def step(self, task: CandidateTask, state: dict) -> tuple[dict, dict]:
        self.budget.remaining()
        if task.consecutive_infrastructure_errors >= 2:
            raise BudgetExpired("Two consecutive infrastructure errors stopped research")
        return super().step(task, state)

    def parse_eval_result(self, node: Any, eval_result: dict) -> None:
        # LLMAnalyze cannot declare validity or overwrite the trusted fitness.
        node.absorb_exec_result(eval_result[EXECUTION_OUTPUT])
        node.analysis = node.term_out
        node.is_buggy = not eval_result[VALID_SOLUTION]
        info = eval_result[AUX_EVAL_INFO]
        if node.is_buggy:
            node.metric = WorstMetricValue(info=info)
        else:
            node.metric = MetricValue(eval_result[VALIDATION_FITNESS], maximize=True, info=info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("architecture", "recipe"))
    parser.add_argument("--work-dir", required=True, type=Path, help="A new directory for this research lane")
    parser.add_argument("--evaluator-command", default=f"{shlex.quote(sys.executable)} -m dflash2_integration.evaluate")
    parser.add_argument("--max-trials", type=int, help="Default: architecture 1, recipe 4; original ceiling: 4")
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--deadline-unix", type=float, help="Shared absolute deadline for separate research lanes")
    parser.add_argument("--bridge-url", default="http://127.0.0.1:18767/v1/chat/completions")
    parser.add_argument("--evaluation-timeout-seconds", type=float, default=14400)
    parser.add_argument("--llm-timeout-seconds", type=float, default=600)
    parser.add_argument("--search-config", type=Path, default=ASSETS / "search.json")
    parser.add_argument("--candidate-schema", type=Path, default=ASSETS / "candidate.schema.json")
    parser.add_argument("--baseline-result", type=Path, help="An operator-verified aggregate baseline result")
    args = parser.parse_args()
    if args.max_trials is None:
        args.max_trials = 1 if args.kind == "architecture" else 4
    if not 1 <= args.max_trials <= 4:
        parser.error("The original trial ceiling permits 1 through 4 trials per lane")
    for name in ("max_hours", "evaluation_timeout_seconds", "llm_timeout_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.deadline_unix is not None and (not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time()):
        parser.error("--deadline-unix must be a finite future timestamp")
    url = urlsplit(args.bridge_url)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"} or url.username or url.password:
        parser.error("--bridge-url must be an unauthenticated loopback HTTP URL")
    if url.path not in {"", "/", "/v1", "/v1/chat/completions"} or url.query or url.fragment:
        parser.error("--bridge-url must name the loopback bridge or its chat endpoint")
    args.bridge_url = url._replace(path="/v1/chat/completions").geturl()
    args.evaluator_command = shlex.split(args.evaluator_command)
    if not args.evaluator_command:
        parser.error("--evaluator-command must name a real evaluator command")
    return args


def main() -> None:
    args = parse_args()
    seconds = args.max_hours * 3600
    if args.deadline_unix is not None:
        seconds = min(seconds, args.deadline_unix - time.time())
    budget = Budget(seconds)
    settings = json.loads(args.search_config.read_text(encoding="utf-8"))
    schema_text = args.candidate_schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    Draft202012Validator.check_schema(schema)
    # Refuse to overwrite a reference tree, an old journal, or another lane.
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    get_logger().config(None)
    random.seed(settings["search_seed"])
    memory = MemoryOpConfig(memory_processor="simple_memory", memory_op_kwargs={
        "include_code": True, "include_buggy_nodes": True, "max_length": 12000,
    })
    cfg = GreedySolverConfig(
        step_limit=args.max_trials + 1, num_drafts=settings["num_drafts"],
        debug_prob=settings["debug_prob"], max_debug_depth=settings["max_debug_depth"],
        max_llm_call_retries=settings["max_llm_call_retries"], available_packages=[], operators={},
        execution_timeout=int(args.evaluation_timeout_seconds), time_limit_secs=max(1, int(budget.remaining())),
        data_preview=False, use_complexity=False, use_test_score=False, export_search_results=False,
        exp_name="dflash2-" + args.kind, checkpoint_path=str(work_dir / "checkpoint"),
        memory=memory, debug_memory=memory,
    )
    baseline = None
    if args.baseline_result is not None:
        baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
        if not isinstance(baseline_result, dict) or baseline_result.get("status") != "valid":
            raise ValueError("The baseline result must be a valid aggregate evaluator result")
        rate = baseline_result.get("useful_output_tokens_per_second")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("The baseline result must contain a finite positive useful throughput")
        baseline = {
            "status": "valid", "useful_output_tokens_per_second": rate,
            "comparison": public_comparison(baseline_result.get("comparison")),
        }
        write_json(work_dir / "baseline_result.json", baseline)
    task_description = (ASSETS / "task.jinja").read_text(encoding="utf-8")
    task_description = Environment(undefined=StrictUndefined).from_string(task_description).render(
        kind=args.kind, candidate_schema=schema_text,
        baseline_context=json.dumps(baseline, allow_nan=False) if baseline else "No verified aggregate baseline result was supplied.",
    )
    manifest = {
        "kind": args.kind, "max_trials": args.max_trials, "max_hours": args.max_hours,
        "deadline_unix": args.deadline_unix,
        "bridge_url": args.bridge_url, "evaluator_command": args.evaluator_command,
        "model": MODEL, "schema_sha256": hashlib.sha256(schema_text.encode()).hexdigest(),
        "solver_class": "dojo.solvers.greedy.greedy.Greedy", "solver_config": asdict(cfg),
        "fitness_authority": "immutable evaluator stdout only", "license": "CC BY-NC 4.0",
        "baseline_reference": baseline,
    }
    write_json(work_dir / "run.json", manifest)
    (work_dir / "candidate.schema.json").write_text(schema_text, encoding="utf-8")
    (work_dir / "task.txt").write_text(task_description, encoding="utf-8")
    job_bridge_url = urlsplit(args.bridge_url)._replace(path="/jobs", query="", fragment="").geturl()
    task = CandidateTask(work_dir, args.evaluator_command, args.kind, schema, budget, args.max_trials, args.evaluation_timeout_seconds, job_bridge_url)
    solver = DFlashGreedy(cfg, {TASK_DESCRIPTION: task_description, "lower_is_better": False}, args.bridge_url, budget, args.llm_timeout_seconds)
    stop_reason = "trial_limit"
    failed = False
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    try:
        solver(task, {})
    except BudgetExpired as error:
        stop_reason = str(error)
    except KeyboardInterrupt:
        stop_reason = "operator_interrupt"
    except Exception as error:
        failed = True
        # Record the error type without remote response bodies or secret text.
        stop_reason = "research_error:" + type(error).__name__
        logging.error("Research stopped: %s", stop_reason)
    finally:
        # Native journal/checkpoint survives a failed bridge or a wall deadline.
        solver.state.running_time = time.monotonic() - budget.started
        solver.save_checkpoint()
        write_json(work_dir / "search_data.json", solver.journal.export_data())
    if task.consecutive_infrastructure_errors >= 2:
        stop_reason = "two_consecutive_infrastructure_errors"
        failed = True
    best = solver.journal.get_best_node()
    best_candidate = None
    if best is not None:
        best_candidate = candidate_from_code(best.code, task.validator, args.kind)
        write_json(work_dir / "best_candidate.json", best_candidate)
    summary = {
        "kind": args.kind, "trials": task.trials, "stop_reason": stop_reason,
        "wall_seconds": time.monotonic() - budget.started,
        "best_useful_output_tokens_per_second": best.metric.value if best is not None else None,
        "best_candidate": best_candidate, "journal": str(work_dir / "checkpoint" / "journal.jsonl"),
        "search_data": str(work_dir / "search_data.json"),
        "result_role": "shortlist; not a verified winner",
        "comparison": best.metric.info.get("comparison") if best is not None else None,
        "evidence_scope": "development screen; no isolated paired confidence or multi-seed confirmation",
    }
    write_json(work_dir / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
