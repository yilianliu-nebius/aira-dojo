"""Immutable native-corpus pilot selection. No foreign-model labels enter this path."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

TAPS = (6, 20, 34, 48, 62)
PUBLISHED_TAPS = (5, 19, 33, 47, 61)

def prepare_data(reference_config: str, output: str, *, train_rows: int = 8,
                 development_rows: int = 2, sequence_length: int = 512) -> dict:
    if min(train_rows, development_rows) < 1 or sequence_length % 256:
        raise ValueError("Positive row counts and a sequence length divisible by 256 are required")
    source = Path(reference_config)
    config = json.loads(source.read_text())
    train = config["data"]["train"]
    development = next(iter(config["data"]["validation"].values()))
    if train["location"] == development["location"]:
        raise ValueError("Training and development must use distinct native splits")
    if tuple(train["layer_ids"]) != TAPS or tuple(development["layer_ids"]) != TAPS:
        raise ValueError("The frozen five-tap contract differs")
    result = {"reference_config": str(source), "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "teacher": "Qwen/Qwen3.8-27B", "teacher_precision": "FP8", "tap_ids": list(TAPS),
              "published_tap_ids": list(PUBLISHED_TAPS), "last_hidden_layer_id": 64,
              "sequence_length": sequence_length,
              "selection": "first supervised conversation prefix in each eligible native row, in source order",
              "eligibility": {"maximum_source_rows_per_split": 64, "condition": "native loss_mask contains a real supervised token"},
              "train": {"location": train["location"], "rows": train_rows},
              "development": {"location": development["location"], "rows": development_rows}}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("A frozen data selection already exists with a different contract")
    if not path.exists():
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result
