"""Capture frozen native FP8 features through the qualified Papyrax/Mooncake client."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .prepare_data import prepare_data

def capture(reference_config: str, output_dir: str, *, train_rows: int = 8,
            development_rows: int = 2, sequence_length: int = 512) -> dict:
    import jax
    import numpy as np
    from papyrax.training.dataset.config import YtDisaggregatedPrefillDatasetConfig
    from papyrax.training.dataset.disaggregated_prefill_dataset import YtDisaggregatedPrefillDataset
    from papyrax.training.dataset.yt_dataset import YtDatasetBase
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    selection = prepare_data(reference_config, str(root / "selection.json"), train_rows=train_rows,
                             development_rows=development_rows, sequence_length=sequence_length)
    complete = root / "manifest.json"
    if complete.exists():
        manifest = json.loads(complete.read_text())
        for record in manifest["records"]:
            path = root / record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("The frozen cache checksum differs")
        return manifest
    raw = json.loads(Path(reference_config).read_text())
    class BoundedNativeDataset(YtDisaggregatedPrefillDataset):
        @property
        def seq_len(self):
            return sequence_length
        def _deserialize_ftapi_tokenized_row(self, tree):
            tree = YtDatasetBase._deserialize(tree)
            limit = sequence_length + 1 if "attention_mask" in tree else sequence_length
            token_fields = {"input_ids", "labels", "attention_mask", "target_ids", "segment_ids", "loss_mask", "positions"}
            tree = {key: np.asarray(value)[:limit] if key in token_fields else value
                    for key, value in tree.items()}
            return super()._deserialize_ftapi_tokenized_row(tree)
    records = []
    skipped_empty_indices = {"train": [], "development": []}
    for split in ("train", "development"):
        config = dict(raw["data"]["train"] if split == "train" else next(iter(raw["data"]["validation"].values())))
        config.update(batch_size=1, max_in_flight_requests=1, request_id_prefix="aira-frozen-" + split,
                      max_in_flight_activation_bytes=512 * 1024**2,
                      max_activation_bytes_per_request=512 * 1024**2)
        dataset = BoundedNativeDataset(YtDisaggregatedPrefillDatasetConfig.model_validate(config))
        selected = 0
        try:
            for index in range(selection["eligibility"]["maximum_source_rows_per_split"]):
                acquired = False
                try:
                    fetched = dataset.get_batch(index)
                    acquired = True
                    batch = {key: np.array(jax.device_get(value), copy=True)
                             for key, value in fetched.items()}
                    valid_segments = batch["segment_ids"][batch["loss_mask"].astype(bool)]
                    if not valid_segments.size:
                        skipped_empty_indices[split].append(index)
                        continue
                    for key in ("target_hidden_states", "intermediate_outputs"):
                        batch[key] = batch[key].astype(np.float32)
                    selected_segment = valid_segments[0]
                    keep = batch["segment_ids"] == selected_segment
                    batch["loss_mask"] = batch["loss_mask"].astype(bool) & keep
                    positions = np.flatnonzero(keep[0])
                    batch = {key: value[:, positions] for key, value in batch.items()}
                    for key in ("target_hidden_states", "intermediate_outputs"):
                        if not np.isfinite(batch[key]).all():
                            raise ValueError(f"Nonfinite {key} in {split} row {index}")
                    name = f"{split}-{selected:04d}.npz"
                    path = root / name
                    np.savez(path, **batch)
                    records.append({"split": split, "row": index, "source_index": index,
                                    "selection_index": selected, "segment_id": int(selected_segment), "file": name,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "supervised_tokens": int(batch["loss_mask"].sum()),
                                    "shapes": {key: list(value.shape) for key, value in batch.items()},
                                    "dtypes": {key: str(value.dtype) for key, value in batch.items()}})
                    selected += 1
                    print(json.dumps({"event": "finite_cache_record", "split": split, "source_index": index,
                                      "selection_index": selected - 1, "file": str(path)}), flush=True)
                finally:
                    if acquired:
                        dataset.release_train_batch()
                if selected == selection[split]["rows"]:
                    break
            if selected != selection[split]["rows"]:
                raise ValueError(f"Only {selected} eligible {split} rows in the fixed 64-row scan")
        finally:
            dataset.close()
    manifest = {"selection": selection, "finite": True, "records": records,
                "skipped_empty_indices": skipped_empty_indices}
    complete.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-rows", type=int, default=8)
    parser.add_argument("--development-rows", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(capture(args.reference_config, args.output_dir, train_rows=args.train_rows,
                             development_rows=args.development_rows, sequence_length=args.sequence_length)))

if __name__ == "__main__":
    main()
