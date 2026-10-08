"""Pinned SpecForge DFlash2 continuation; free codebook recipe, not recovered projections."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

SPEC_FORGE_COMMIT = "fdcc2eddf1b56b5b24a5a15d3abcb366adfeb5aa"

def serial(value):
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seconds", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output_dir)
    started = time.perf_counter()
    import numpy as np
    import torch
    from torch import nn
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3Config
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
    from dflash2_integration.training import validate_candidate, RELEASE
    candidate = validate_candidate(json.loads(Path(args.candidate).read_text()))
    actual_commit = (root / "deps/SpecForge/.git/refs/heads/main").read_text().strip()
    if actual_commit != SPEC_FORGE_COMMIT:
        raise ValueError("The immutable SpecForge commit differs")
    torch.manual_seed(candidate["seed"])
    torch.cuda.manual_seed_all(candidate["seed"])
    device = "cuda:0"
    release = root / "artifacts" / RELEASE
    config = Qwen3Config.from_pretrained(release, local_files_only=True)
    config._attn_implementation = "sdpa"
    model = DFlash2DraftModel(config).to(dtype=torch.bfloat16)
    released = load_file(str(release / "model.safetensors"))
    # No absent-key tolerance and no codebook reinitialization.
    model.load_state_dict(released, strict=True)
    del released
    model.to(device)
    target = Path("/home/ubuntu/models/Qwen3.8-27B-FP8")
    contract = json.loads((root / "artifacts/teacher-head-contract.json").read_text())
    def tensor(key):
        matches = []
        for file in target.glob("*.safetensors"):
            with safe_open(file, framework="pt", device="cpu") as source:
                if key in source.keys():
                    matches.append(source.get_tensor(key))
        if len(matches) != 1:
            raise ValueError(f"Expected one frozen target tensor {key}; found {len(matches)}")
        value = matches[0]
        if value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Frozen target tensor {key} is not a dense supported tensor")
        return value.to(device=device, dtype=torch.bfloat16)
    embedding = nn.Embedding.from_pretrained(tensor(contract["embedding_key"]), freeze=True)
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device="meta", dtype=torch.bfloat16)
    head.weight = nn.Parameter(tensor(contract["head_key"]), requires_grad=False)
    norm_weight = None if contract["last_hidden_normalized"] else tensor(contract["norm_key"])
    cache = root / "artifacts/feature-cache-supervised"
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest["finite"] or manifest["selection"]["teacher_precision"] != "FP8":
        raise ValueError("The shared cache must contain finite FP8 teacher features")
    def batch(record):
        file = cache / record["file"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("The immutable cache checksum differs")
        with np.load(file, allow_pickle=False) as data:
            ids = torch.from_numpy(data["input_ids"].astype(np.int64)).to(device)
            # Papyrax labels[t] and loss_mask[t] refer to token t+1. SpecForge uses token-index masks.
            mask = torch.from_numpy(data["loss_mask"].astype(np.float32)).to(device)
            aligned_mask = torch.zeros_like(mask)
            aligned_mask[:, 1:] = mask[:, :-1]
            features = torch.from_numpy(data["intermediate_outputs"].astype(np.float32)).to(device, dtype=torch.bfloat16)
            features = features.reshape(*ids.shape, -1)
            final = torch.from_numpy(data["target_hidden_states"].astype(np.float32)).to(device, dtype=torch.bfloat16)
            if not np.array_equal(data["target_ids"][:, :-1], data["input_ids"][:, 1:]):
                raise ValueError("Native next-token labels differ from SpecForge same-position labels")
        if norm_weight is not None:
            normalized = final.float() * torch.rsqrt(final.float().square().mean(-1, keepdim=True) + contract["norm_epsilon"])
            scale = norm_weight.float() + (1 if contract["norm_zero_centered"] else 0)
            final = (normalized * scale).to(torch.bfloat16)
        if not torch.isfinite(features).all() or not torch.isfinite(final).all():
            raise ValueError("Nonfinite teacher features")
        if ids.shape[1] < 9 or aligned_mask.sum() == 0:
            raise ValueError("A cache conversation has no usable block supervision")
        return dict(input_ids=ids, hidden_states=features, loss_mask=aligned_mask,
                    target_last_hidden_states=final)
    train_records = [row for row in manifest["records"] if row["split"] == "train"]
    dev_records = [row for row in manifest["records"] if row["split"] == "development"]
    def conversation_prefix_hash(record):
        with np.load(cache / record["file"], allow_pickle=False) as data:
            return hashlib.sha256(data["input_ids"].tobytes()).hexdigest()
    train_hashes = {conversation_prefix_hash(record) for record in train_records}
    dev_hashes = {conversation_prefix_hash(record) for record in dev_records}
    if train_hashes & dev_hashes:
        raise ValueError("A selected conversation prefix occurs in both native splits")
    wrapper = OnlineDFlashModel(model, head, embedding, config.dflash_config["mask_token_id"],
                               block_size=8, attention_backend="sdpa", num_anchors=8,
                               objective_chunk_blocks=8, loss_type="dflash", loss_decay_gamma=0.8,
                               selector_loss_alpha=candidate["selector_loss_weight"], metric_top_k=16,
                               teacher_metrics=True).to(device)
    fixed = batch(dev_records[0])
    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        anchors, keep, hidden = wrapper._forward_draft_blocks(input_ids=fixed["input_ids"],
                        hidden_states=fixed["hidden_states"], loss_mask=fixed["loss_mask"])
        hidden = hidden.reshape(1, anchors.shape[1], 8, config.hidden_size)
        indices = (anchors.unsqueeze(-1) + torch.arange(8, device=device)).clamp(max=fixed["input_ids"].shape[1]-1)
        labels = fixed["input_ids"].unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, indices)
        predecessors = torch.cat((labels[:, :, :1], labels[:, :, :-1]), -1)
        unary, ids = model.transform_unary_logits(head(hidden)).topk(16, dim=-1)
        old_scores = model.candidate_selector.score_candidates(candidate_ids=ids, unary_logits=unary,
                                    hidden_states=hidden, predecessor_ids=predecessors).float()
        old_probs = old_scores.softmax(-1)
    rank = candidate["selector_rank"]
    growth = {"rank_before": 256, "rank_after": rank, "logit_scale": "No rank normalization in pinned native or stock selector", "tolerance_logits": 0.03125, "tolerance_probability": 0.005}
    if rank != 256:
        selector = model.candidate_selector
        def extend(value, axis, zero):
            shape = list(value.shape)
            shape[axis] = rank - 256
            extra = torch.zeros(shape, device=device, dtype=value.dtype) if zero else torch.randn(shape, device=device, dtype=value.dtype) * 0.02
            return nn.Parameter(torch.cat((value.detach(), extra), dim=axis))
        selector.predecessor_codebook = extend(selector.predecessor_codebook, 1, False)
        selector.successor_codebook = extend(selector.successor_codebook, 1, True)
        selector.hidden_projection.weight = extend(selector.hidden_projection.weight, 0, False)
        selector.hidden_projection.out_features = rank
        config.dflash_config["selector_rank"] = rank
    with torch.no_grad():
        scores = model.candidate_selector.score_candidates(candidate_ids=ids, unary_logits=unary,
                                hidden_states=hidden, predecessor_ids=predecessors).float()
        growth.update(logit_max_error=(scores-old_scores).abs().max().item(),
                      probability_max_error=(scores.softmax(-1)-old_probs).abs().max().item())
    if growth["logit_max_error"] > growth["tolerance_logits"] or growth["probability_max_error"] > growth["tolerance_probability"]:
        raise ValueError("Rank growth failed initial probability parity")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    released_parameter_count = parameter_count - (rank - 256) * (2 * config.vocab_size + config.hidden_size)
    growth["parameters"] = parameter_count
    growth["released_parameters"] = released_parameter_count
    growth["additional_parameters"] = parameter_count - released_parameter_count
    growth["parameter_increase_fraction"] = growth["additional_parameters"] / released_parameter_count
    if growth["parameter_increase_fraction"] > 0.05:
        raise ValueError("The architecture exceeds the five-percent parameter increase bound")
    (output / "initialization.json").write_text(json.dumps(growth, indent=2))
    def evaluate_records():
        model.eval()
        reports = []
        with torch.no_grad():
            for record in dev_records:
                torch.manual_seed(42)
                loss, acc, metrics = wrapper(**batch(record), collect_detailed_metrics=True)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite development loss")
                reports.append({"file": record["file"], "loss": loss.item(), "accuracy": acc.item(), **serial(metrics)})
        return reports
    before = evaluate_records()
    (output / "diagnostics-before.json").write_text(json.dumps(before, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    model.train()
    updates, supervised = 0, 0
    first_update = None
    records = []
    gradients = {}
    torch.manual_seed(42)
    for step in range(args.steps):
        if first_update is not None and time.perf_counter() - first_update >= args.seconds:
            break
        update_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, acc, metrics = wrapper(**batch(train_records[step % len(train_records)]), collect_detailed_metrics=True)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError(f"Nonfinite gradient {name}")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if rank > 256:
            selector = model.candidate_selector
            gradients = {"predecessor_extra": selector.predecessor_codebook.grad[:, 256:].float().norm().item(),
                         "successor_extra": selector.successor_codebook.grad[:, 256:].float().norm().item(),
                         "hidden_extra": selector.hidden_projection.weight.grad[256:].float().norm().item()}
        probe = model.fc.weight[:32, :32].detach().clone()
        optimizer.step()
        torch.cuda.synchronize()
        changed = not torch.equal(model.fc.weight[:32, :32].detach(), probe)
        updates += 1
        supervised += int(metrics["accuracy_denom"].item())
        ended = time.perf_counter()
        if first_update is None:
            first_update = update_start
        row = {"step": updates, "loss": loss.item(), "accuracy": acc.item(), "gradient_norm": norm.item(),
               "fc_probe_changed": changed, "update_seconds": ended-update_start,
               "measured_seconds": ended-first_update, "supervised_tokens": int(metrics["accuracy_denom"].item()),
               "rank_growth_gradients": gradients, "metrics": serial(metrics)}
        records.append(row)
        with (output / "updates.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        readiness_path = output / "readiness.json"
        temporary_readiness = output / "readiness.tmp"
        temporary_readiness.write_text(json.dumps({"updates": updates, "supervised_tokens": supervised,
                                                  "first_update_seconds": records[0]["update_seconds"]}))
        temporary_readiness.replace(readiness_path)
        print(json.dumps({"event": "optimizer_update", **row}), flush=True)
    if not any(record["fc_probe_changed"] for record in records):
        raise ValueError("No released backbone parameter changed in the fixed update probe")
    if updates < 2 or supervised <= 0:
        raise ValueError("The bounded trial did not reach two genuine updates")
    if rank > 256 and not all(value > 0 for value in gradients.values()):
        raise ValueError("Rank-growth factors did not receive nonzero gradients")
    after = evaluate_records()
    (output / "diagnostics-after.json").write_text(json.dumps(after, indent=2))
    export_start = time.perf_counter()
    checkpoint = output / "checkpoint"
    checkpoint.mkdir()
    model.cpu()
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, str(checkpoint / "model.safetensors"))
    config.save_pretrained(checkpoint)
    # Reload every saved tensor through the native HF model contract before handoff.
    reloaded = DFlash2DraftModel(config).to(dtype=torch.bfloat16)
    reloaded.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    timings = {"startup_seconds": first_update-started, "measured_update_seconds": records[-1]["measured_seconds"],
               "export_seconds": time.perf_counter()-export_start, "total_seconds": time.perf_counter()-started,
               "step_cap": args.steps, "seconds_cap": args.seconds}
    result = {"checkpoint": str(checkpoint), "updates": updates, "supervised_tokens": supervised,
              "diagnostics": {"before": str(output / "diagnostics-before.json"), "after": str(output / "diagnostics-after.json"), "initialization": growth},
              "artifacts": [str(path) for path in output.iterdir()], "timings": timings,
              "data": {"manifest": str(cache / "manifest.json"), "manifest_sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest(),
                       "selected_conversation_prefixes_disjoint": True},
              "recipe": {"implementation": "SpecForge", "commit": SPEC_FORGE_COMMIT,
                         "objective": "native hard-token DFlash CE plus strict natural-topK selector CE",
                         "selector_parameterization": "free released full-vocabulary codebooks",
                         "warm_start": str(release), "optimizer": "AdamW", "seed": 42,
                         "anchors": 8, "block_size": 8, "target_model": "Qwen/Qwen3.8-27B", "teacher_precision": "FP8"}}
    (output / "timings.json").write_text(json.dumps(timings, indent=2))
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
