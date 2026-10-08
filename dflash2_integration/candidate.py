"""Literal, compositional DFlash2 architecture contract; no tensor allocations."""
from __future__ import annotations

import hashlib
import json
import math

HIDDEN_SIZE = 5120
VOCAB_SIZE = 248320
RELEASED_PARAMETERS = 1924404480
PARAMETER_GROWTH_LIMIT = 0.05
FIELDS = {"kind", "hypothesis", "predicted_metric", "selector_rank", "learning_rate",
          "selector_loss_weight", "seed", "architecture"}


def validate_architecture(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"convolutions", "correctors"}:
        raise ValueError("Architecture must contain exactly convolutions and correctors")
    convolutions, correctors = value["convolutions"], value["correctors"]
    if not isinstance(convolutions, list) or len(convolutions) > 10:
        raise ValueError("Convolutions must be a list of at most ten layer/sublayer placements")
    if not isinstance(correctors, list) or len(correctors) > 5:
        raise ValueError("Correctors must be a list of at most five layer placements")
    conv_slots, corrector_slots = set(), set()
    for record in convolutions:
        if not isinstance(record, dict) or set(record) != {"layer", "sublayer", "taps", "group_size"}:
            raise ValueError("A convolution needs layer, sublayer, taps, and group_size")
        layer, sublayer, taps, group = record["layer"], record["sublayer"], record["taps"], record["group_size"]
        if type(layer) is not int or not 0 <= layer < 5 or sublayer not in ("attention", "mlp"):
            raise ValueError("Convolutions target attention or mlp in layers zero through four")
        if (not isinstance(taps, list) or not 2 <= len(taps) <= 8
                or any(type(tap) is not int or not 0 <= tap < 8 for tap in taps)
                or taps != sorted(set(taps)) or taps[:2] != [0, 1]):
            raise ValueError("Taps must be sorted unique positions in zero through seven, including zero and one")
        if type(group) is not int or group not in (8, 16):
            raise ValueError("Convolution group_size must be eight or sixteen")
        slot = (layer, sublayer)
        if slot in conv_slots:
            raise ValueError("A layer/sublayer convolution cannot be declared twice")
        if taps == [0, 1] and group == 16:
            raise ValueError("An unchanged convolution is not an architecture mutation")
        conv_slots.add(slot)
    for record in correctors:
        if not isinstance(record, dict) or set(record) != {"layer", "rank", "gated"}:
            raise ValueError("A corrector needs layer, rank, and gated")
        layer, rank, gated = record["layer"], record["rank"], record["gated"]
        if type(layer) is not int or not 0 <= layer < 5:
            raise ValueError("Correctors target layers zero through four")
        if type(rank) is not int or rank not in (32, 64, 128, 256) or type(gated) is not bool:
            raise ValueError("Corrector rank must be 32/64/128/256 and gated must be a boolean")
        if layer in corrector_slots:
            raise ValueError("A layer corrector cannot be declared twice")
        corrector_slots.add(layer)
    return {
        "convolutions": sorted([dict(record, taps=list(record["taps"])) for record in convolutions],
                               key=lambda record: (record["layer"], record["sublayer"])),
        "correctors": sorted([dict(record) for record in correctors], key=lambda record: record["layer"]),
    }


def estimated_parameters(candidate: dict) -> int:
    total = RELEASED_PARAMETERS + (candidate["selector_rank"] - 256) * (2 * VOCAB_SIZE + HIDDEN_SIZE)
    old_conv = 4 * HIDDEN_SIZE + HIDDEN_SIZE * 4 * (HIDDEN_SIZE // 16)
    for record in candidate["architecture"]["convolutions"]:
        taps = max(record["taps"]) + 1
        new_conv = 2 * taps * HIDDEN_SIZE + HIDDEN_SIZE * 2 * taps * (HIDDEN_SIZE // record["group_size"])
        total += new_conv - old_conv
    for record in candidate["architecture"]["correctors"]:
        total += 2 * HIDDEN_SIZE * record["rank"] + (HIDDEN_SIZE if record["gated"] else 0)
    return total


def validate_candidate(value: object, *, allow_baseline: bool = False) -> dict:
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError(f"Candidate must contain exactly {sorted(FIELDS)}")
    if value["kind"] not in ("architecture", "recipe"):
        raise ValueError("Candidate kind must be architecture or recipe")
    for key in ("hypothesis", "predicted_metric"):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 4000:
            raise ValueError(f"{key} must be nonempty text of at most 4000 characters")
    if type(value["selector_rank"]) is not int or value["selector_rank"] not in (256, 384):
        raise ValueError("Selector rank must be 256 or 384")
    if type(value["seed"]) is not int or value["seed"] != 42:
        raise ValueError("The matched architecture screen fixes seed 42")
    for key, low, high in (("learning_rate", 1e-5, 1e-4), ("selector_loss_weight", 0.1, 1.0)):
        number = value[key]
        if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{key} is outside its fixed finite range")
    result = dict(value, architecture=validate_architecture(value["architecture"]))
    changed = bool(result["architecture"]["convolutions"] or result["architecture"]["correctors"])
    if result["kind"] == "architecture":
        if result["learning_rate"] != 3e-5 or result["selector_loss_weight"] != 0.1:
            raise ValueError("Architecture trials keep the matched training recipe")
        if not allow_baseline and not changed:
            raise ValueError("Architecture proposals require a new convolution or corrector; the rank-only screen is complete")
    elif result["selector_rank"] != 256 or changed:
        raise ValueError("Recipe trials keep the released architecture")
    if estimated_parameters(result) > RELEASED_PARAMETERS * (1 + PARAMETER_GROWTH_LIMIT):
        raise ValueError("The composed architecture exceeds five-percent actual parameter growth")
    return result


def architecture_fingerprint(candidate: dict) -> str:
    identity = {key: candidate[key] for key in ("kind", "selector_rank", "learning_rate",
                                              "selector_loss_weight", "seed", "architecture")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
