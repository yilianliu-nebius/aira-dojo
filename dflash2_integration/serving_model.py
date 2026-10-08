"""Trusted research topology for the pinned stock SGLang DFlash2 model.

The external model registry imports ``serving_models`` in every worker. Only
DFlash2DraftModel changes; the stock worker, verifier, selector and CUDA graph
paths remain in use. Checkpoints contain tensors and a validated plan, not code.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import (
    DFlash2DraftModel as StockDFlash2DraftModel,
    DFlashDecoderLayer,
)

from .architecture import (
    ResidualCorrector,
    assert_convolution_sparsity,
    resize_convolution,
)
from .candidate import validate_architecture


SERVING_PLUGIN = "dflash2_integration.serving_model"
EXTERNAL_MODEL_PACKAGE = "dflash2_integration.serving_models"


class ResearchDecoderLayer(DFlashDecoderLayer):
    """Keep the stock fused residual contract and add a trained delta."""

    def __init__(self, config, layer_id, **kwargs):
        super().__init__(config=config, layer_id=layer_id, **kwargs)
        plan = getattr(config, "research_architecture", None)
        self.research_corrector = None
        if plan is None:
            return
        for entry in plan["correctors"]:
            if entry["layer"] == layer_id:
                reference = self.input_layernorm.weight
                self.research_corrector = ResidualCorrector(
                    int(config.hidden_size), entry["rank"], entry["gated"],
                    device=reference.device, dtype=reference.dtype,
                )
                break

    def forward(self, positions, hidden_states, forward_batch, residual):
        hidden_states, residual = super().forward(
            positions, hidden_states, forward_batch, residual
        )
        if self.research_corrector is not None and hidden_states.numel() != 0:
            # Stock returns the MLP output and its separate residual. The native
            # layer returns their sum. Do not add the residual a second time.
            full_hidden = hidden_states + residual
            hidden_states = hidden_states + self.research_corrector(full_hidden)
        return hidden_states, residual


class DFlash2DraftModel(StockDFlash2DraftModel):
    """Stock DFlash2 with explicit per-layer modules and strict tensor loading."""

    decoder_layer_cls = ResearchDecoderLayer

    def __init__(self, config, quant_config=None, prefix: str = ""):
        plan = getattr(config, "research_architecture", None)
        if plan is not None:
            plan = validate_architecture(plan)
            if getattr(config, "research_serving_plugin", None) != SERVING_PLUGIN:
                raise ValueError(
                    "Research checkpoints must declare the trusted serving plugin "
                    f"{SERVING_PLUGIN!r}."
                )
            config.research_architecture = plan
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self._research_convolutions = []
        if plan is None:
            return
        for entry in plan["convolutions"]:
            layer = self.layers[entry["layer"]]
            attribute = entry["sublayer"] + "_conv"
            old_conv = getattr(layer, attribute)
            if old_conv is None:
                raise ValueError("Research convolution requires a stock convolution.")
            conv = resize_convolution(
                old_conv, taps=entry["taps"], group_size=entry["group_size"]
            )
            setattr(layer, attribute, conv)
            self._research_convolutions.append(conv)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Reject unknown, duplicate or missing tensors before worker capture.

        Preserve the stock QKV and MLP shard loaders. Load corrector tensors
        directly, so no stock substring mapping can discard them. Accept the
        stock ``model.`` prefix and native encoder aliases only.
        """
        parameters = dict(self.named_parameters())
        seen: dict[str, set[object]] = {}
        aliases = {
            "encoder.fc.weight": "fc.weight",
            "encoder.output_norm_enc.weight": "hidden_norm.weight",
        }
        fused = (
            ("q_proj", "qkv_proj", "q"),
            ("k_proj", "qkv_proj", "k"),
            ("v_proj", "qkv_proj", "v"),
            ("gate_proj", "gate_up_proj", 0),
            ("up_proj", "gate_up_proj", 1),
        )

        def validated_weights():
            for original_name, tensor in weights:
                name = original_name.removeprefix("model.")
                name = aliases.get(name, name)
                resolved = name
                shard = "full"
                if resolved not in parameters:
                    for source, destination, shard_id in fused:
                        token = f".{source}."
                        if token in name:
                            resolved = name.replace(token, f".{destination}.")
                            shard = shard_id
                            break
                if resolved not in parameters:
                    raise ValueError(f"Unknown DFlash2 checkpoint tensor: {original_name}")
                parts = seen.setdefault(resolved, set())
                if shard in parts or "full" in parts or (shard == "full" and parts):
                    raise ValueError(f"Duplicate DFlash2 checkpoint tensor: {original_name}")
                parts.add(shard)
                parameter = parameters[resolved]
                custom = ".research_corrector." in resolved
                convolution = ".attention_conv." in resolved or ".mlp_conv." in resolved
                if custom or convolution:
                    if tuple(tensor.shape) != tuple(parameter.shape):
                        raise ValueError(
                            f"DFlash2 tensor {original_name} has shape {tuple(tensor.shape)}; "
                            f"the topology requires {tuple(parameter.shape)}."
                        )
                if custom:
                    default_weight_loader(parameter, tensor)
                else:
                    yield name, tensor

        super().load_weights(validated_weights())
        missing = []
        for name in parameters:
            parts = seen.get(name, set())
            if "full" in parts:
                continue
            required = {"full"}
            if ".qkv_proj." in name:
                required = {"q", "k", "v"}
            elif ".gate_up_proj." in name:
                required = {0, 1}
            if parts != required:
                missing.append(name)
        if missing:
            raise ValueError(f"Missing DFlash2 checkpoint tensors or shards: {missing}")
        for conv in self._research_convolutions:
            assert_convolution_sparsity(conv)
        return set(seen)


EntryClass = DFlash2DraftModel
