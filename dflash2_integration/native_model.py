"""Explicit SpecForge decoder extensions for trusted research checkpoints."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from specforge.modeling.draft.dflash2 import DFlash2DraftModel, Qwen3DFlash2DecoderLayer

from dflash2_integration.architecture import (
    PreservedKernelProjection, ResidualCorrector, assert_convolution_sparsity, resize_convolution,
)
from dflash2_integration.candidate import validate_architecture


class ResearchQwen3DFlash2DecoderLayer(Qwen3DFlash2DecoderLayer):
    """Apply the corrector to the complete native residual state."""

    def forward(self, *args, **kwargs):
        hidden = super().forward(*args, **kwargs)
        corrector = getattr(self, "research_corrector", None)
        return hidden if corrector is None else hidden + corrector(hidden)


def _apply_layer_plan(layer, layer_idx, hidden_size, plan):
    for entry in plan["convolutions"]:
        if entry["layer"] == layer_idx:
            name = f'{entry["sublayer"]}_conv'
            setattr(layer, name, resize_convolution(getattr(layer, name), taps=entry["taps"], group_size=entry["group_size"]))
    for entry in plan["correctors"]:
        if entry["layer"] == layer_idx:
            reference = layer.mlp_conv.base_kernel
            layer.research_corrector = ResidualCorrector(hidden_size, entry["rank"], entry["gated"],
                                                        device=reference.device, dtype=reference.dtype)
            layer.research_corrector.train(layer.training)


class ResearchDFlash2DraftModel(DFlash2DraftModel):
    """Keep native tensor names and load an explicit compositional plan."""

    decoder_layer_class = ResearchQwen3DFlash2DecoderLayer
    _no_split_modules = ["ResearchQwen3DFlash2DecoderLayer"]

    def __init__(self, config):
        plan = getattr(config, "research_architecture", None)
        if plan is not None:
            config.research_architecture = validate_architecture(plan)
        super().__init__(config)

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        projection = getattr(module, "kernel_projection", None)
        if isinstance(projection, PreservedKernelProjection):
            for parameter in projection.parameters():
                nn.init.zeros_(parameter)
            return
        super()._init_weights(module)
        if isinstance(module, ResidualCorrector):
            # HF post_init also visits each child Linear before this parent.
            nn.init.zeros_(module.up.weight)
            if module.gate is not None:
                nn.init.zeros_(module.gate.weight)

    def _build_decoder_layer(self, config, layer_idx, kernels):
        layer = super()._build_decoder_layer(config, layer_idx, kernels)
        plan = getattr(config, "research_architecture", None)
        if plan is not None:
            _apply_layer_plan(layer, layer_idx, int(config.hidden_size), plan)
        return layer

    def apply_research_architecture(self, plan):
        if getattr(self.config, "research_architecture", None) is not None:
            raise ValueError("Apply an architecture only after the strict released baseline load")
        plan = validate_architecture(plan)
        for layer_idx, layer in enumerate(self.layers):
            _apply_layer_plan(layer, layer_idx, int(self.config.hidden_size), plan)
        self.config.research_architecture = deepcopy(plan)
        self.config.research_serving_plugin = "dflash2_integration.serving_model"
        self.config.research_requires_trusted_serving_plugin = True
        self.config.architectures = ["DFlash2DraftModel"]
        self.assert_architecture_sparsity()

    def assert_architecture_sparsity(self):
        for layer in self.layers:
            assert_convolution_sparsity(layer.attention_conv)
            assert_convolution_sparsity(layer.mlp_conv)
