"""Torch-only compositional DFlash2 modules shared by training and serving."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualCorrector(nn.Module):
    """Return a zero-initialized low-rank delta, not the residual sum."""

    def __init__(self, hidden_size, rank, gated, *, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.down = nn.Linear(hidden_size, rank, bias=False, **factory)
        self.up = nn.Linear(rank, hidden_size, bias=False, **factory)
        self.gate = nn.Linear(hidden_size, 1, bias=False, **factory) if gated else None
        nn.init.zeros_(self.up.weight)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)

    def forward(self, hidden):
        value = F.silu(self.down(hidden))
        if self.gate is not None:
            value = value * torch.sigmoid(self.gate(hidden))
        return self.up(value)


class PreservedKernelProjection(nn.Module):
    """Keep inherited GEMM shapes; allow independent finer-group coefficients."""

    def __init__(self, hidden_size, taps, group_size, *, device, dtype):
        super().__init__()
        self.coarse_groups = hidden_size // 16
        self.num_groups = hidden_size // group_size
        self.taps = taps
        self.out_features = 2 * taps * self.num_groups
        factory = {"device": device, "dtype": dtype}
        self.inherited = nn.ModuleList([
            nn.Linear(hidden_size, 4 * self.coarse_groups, bias=False, **factory)
            for _ in range(16 // group_size)
        ])
        self.extension = (
            nn.Linear(hidden_size, 2 * (taps - 2) * self.num_groups, bias=False, **factory)
            if taps > 2 else None
        )

    def forward(self, hidden):
        shape = (*hidden.shape[:-1], 2, 2, self.coarse_groups)
        coarse = [projection(hidden).view(shape) for projection in self.inherited]
        inherited = (
            coarse[0] if len(coarse) == 1
            else torch.stack(coarse, dim=-1).flatten(-2)
        )
        if self.extension is not None:
            extra = self.extension(hidden).view(*hidden.shape[:-1], 2, self.taps - 2, self.num_groups)
            inherited = torch.cat((inherited, extra), dim=-2)
        return inherited.reshape(*hidden.shape[:-1], self.out_features)


def _attach_sparsity(conv, taps):
    """Keep excluded physical taps zero without persistent checkpoint buffers."""
    active = torch.zeros(conv.taps, device=conv.base_kernel.device, dtype=conv.base_kernel.dtype)
    active[taps] = 1
    conv.research_taps = tuple(taps)
    conv.register_buffer("_research_base_mask", active.view(1, -1, 1), persistent=False)
    rows = active[2:].view(1, -1, 1).expand(2, conv.taps - 2, conv.num_groups).reshape(-1, 1)
    conv.register_buffer("_research_projection_mask", rows, persistent=False)
    if conv.base_kernel.requires_grad:
        conv.base_kernel.register_hook(lambda grad: grad * conv._research_base_mask)
    extension = conv.kernel_projection.extension
    if extension is not None and extension.weight.requires_grad:
        extension.weight.register_hook(lambda grad: grad * conv._research_projection_mask)


@torch.no_grad()
def resize_convolution(old_conv, *, taps: list[int], group_size: int):
    """Preserve each released coefficient and allocate zero new coefficients."""
    if taps != sorted(set(taps)) or not {0, 1}.issubset(taps) or not all(type(tap) is int and 0 <= tap < old_conv.block_size for tap in taps):
        raise ValueError("Convolution taps must be sorted, unique, in the block, and include 0 and 1")
    if old_conv.taps != 2 or old_conv.group_size != 16 or group_size not in (8, 16):
        raise ValueError("Resize requires the pinned two-tap, group-sixteen released convolution")
    hidden_size = old_conv.base_kernel.shape[-1]
    physical_taps = max(taps) + 1
    device, dtype = old_conv.base_kernel.device, old_conv.base_kernel.dtype
    # Build only metadata for the stock wrapper, not a discarded wide matrix.
    with torch.device("meta"):
        conv = type(old_conv)(hidden_size=hidden_size, block_size=old_conv.block_size,
                              taps=physical_taps, group_size=group_size)
    conv.base_kernel = nn.Parameter(
        torch.zeros(2, physical_taps, hidden_size, device=device, dtype=dtype),
        requires_grad=old_conv.base_kernel.requires_grad,
    )
    conv.base_kernel[:, :2].copy_(old_conv.base_kernel)
    conv.kernel_projection = PreservedKernelProjection(
        hidden_size, physical_taps, group_size, device=device, dtype=dtype,
    )
    for inherited in conv.kernel_projection.inherited:
        inherited.weight.copy_(old_conv.kernel_projection.weight)
        inherited.weight.requires_grad_(old_conv.kernel_projection.weight.requires_grad)
    if conv.kernel_projection.extension is not None:
        nn.init.zeros_(conv.kernel_projection.extension.weight)
        conv.kernel_projection.extension.weight.requires_grad_(old_conv.kernel_projection.weight.requires_grad)
    conv.train(old_conv.training)
    _attach_sparsity(conv, taps)
    assert_convolution_sparsity(conv)
    return conv


@torch.no_grad()
def assert_convolution_sparsity(conv):
    """Reject nonzero excluded tap tensors in either native backend."""
    if not hasattr(conv, "research_taps"):
        return
    excluded = [tap for tap in range(conv.taps) if tap not in conv.research_taps]
    if not excluded:
        return
    extension = conv.kernel_projection.extension
    projection = extension.weight.view(2, conv.taps - 2, conv.num_groups, -1)
    excluded_extension = [tap - 2 for tap in excluded]
    if torch.count_nonzero(conv.base_kernel[:, excluded]).item() or torch.count_nonzero(projection[:, excluded_extension]).item():
        raise ValueError("An excluded sparse convolution tap is nonzero")
    if conv.base_kernel.grad is not None and torch.count_nonzero(conv.base_kernel.grad[:, excluded]).item():
        raise ValueError("An excluded sparse base tap has a nonzero gradient")
    if extension.weight.grad is not None:
        grad = extension.weight.grad.view_as(projection)
        if torch.count_nonzero(grad[:, excluded_extension]).item():
            raise ValueError("An excluded sparse projection tap has a nonzero gradient")


def architecture_parameter_growth(model, plan):
    """Count physically allocated parameters before any model mutation."""
    hidden_size = int(model.config.hidden_size)
    convolutions = []
    for entry in plan["convolutions"]:
        name = f'layers.{entry["layer"]}.{entry["sublayer"]}_conv'
        old_conv = getattr(model.layers[entry["layer"]], f'{entry["sublayer"]}_conv')
        before = sum(parameter.numel() for parameter in old_conv.parameters())
        physical_taps = max(entry["taps"]) + 1
        after = 2 * physical_taps * hidden_size + 2 * physical_taps * (hidden_size // entry["group_size"]) * hidden_size
        convolutions.append({"name": name, "parameters_before": before, "parameters_after": after,
                             "additional_parameters": after - before, "physical_taps": physical_taps,
                             "active_taps": entry["taps"], "group_size": entry["group_size"]})
    correctors = [{"name": f'layers.{entry["layer"]}.research_corrector', "rank": entry["rank"],
                   "gated": entry["gated"], "additional_parameters": 2 * hidden_size * entry["rank"] + (hidden_size if entry["gated"] else 0)}
                  for entry in plan["correctors"]]
    return {"convolutions": convolutions, "correctors": correctors,
            "additional_parameters": sum(entry["additional_parameters"] for entry in convolutions + correctors)}
