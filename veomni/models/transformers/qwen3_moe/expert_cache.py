from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch


@dataclass
class MoECacheContext:
    expert_cache: "MoEExpertCache"
    cache_position: torch.Tensor
    warmup_tokens: int
    topk: int
    align_topk: int
    aux_loss_weight: float
    batch_size: int
    sequence_length: int
    aux_losses: List[torch.Tensor]


_CACHE_CONTEXT: List[Optional[MoECacheContext]] = []


@contextmanager
def moe_cache_context(
    expert_cache: "MoEExpertCache",
    cache_position: torch.Tensor,
    warmup_tokens: int,
    topk: int,
    align_topk: int,
    aux_loss_weight: float,
    batch_size: int,
    sequence_length: int,
) -> Iterator[MoECacheContext]:
    context = MoECacheContext(
        expert_cache=expert_cache,
        cache_position=cache_position,
        warmup_tokens=warmup_tokens,
        topk=topk,
        align_topk=align_topk,
        aux_loss_weight=aux_loss_weight,
        batch_size=batch_size,
        sequence_length=sequence_length,
        aux_losses=[],
    )
    _CACHE_CONTEXT.append(
        context
    )
    try:
        yield context
    finally:
        _CACHE_CONTEXT.pop()


def get_moe_cache_context() -> Optional[MoECacheContext]:
    return _CACHE_CONTEXT[-1] if _CACHE_CONTEXT else None


def add_cache_aux_loss(loss: torch.Tensor) -> None:
    context = get_moe_cache_context()
    if context is None:
        return
    context.aux_losses.append(loss)


class MoEExpertCache:
    """Per-layer, per-sample expert cache with LRU updates."""

    def __init__(self, num_layers: int, batch_size: int, budget: int, device: torch.device):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.budget = budget
        self.device = device

        self._cache: List[torch.Tensor] = [
            torch.full((batch_size, budget), -1, dtype=torch.long, device=device) for _ in range(num_layers)
        ]
        self._age: List[torch.Tensor] = [
            torch.full((batch_size, budget), -1, dtype=torch.long, device=device) for _ in range(num_layers)
        ]
        self._tick: List[torch.Tensor] = [
            torch.zeros((batch_size,), dtype=torch.long, device=device) for _ in range(num_layers)
        ]
        self.ema_scale: List[torch.Tensor] = [
            torch.full((batch_size,), float("inf"), dtype=torch.float, device=device) for _ in range(num_layers)
        ]

    def reset(self) -> None:
        for layer_idx in range(self.num_layers):
            self._cache[layer_idx].fill_(-1)
            self._age[layer_idx].fill_(-1)
            self._tick[layer_idx].zero_()
            self.ema_scale[layer_idx].fill_(float("inf"))

    def get(self, layer_idx: int) -> torch.Tensor:
        return self._cache[layer_idx]

    def get_mask(self, layer_idx: int, num_experts: int) -> torch.Tensor:
        cache = self._cache[layer_idx]
        mask = torch.zeros((self.batch_size, num_experts), dtype=torch.bool, device=cache.device)
        valid = cache.ge(0)
        if valid.any():
            batch_idx, slot_idx = valid.nonzero(as_tuple=True)
            expert_ids = cache[batch_idx, slot_idx]
            mask[batch_idx, expert_ids] = True
        return mask

    def update(self, layer_idx: int, expert_ids: torch.Tensor) -> None:
        """expert_ids: [B, K]."""
        assert expert_ids.dim() == 2 and expert_ids.shape[0] == self.batch_size
        batch_indices = torch.arange(self.batch_size, device=self.device)
        self.update_tokens(layer_idx, batch_indices, expert_ids)

    def update_tokens(self, layer_idx: int, batch_indices: torch.Tensor, expert_ids: torch.Tensor) -> None:
        """Update cache for arbitrary token rows.

        batch_indices: [N]
        expert_ids: [N, K]
        """
        cache = self._cache[layer_idx]
        age = self._age[layer_idx]
        tick = self._tick[layer_idx]

        for row, batch_idx in enumerate(batch_indices.tolist()):
            for expert_id in expert_ids[row].tolist():
                if expert_id < 0:
                    continue
                current_tick = int(tick[batch_idx].item()) + 1
                tick[batch_idx] = current_tick
                cache_row = cache[batch_idx]
                age_row = age[batch_idx]
                match = (cache_row == expert_id).nonzero(as_tuple=True)[0]
                if match.numel() > 0:
                    age_row[match[0]] = current_tick
                    continue
                empty = (cache_row < 0).nonzero(as_tuple=True)[0]
                if empty.numel() > 0:
                    slot = empty[0].item()
                else:
                    slot = age_row.argmin().item()
                cache_row[slot] = expert_id
                age_row[slot] = current_tick


def cache_prior_scale(
    logits: torch.Tensor,
    cache: MoEExpertCache,
    layer_idx: int,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    peak_to_peak = logits.amax(dim=1) - logits.amin(dim=1)
    ema_old = cache.ema_scale[layer_idx][batch_indices]
    ema_new = torch.where(
        ema_old.isinf(),
        peak_to_peak.to(ema_old.dtype),
        torch.lerp(peak_to_peak.to(ema_old.dtype), ema_old, 0.95),
    )
    cache.ema_scale[layer_idx][batch_indices] = ema_new
    return ema_new.to(peak_to_peak.dtype)


def cache_prior_promote(
    logits: torch.Tensor,
    cache: MoEExpertCache,
    layer_idx: int,
    batch_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    topk_idx: torch.Tensor,
    bias_scale: float = 0.2,
) -> torch.Tensor:
    scale = bias_scale * cache_prior_scale(logits, cache, layer_idx, batch_indices)
    mask = cache_mask.clone()
    mask.scatter_(1, topk_idx, True)
    penalty = scale.view(-1, 1) * (~mask).to(logits.dtype)
    return logits - penalty


def attach_moe_layer_indices(model: torch.nn.Module) -> int:
    """Attach layer indices to MoE blocks so the router can index cache."""
    count = 0
    for layer_idx, layer in enumerate(getattr(model, "layers", [])):
        moe_block = getattr(layer, "mlp", None)
        if moe_block is None:
            continue
        if hasattr(moe_block, "gate") and hasattr(moe_block.gate, "__class__"):
            moe_block.layer_idx = layer_idx
            moe_block.gate.layer_idx = layer_idx
            count += 1
    return count
