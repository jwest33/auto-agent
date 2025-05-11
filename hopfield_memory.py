from __future__ import annotations

import math
import torch
from torch import nn  # still inherit from nn.Module


class HopfieldMemory(nn.Module):
    """A fixed-capacity dot-product associative memory.
    """

    def __init__(self, key_dim: int = 64, value_dim: int = 1, capacity: int = 1024):
        super().__init__()
        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        # non-trainable memory banks (buffers → saved on `state_dict`)
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, value_dim))
        self.item_count: int = 0
        self.scale = math.sqrt(float(key_dim))

    # ───────────────────────── Write ───────────────────────── #
    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor):
        assert key.shape[-1] == self.key_dim, "Key dimension mismatch"
        assert value.shape[-1] == self.value_dim, "Value dimension mismatch"
        idx = self.item_count % self.capacity
        self.keys[idx] = key
        self.values[idx] = value
        self.item_count += 1

    # ───────────────────────── Read ────────────────────────── #
    def forward(self, query: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Associatively retrieve a value for query (shape: (key_dim,))."""
        if self.item_count == 0:
            return torch.zeros(self.value_dim, device=query.device)
        active_k = self.keys[: self.item_count]  # (N, d)
        sims = torch.mv(active_k, query) / self.scale  # (N,)
        weights = torch.softmax(sims, dim=0)  # (N,)
        return (weights.unsqueeze(-1) * self.values[: self.item_count]).sum(0)
