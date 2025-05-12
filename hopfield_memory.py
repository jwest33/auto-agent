from __future__ import annotations

import math
import torch
from torch import nn

class HopfieldMemory(nn.Module):
    """A fixed-capacity dot-product associative memory with dynamic resizing capability.
    """

    def __init__(self, key_dim: int = 64, value_dim: int = 1, capacity: int = 1024):
        super().__init__()
        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        # non-trainable memory banks
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, value_dim))
        self.item_count: int = 0
        self.scale = math.sqrt(float(key_dim))

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor):
        """Write a key-value pair to memory with dynamic resizing if needed."""
        # Check if key dimensions match, if not, resize the memory banks
        if key.shape[-1] != self.key_dim:
            new_key_dim = key.shape[-1]
            print(f"Resizing Hopfield memory from key_dim={self.key_dim} to {new_key_dim}")
            
            # Create new memory banks with the new key dimension
            new_keys = torch.zeros(self.capacity, new_key_dim, device=self.keys.device)
            
            # Copy any existing data to new banks (up to min of old/new dimensions)
            if self.item_count > 0:
                min_dim = min(self.key_dim, new_key_dim)
                new_keys[:self.item_count, :min_dim] = self.keys[:self.item_count, :min_dim]
            
            # Update memory banks and key dimension
            self.keys = new_keys
            self.register_buffer("keys", new_keys)
            self.key_dim = new_key_dim
            self.scale = math.sqrt(float(new_key_dim))
        
        # Now proceed with writing
        idx = self.item_count % self.capacity
        self.keys[idx] = key
        self.values[idx] = value
        self.item_count += 1

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Associatively retrieve a value for query, with dynamic dimension handling."""
        # Handle empty memory
        if self.item_count == 0:
            return torch.zeros(self.value_dim, device=query.device)
            
        # Handle dimension mismatch
        if query.shape[-1] != self.key_dim:
            new_key_dim = query.shape[-1]
            print(f"Query dimension {new_key_dim} doesn't match memory dimension {self.key_dim}. Returning zero vector.")
            return torch.zeros(self.value_dim, device=query.device)
        
        # Normal retrieval
        active_k = self.keys[: self.item_count]  # (N, d)
        sims = torch.mv(active_k, query) / self.scale  # (N,)
        weights = torch.softmax(sims, dim=0)  # (N,)
        return (weights.unsqueeze(-1) * self.values[: self.item_count]).sum(0)
