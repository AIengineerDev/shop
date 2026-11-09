"""Rotary position embeddings (RoPE).

RoPE rotates each pair of channels ``(x[2i], x[2i+1])`` -- here in the "rotate-half"
layout, i.e. pairs ``(x[i], x[i + D/2])`` -- by an angle ``pos * theta_i`` where
``theta_i = base ** (-2i / D)``. Because the rotation is applied to *both* q and k
before the dot product, ``<R(p) q, R(p') k> = <q, R(p' - p) k>``: the score depends
only on the relative offset ``p' - p``.

Why this matters for a KV cache: we rotate keys with their *absolute* position once,
at the moment they are produced, and store the rotated key. A query produced later
at position ``t`` is rotated by ``t`` and the dot product automatically "sees" the
relative distance. Nothing in the cache ever has to be re-rotated. The one thing
that must be right is the position vector: ``positions = start_pos + arange(T)``.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` of shape ``(..., T, D)`` with ``cos``/``sin`` of shape ``(T, D/2)``.

    ``cos``/``sin`` are tiled across the two halves so that channel ``i`` and
    channel ``i + D/2`` share an angle (rotate-half convention, as in Llama).
    """
    cos = torch.cat((cos, cos), dim=-1).to(x.dtype)
    sin = torch.cat((sin, sin), dim=-1).to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for positions ``[0, max_seq_len)``.

    Tables are kept in float32 and cast to the input dtype at apply time; the
    attention module calls :meth:`forward` with explicit absolute positions.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        cos, sin = self._build_tables(head_dim, max_seq_len, base)
        # Buffers (not parameters): they move with .to(device) but are not trained.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _build_tables(head_dim: int, max_seq_len: int, base: float) -> Tuple[Tensor, Tensor]:
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )  # (D/2,)
        positions = torch.arange(max_seq_len, dtype=torch.float32)  # (S,)
        angles = torch.outer(positions, inv_freq)  # (S, D/2)
        return angles.cos(), angles.sin()

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Rotate ``x`` of shape ``(B, H, T, D)`` at absolute ``positions`` of shape ``(T,)``."""
        if positions.ndim != 1 or positions.shape[0] != x.shape[-2]:
            raise ValueError(
                f"positions must be 1-D of length T={x.shape[-2]}, got shape {tuple(positions.shape)}"
            )
        if int(positions.max()) >= self.max_seq_len:
            raise ValueError(
                f"RoPE table overflow: position {int(positions.max())} >= max_seq_len={self.max_seq_len}"
            )
        return apply_rope(x, self.cos[positions], self.sin[positions])
