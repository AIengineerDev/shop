"""Key/Value caches for autoregressive decoding.

Memory formula (per model, not per layer)::

    bytes = 2 * n_layers * batch * n_kv_heads * seq_len * head_dim * bytes_per_elem
            ^ K and V

Worked example, Llama-style: 32 layers, 8 KV heads, head_dim 128, 4096 context, bf16,
batch 1::

    2 * 32 * 1 * 8 * 4096 * 128 * 2 = 536_870_912 bytes = 512 MiB

The same model with 64 MHA heads instead of 8 GQA heads would need 4 GiB. That 8x is
the entire point of GQA: the cache stores ``n_kv_heads`` heads and the attention step
expands them to ``n_heads`` on the fly (see :mod:`gqa.attention`).

Design notes
------------
* Storage is pre-allocated once at ``max_seq_len`` so decode steps never allocate.
* ``update`` writes a slice and returns *views* of the valid prefix -- no copy.
* The cache owns the ``seq_len`` bookkeeping; the attention module asks it for the
  absolute ``key_positions()`` so the causal mask can be built purely from positions.
  That is what makes ``start_pos > 0`` and the sliding-window variant work without
  special-casing in the attention code.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def kv_cache_bytes(
    n_layers: int,
    batch: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
    bytes_per_elem: int = 2,
) -> int:
    """Total K+V cache size in bytes across all layers."""
    return 2 * n_layers * batch * n_kv_heads * seq_len * head_dim * bytes_per_elem


class KVCache:
    """Pre-allocated K/V storage for one attention layer.

    Tensors have shape ``(batch, n_kv_heads, max_seq_len, head_dim)``.
    """

    def __init__(
        self,
        batch: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        self.batch = batch
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        shape = (batch, n_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self._seq_len = 0

    # ----------------------------------------------------------------- bookkeeping
    @property
    def seq_len(self) -> int:
        """Number of valid positions currently stored."""
        return self._seq_len

    @property
    def device(self) -> torch.device:
        return self.k.device

    @property
    def dtype(self) -> torch.dtype:
        return self.k.dtype

    def reset(self) -> None:
        """Forget everything without reallocating."""
        self._seq_len = 0

    def memory_bytes(self) -> int:
        """Bytes held by this layer's K and V tensors (allocated, not just valid)."""
        return (self.k.numel() + self.v.numel()) * self.k.element_size()

    # ----------------------------------------------------------------- core API
    def _check(self, start_pos: int, k: Tensor, v: Tensor) -> int:
        if k.shape != v.shape:
            raise ValueError(f"k and v shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.ndim != 4 or k.shape[0] != self.batch or k.shape[1] != self.n_kv_heads or k.shape[3] != self.head_dim:
            raise ValueError(
                f"expected k/v of shape (batch={self.batch}, n_kv_heads={self.n_kv_heads}, T, "
                f"head_dim={self.head_dim}), got {tuple(k.shape)}"
            )
        if not 0 <= start_pos <= self._seq_len:
            raise ValueError(
                f"start_pos={start_pos} must be within [0, seq_len={self._seq_len}]; "
                "positions cannot be skipped"
            )
        return k.shape[2]

    def update(self, start_pos: int, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Write ``k``/``v`` (``(B, n_kv, T, D)``) at ``[start_pos, start_pos+T)``.

        Returns views of the full valid prefix ``[:, :, :seq_len]`` for K and V.
        Writing at ``start_pos < seq_len`` overwrites (rewind), which is how you
        would implement backtracking / speculative-decoding rejection.
        """
        t = self._check(start_pos, k, v)
        end = start_pos + t
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: start_pos={start_pos} + T={t} = {end} > max_seq_len={self.max_seq_len}"
            )
        self.k[:, :, start_pos:end] = k.to(self.k.dtype)
        self.v[:, :, start_pos:end] = v.to(self.v.dtype)
        self._seq_len = end
        return self.k[:, :, :end], self.v[:, :, :end]

    def key_positions(self) -> Tensor:
        """Absolute positions of the stored keys, aligned with the views ``update`` returns."""
        return torch.arange(self._seq_len, device=self.k.device)


class SlidingWindowKVCache(KVCache):
    """Ring-buffer cache that keeps only the last ``window`` positions.

    Storage is ``(batch, n_kv_heads, window, head_dim)``; slot ``p % window`` holds
    position ``p``. The views returned by ``update`` are in *storage* order, not
    position order, so ``key_positions()`` is essential: the attention mask is
    built from positions and does not care about slot order.

    ``max_seq_len`` is still enforced as a logical limit (RoPE tables are finite).
    Rewinding is supported only back to positions that are still in the window.
    """

    def __init__(
        self,
        batch: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        *,
        window: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        super().__init__(batch, n_kv_heads, min(window, max_seq_len), head_dim, dtype=dtype, device=device)
        self.window = min(window, max_seq_len)
        self.logical_max_seq_len = max_seq_len

    def update(self, start_pos: int, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        t = self._check(start_pos, k, v)
        end = start_pos + t
        if end > self.logical_max_seq_len:
            raise ValueError(
                f"KV cache overflow: start_pos={start_pos} + T={t} = {end} > max_seq_len={self.logical_max_seq_len}"
            )
        if self._seq_len - start_pos > self.window:
            raise ValueError(
                f"cannot rewind to start_pos={start_pos}: only the last {self.window} positions are retained"
            )
        if t >= self.window:
            # Only the last `window` tokens of this chunk survive.
            k, v, start_pos, t = k[:, :, -self.window :], v[:, :, -self.window :], end - self.window, self.window
        slots = torch.arange(start_pos, end, device=self.k.device) % self.window
        self.k[:, :, slots] = k.to(self.k.dtype)
        self.v[:, :, slots] = v.to(self.v.dtype)
        self._seq_len = end
        n = min(end, self.window)
        return self.k[:, :, :n], self.v[:, :, :n]

    def key_positions(self) -> Tensor:
        n = min(self._seq_len, self.window)
        slots = torch.arange(n, device=self.k.device)
        if self._seq_len <= self.window:
            return slots  # slot i holds position i
        # Slot s holds the most recent position congruent to s mod window.
        oldest = self._seq_len - self.window
        return oldest + (slots - oldest) % self.window
