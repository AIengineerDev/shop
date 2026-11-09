"""Grouped-Query Attention with optional RoPE and an explicit KV cache.

Head layout: ``n_heads`` query heads share ``n_kv_heads`` key/value heads, with
``n_rep = n_heads // n_kv_heads`` query heads per group. Query head ``h`` reads
KV head ``h // n_rep``.

* ``n_kv_heads == n_heads``  -> Multi-Head Attention (every query head has its own KV).
* ``1 < n_kv_heads < n_heads`` -> Grouped-Query Attention.
* ``n_kv_heads == 1``        -> Multi-Query Attention (one KV head for everyone).

The KV cache stores ``n_kv_heads`` heads. Expansion to ``n_heads`` happens *only*
inside :meth:`GroupedQueryAttention.forward`, right before the score computation,
via ``repeat_interleave`` on the head axis. Storing expanded heads would throw away
the memory and bandwidth savings that motivate GQA in the first place.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gqa.kv_cache import KVCache
from gqa.rope import RotaryEmbedding


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand ``(B, n_kv, S, D)`` to ``(B, n_kv * n_rep, S, D)``.

    Query head ``h`` lines up with KV head ``h // n_rep`` (interleaved repeat).
    """
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def build_attention_mask(
    query_positions: Tensor,
    key_positions: Tensor,
    attn_mask: Optional[Tensor] = None,
    *,
    is_causal: bool = True,
) -> Optional[Tensor]:
    """Return a boolean mask broadcastable to ``(B, 1, T, S)`` where ``True`` = attend.

    The causal part is built from *absolute positions*, so it is correct for any
    ``start_pos`` and for caches whose storage order is not position order.
    ``attn_mask`` may be ``(B, S)`` (key padding, ``True`` = real token) or any
    boolean tensor broadcastable to ``(B, 1, T, S)``.
    """
    mask: Optional[Tensor] = None
    if is_causal:
        mask = key_positions[None, :] <= query_positions[:, None]  # (T, S)
        mask = mask[None, None]  # (1, 1, T, S)
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            raise TypeError("attn_mask must be a boolean tensor (True = attend)")
        if attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, None, :]  # (B, 1, 1, S)
        elif attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None]  # (B, 1, T, S)
        mask = attn_mask if mask is None else (mask & attn_mask)
    return mask


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        *,
        rope: Optional[RotaryEmbedding] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        if rope is not None and rope.head_dim != self.head_dim:
            raise ValueError(f"rope.head_dim={rope.head_dim} != head_dim={self.head_dim}")
        self.rope = rope
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        *,
        cache: Optional[KVCache] = None,
        start_pos: int = 0,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Attend over ``x`` of shape ``(B, T, d_model)``.

        Prefill: ``cache`` given, ``start_pos == 0``, ``T`` = prompt length.
        Decode:  ``cache`` given, ``start_pos == cache.seq_len``, ``T`` = 1 (or a chunk).
        Training / reference: ``cache is None``; ``start_pos`` must be 0 and the
        sequence attends to itself.
        """
        bsz, seqlen, _ = x.shape
        if cache is None and start_pos != 0:
            raise ValueError("start_pos > 0 requires a cache (there is nothing to attend to otherwise)")

        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q_pos = torch.arange(start_pos, start_pos + seqlen, device=x.device)
        if self.rope is not None:
            q = self.rope(q, q_pos)
            k = self.rope(k, q_pos)  # keys are rotated once, at their absolute position

        if cache is not None:
            k, v = cache.update(start_pos, k, v)  # (B, n_kv, S, D) views of the valid prefix
            k_pos = cache.key_positions()
        else:
            k_pos = q_pos

        # GQA expansion happens here and only here.
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        mask = build_attention_mask(q_pos, k_pos, attn_mask, is_causal=is_causal)
        out = self._scaled_dot_product(q, k, v, mask)  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.n_heads * self.head_dim)
        return self.o_proj(out)

    def _scaled_dot_product(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, S)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        # Softmax in float32 for stability regardless of activation dtype.
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        # A fully-masked row (e.g. a padding query) is all -inf -> NaN; make it zeros.
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        return torch.matmul(probs, v)
