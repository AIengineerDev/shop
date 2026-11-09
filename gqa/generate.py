"""A tiny pre-norm decoder and a ``generate`` loop with and without the KV cache.

``generate(use_cache=True)`` prefills the prompt once, then feeds one token per step
with ``start_pos = current_length``. ``generate(use_cache=False)`` re-runs the whole
sequence every step. Both must produce identical tokens and logits; the tests check
that at every step.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from gqa.attention import GroupedQueryAttention
from gqa.kv_cache import KVCache
from gqa.rope import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden: int, bias: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        *,
        rope: Optional[RotaryEmbedding] = None,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, rope=rope, dropout=dropout, bias=bias
        )
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio * d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        *,
        cache: Optional[KVCache] = None,
        start_pos: int = 0,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cache=cache, start_pos=start_pos, attn_mask=attn_mask)
        return x + self.mlp(self.mlp_norm(x))


class TinyDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        *,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)  # shared by all layers
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, n_heads, n_kv_heads, rope=self.rope, dropout=dropout, bias=bias)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def make_caches(
        self,
        batch: int,
        max_seq_len: Optional[int] = None,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> List[KVCache]:
        """One :class:`KVCache` per layer. Dtype/device default to the model's."""
        p = next(self.parameters())
        return [
            KVCache(
                batch,
                self.n_kv_heads,
                max_seq_len or self.max_seq_len,
                self.head_dim,
                dtype=dtype or p.dtype,
                device=device or p.device,
            )
            for _ in self.blocks
        ]

    def forward(
        self,
        ids: Tensor,
        *,
        caches: Optional[List[KVCache]] = None,
        start_pos: int = 0,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return logits of shape ``(B, T, vocab)`` for token ids ``(B, T)``."""
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} caches, got {len(caches)}")
        x = self.embed(ids)
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x = block(x, cache=cache, start_pos=start_pos, attn_mask=attn_mask)
        return self.lm_head(self.norm(x))


def _pick_next(logits: Tensor, greedy: bool) -> Tensor:
    if greedy:
        return logits.argmax(dim=-1)
    return torch.multinomial(torch.softmax(logits.float(), dim=-1), num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model: TinyDecoder,
    prompt_ids: Tensor,
    max_new_tokens: int,
    *,
    use_cache: bool = True,
    greedy: bool = True,
    return_logits: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Generate ``max_new_tokens`` tokens after ``prompt_ids`` of shape ``(B, P)``.

    Returns the full sequence ``(B, P + max_new_tokens)``; with ``return_logits``
    also the next-token logits used at each step, shape ``(B, max_new_tokens, vocab)``.

    ``use_cache=True``: prefill the prompt, then one forward of ``T=1`` per step at
    ``start_pos = len so far`` -- O(S) work per step.
    ``use_cache=False``: re-run the model on the whole sequence every step and read
    the last position -- O(S^2) per step, used as the reference in tests.
    """
    was_training = model.training
    model.eval()
    bsz, prompt_len = prompt_ids.shape
    total = prompt_len + max_new_tokens
    if total > model.max_seq_len:
        raise ValueError(
            f"KV cache overflow: prompt_len={prompt_len} + max_new_tokens={max_new_tokens} = {total} "
            f"> max_seq_len={model.max_seq_len}"
        )

    ids = prompt_ids
    step_logits: List[Tensor] = []
    caches = model.make_caches(bsz) if use_cache else None

    # Prefill (cached path) -- the only time T > 1 goes through the model.
    if use_cache:
        logits = model(ids, caches=caches, start_pos=0)[:, -1]
    else:
        logits = model(ids)[:, -1]

    for _ in range(max_new_tokens):
        step_logits.append(logits)
        next_id = _pick_next(logits, greedy)
        ids = torch.cat([ids, next_id[:, None]], dim=1)
        if ids.shape[1] == total:
            break
        if use_cache:
            logits = model(next_id[:, None], caches=caches, start_pos=ids.shape[1] - 1)[:, -1]
        else:
            logits = model(ids)[:, -1]

    if was_training:
        model.train()
    if return_logits:
        return ids, torch.stack(step_logits, dim=1)
    return ids
