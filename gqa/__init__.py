"""Grouped-Query Attention with an explicit KV cache for autoregressive decoding.

Modules
-------
rope       Rotary position embeddings applied at absolute positions.
kv_cache   Pre-allocated per-layer K/V storage (plus a sliding-window variant).
attention  GroupedQueryAttention: MHA when n_kv_heads == n_heads, MQA when n_kv_heads == 1.
generate   A tiny decoder and a generate() loop with and without the cache.
"""
from __future__ import annotations

from gqa.attention import GroupedQueryAttention
from gqa.generate import DecoderBlock, TinyDecoder, generate
from gqa.kv_cache import KVCache, SlidingWindowKVCache, kv_cache_bytes
from gqa.rope import RotaryEmbedding, apply_rope

__all__ = [
    "GroupedQueryAttention",
    "DecoderBlock",
    "TinyDecoder",
    "generate",
    "KVCache",
    "SlidingWindowKVCache",
    "kv_cache_bytes",
    "RotaryEmbedding",
    "apply_rope",
]
