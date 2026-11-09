# Grouped-Query Attention with a KV cache

A small, dependency-light PyTorch implementation of Grouped-Query Attention (GQA) with
an explicit, pre-allocated Key/Value cache for autoregressive decoding. Multi-Head
Attention (MHA) and Multi-Query Attention (MQA) are not separate code paths: they are
the two ends of the `n_kv_heads` dial.

```
gqa/
  rope.py        RotaryEmbedding: cos/sin tables, applied at absolute positions
  kv_cache.py    KVCache (pre-allocated) + SlidingWindowKVCache (ring buffer)
  attention.py   GroupedQueryAttention + position-based mask builder
  generate.py    RMSNorm / MLP / DecoderBlock / TinyDecoder + generate(use_cache=...)
tests/gqa/       51 CPU tests, ~2 s
```

```bash
python3 -m pytest tests/gqa -q
```

The rest of this document is the walkthrough I would give in a 45-minute interview.

---

## 1. MHA → GQA → MQA in one parameter

Attention needs, per layer, `n_heads` query heads and *some number* of key/value heads.

| Variant | `n_kv_heads` | Query heads per KV head | KV cache size (relative) | Used by |
|---|---|---|---|---|
| MHA | `= n_heads` | 1 | 1× | GPT-2/3, Llama-1, Llama-2 ≤13B |
| GQA | `1 < g < n_heads` | `n_heads / g` | `g / n_heads` | Llama-2-70B, Llama-3 (8 KV heads), Mistral, Gemma-2 |
| MQA | `1` | `n_heads` | `1 / n_heads` | PaLM, Falcon, StarCoder |

Why it exists: at decode time the KV cache, not the weights, dominates memory traffic.
Each generated token re-reads every cached K and V. MQA cuts that by `n_heads`× but
measurably hurts quality and is unstable to train; GQA with 8 groups recovers almost
all of MHA's quality at a fraction of the cache. Ainslie et al. (2023) also showed you
can *uptrain* an MHA checkpoint into GQA by mean-pooling the KV heads of each group.

In code (`attention.py`), the only places `n_kv_heads` appears are:

```python
self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim)   # smaller projection
self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim)
...
k = repeat_kv(k, self.n_rep)   # n_rep = n_heads // n_kv_heads, right before QK^T
v = repeat_kv(v, self.n_rep)
```

`repeat_kv` is `repeat_interleave(n_rep, dim=1)`, so **query head `h` reads KV head
`h // n_rep`** (contiguous groups). The tests pin this down against a hand-written
per-head loop, and the `n_kv_heads == n_heads` case is checked bit-for-bit (1e-5 in
fp64) against `torch.nn.MultiheadAttention` with copied weights.

## 2. The KV cache memory formula

```
bytes = 2 · n_layers · batch · n_kv_heads · seq_len · head_dim · bytes_per_elem
        ↑ K and V
```

Worked example, Llama-3-8B-ish: 32 layers, 8 KV heads, head_dim 128, 4 096 context,
bf16 (2 bytes), batch 1:

```
2 · 32 · 1 · 8 · 4096 · 128 · 2  =  536 870 912 B  =  512 MiB per sequence
```

The same shape with MHA (32 query heads → 32 KV heads) would be 2 GiB; with 64 heads
(70B-class) 4 GiB. Note what the formula does *not* contain: `n_heads` or `d_model`.
The cache size is governed by the KV side only, which is why GQA is a decode-time
optimisation and MQA/GQA have no effect on prefill FLOPs.

Scaling intuition to say out loud:

* Linear in batch and context. A batch of 32 at 8k context on the 8-KV-head model is
  32 GiB of cache: that is the number that decides how many concurrent users fit on a GPU.
* Pre-allocating at `max_seq_len` (as `KVCache` does) trades waste for zero allocations
  per step. Production servers use paged allocation (vLLM) to reclaim the waste.
* bf16 → fp8/int8 KV quantisation halves it again; that is a separate knob.

`kv_cache.py` exposes exactly this formula as `kv_cache_bytes(...)`, and the test
suite asserts the 512 MiB number.

## 3. Why expand KV heads at attention time, not in the cache

The cache stores tensors of shape `(batch, n_kv_heads, max_seq_len, head_dim)`. We could
have stored `n_heads` copies and simplified the attention code, but:

1. **Memory.** The entire point of GQA is the `n_heads / n_kv_heads` saving. Storing
   expanded heads gives the memory footprint of MHA with the quality of GQA, the worst
   of both.
2. **Bandwidth.** Decode is memory-bound. What matters is bytes *read* per step. Reading
   `n_kv_heads` heads and expanding in registers/SRAM is cheap; reading `n_heads` from
   HBM is `n_rep`× more traffic.
3. **Locality of the trick.** `repeat_kv` is a one-liner that lives exactly where the
   math needs it. Everything upstream (projection, RoPE, cache) only knows about
   `n_kv_heads`. In a fused kernel (FlashAttention supports GQA natively) the expansion
   is just index arithmetic `kv_head = q_head // n_rep` and never materialises at all.

`repeat_interleave` here *does* materialise a copy, which is fine for a reference
implementation. The zero-copy alternative is `k[:, :, None].expand(B, n_kv, n_rep, S, D).reshape(B, n_heads, S, D)`,
which is still a copy after `reshape` in eager PyTorch; only a fused kernel avoids it.

## 4. RoPE and the cache: rotate once, at the absolute position

RoPE rotates each 2-D channel pair of q and k by `pos · θ_i`. Because rotation matrices
compose, `⟨R(p)q, R(p')k⟩ = ⟨q, R(p'−p)k⟩`: the score depends only on the relative
distance. Two consequences for caching:

* **Keys are rotated once and stored rotated.** A key produced at position 7 is rotated
  by 7 and cached. Later queries at positions 8, 9, … are rotated by their own
  positions, and the dot product automatically "sees" distances 1, 2, …. Nothing in the
  cache is ever re-rotated.
* **The one thing that must be right is `positions = start_pos + arange(T)`.** If a
  decode step rotated its single token at position 0 (a classic bug when the attention
  module is reused from training code), relative distances to every cached key would be
  wrong and outputs would silently degrade. `test_cached_decode_equals_full_sequence_every_step`
  would catch it immediately.

The tests check the invariance directly (shifting both positions leaves the score
unchanged) and end-to-end: a left-padded sequence sitting at positions 2..5 produces the
same outputs as the unpadded sequence at 0..3, because only relative positions matter.

`RotaryEmbedding` precomputes `cos`/`sin` tables up to `max_seq_len` as non-persistent
buffers (they move with `.to(device)` but are not saved in checkpoints) and indexes them
with the position vector. For context extension you would change the table (NTK-aware
scaling, YaRN), not the attention code.

## 5. Prefill vs decode

Per layer, with sequence length `S`, model width `d`, head dim `hd`:

| Phase | What runs | Attention FLOPs | Bound by |
|---|---|---|---|
| Prefill (prompt of `T` tokens, one forward) | `T×T` scores | `O(T² · d)` | compute (big matmuls) |
| Decode step with cache | `1×S` scores | `O(S · d)` | memory bandwidth (read `S` cached K,V) |
| Decode step **without** cache | recompute `S×S` | `O(S² · d)` | compute, and redundant |

Over a generation of `N` tokens the cached decoder does `O(N · S)` attention work; the
naive one does `O(N · S²)`. `generate(use_cache=False)` in `generate.py` is that naive
loop, kept purely as the oracle for the equivalence test.

Decode is memory-bound because each step is a matrix-*vector* product: every weight and
every cached byte is read once to produce one token. That is why batching helps (reads
are amortised), why GQA helps (fewer cache bytes), and why speculative decoding helps
(several tokens per weight read).

Where a **sliding window** fits: `SlidingWindowKVCache` is a ring buffer of `window`
slots, so cache memory and per-step bandwidth become `O(window)` instead of `O(S)`.
Because the attention mask is built from `key_positions()` rather than from slot order,
the attention code did not change at all to support it; the test checks it against a
banded mask over the full sequence. Mistral uses this with `window = 4096`.

## 6. Causal masking when `start_pos > 0`

A decode chunk of `T` queries at positions `start_pos .. start_pos+T-1` attends to `S`
cached keys at positions `0 .. S-1`. The naive `torch.tril(T, S)` is **wrong** here; the
correct mask is the shifted triangle

```
allowed[i, j] = key_pos[j] <= start_pos + i
```

`build_attention_mask` builds exactly that from the two position vectors, which has
three nice properties:

* It is correct for prefill (`start_pos = 0`), single-token decode (the mask is all-True
  because every cached key is in the past), and multi-token chunks (speculative
  decoding verification).
* It composes with a user mask by `&`. Key-padding masks `(B, S)` and full `(B, T, S)`
  masks are both accepted, with `True = attend` (note: `nn.MultiheadAttention` uses the
  opposite convention; the test translates).
* A fully-masked query row (a left-padding position) would be `softmax(all −inf) = NaN`;
  the implementation zeroes those rows with `nan_to_num` so padding never poisons the batch.

Softmax runs in fp32 regardless of activation dtype, which is what you want under bf16.

## 7. What the tests prove

| Test | Claim |
|---|---|
| `test_cached_decode_equals_full_sequence_every_step` (MHA/GQA/MQA × batch 1/3) | cache + RoPE + mask are exactly equivalent to recomputation at every step |
| `test_multi_token_decode_chunks_equal_full_sequence` | chunked decode (4,3,1,2,2) is exact |
| `test_mha_special_case_matches_nn_multihead_attention` | `n_kv_heads == n_heads` is literally MHA |
| `test_matches_explicit_per_head_reference` | the `h // n_rep` grouping is what the vectorised code does |
| `test_causal_mask_with_start_pos_offset`, `test_decode_step_ignores_future_tokens` | mask correctness with offset |
| `test_key_padding_mask_left_padded_batch`, `test_padding_mask_with_cached_decode` | padding masks, incl. over the cache |
| `test_sliding_window_cache_matches_windowed_reference` | ring buffer == banded attention |
| `test_relative_position_invariance` | RoPE depends on relative offset only |
| `test_kv_cache_bytes_worked_example` | the 512 MiB number |
| `test_gradients_flow_in_training_mode`, `test_training_step_without_cache_backprops` | trainable without a cache |
| overflow tests | a clear `ValueError` naming `max_seq_len` |

Equivalence tests run in float64 so that the 1e-5 tolerance tests the algebra, not fp32
rounding. In production you would test at the deployment dtype with a looser tolerance.

## 8. Likely follow-up questions

* **How would you uptrain MHA → GQA?** Mean-pool the K and V projection weights of each
  group of `n_rep` heads into one, then fine-tune for a few percent of the original steps.
* **Why not MQA everywhere?** Quality drop and training instability; one KV head is a
  bottleneck for diverse attention patterns. 8 groups is the empirical sweet spot.
* **What breaks with batching + padding + cache?** Right-padding breaks decode (the next
  token would be appended after the pads); left-pad and carry a key-padding mask that
  grows by one `True` per step, which is what `test_padding_mask_with_cached_decode` does.
* **How does this interact with FlashAttention / SDPA?** Replace `_scaled_dot_product`
  with `F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)`
  (PyTorch ≥ 2.5); the cache, RoPE and mask code stay unchanged.
* **Paged attention?** Replace the contiguous `(B, H, S, D)` tensors with a block table
  mapping logical positions to physical blocks; `update` writes into blocks and
  `key_positions()` still returns absolute positions, so the mask logic is unchanged.
* **Speculative decoding?** Verify `k` draft tokens as one multi-token chunk (supported),
  then on rejection call `update` with a smaller `start_pos` to rewind (supported; the
  cache allows `start_pos <= seq_len`).
