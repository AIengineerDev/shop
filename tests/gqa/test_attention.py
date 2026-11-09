from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from gqa.attention import GroupedQueryAttention, build_attention_mask, repeat_kv
from gqa.kv_cache import KVCache, SlidingWindowKVCache
from gqa.rope import RotaryEmbedding
from tests.gqa.conftest import ATOL, RTOL

D_MODEL, N_HEADS, MAX_SEQ = 32, 4, 64


def make_attn(n_kv_heads: int, *, rope: bool = True, bias: bool = False, dropout: float = 0.0):
    r = RotaryEmbedding(D_MODEL // N_HEADS, MAX_SEQ) if rope else None
    return GroupedQueryAttention(D_MODEL, N_HEADS, n_kv_heads, rope=r, dropout=dropout, bias=bias).double().eval()


def x64(*shape):
    return torch.randn(*shape, dtype=torch.float64)


# --------------------------------------------------------------------------- cache == recompute
@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])  # MHA, GQA, MQA
@pytest.mark.parametrize("batch", [1, 3])
def test_cached_decode_equals_full_sequence_every_step(n_kv_heads: int, batch: int):
    attn = make_attn(n_kv_heads)
    prompt_len, steps = 5, 6
    x = x64(batch, prompt_len + steps, D_MODEL)
    full = attn(x)  # uncached reference over the whole sequence

    cache = KVCache(batch, n_kv_heads, MAX_SEQ, D_MODEL // N_HEADS, dtype=torch.float64)
    out = attn(x[:, :prompt_len], cache=cache, start_pos=0)
    torch.testing.assert_close(out, full[:, :prompt_len], atol=ATOL, rtol=RTOL)
    for t in range(prompt_len, prompt_len + steps):
        out = attn(x[:, t : t + 1], cache=cache, start_pos=t)
        assert out.shape == (batch, 1, D_MODEL)
        torch.testing.assert_close(out[:, 0], full[:, t], atol=ATOL, rtol=RTOL)
    assert cache.seq_len == prompt_len + steps


def test_multi_token_decode_chunks_equal_full_sequence():
    attn = make_attn(2)
    x = x64(3, 12, D_MODEL)
    full = attn(x)
    cache = KVCache(3, 2, MAX_SEQ, D_MODEL // N_HEADS, dtype=torch.float64)
    pos = 0
    for chunk in (4, 3, 1, 2, 2):
        out = attn(x[:, pos : pos + chunk], cache=cache, start_pos=pos)
        assert out.shape == (3, chunk, D_MODEL)
        torch.testing.assert_close(out, full[:, pos : pos + chunk], atol=ATOL, rtol=RTOL)
        pos += chunk
    assert cache.seq_len == 12


def test_cache_stores_kv_heads_not_query_heads():
    attn = make_attn(1)
    cache = KVCache(2, 1, MAX_SEQ, D_MODEL // N_HEADS, dtype=torch.float64)
    attn(x64(2, 4, D_MODEL), cache=cache)
    assert cache.k.shape[1] == 1  # not N_HEADS


# --------------------------------------------------------------------------- MHA reference
def test_mha_special_case_matches_nn_multihead_attention():
    attn = make_attn(N_HEADS, rope=False, bias=True)
    ref = nn.MultiheadAttention(D_MODEL, N_HEADS, bias=True, batch_first=True).double().eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]))
        ref.in_proj_bias.copy_(torch.cat([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias]))
        ref.out_proj.weight.copy_(attn.o_proj.weight)
        ref.out_proj.bias.copy_(attn.o_proj.bias)

    x = x64(3, 9, D_MODEL)
    causal_block = torch.triu(torch.ones(9, 9, dtype=torch.bool), diagonal=1)  # torch: True = blocked
    expected, _ = ref(x, x, x, attn_mask=causal_block, need_weights=False)
    torch.testing.assert_close(attn(x), expected, atol=ATOL, rtol=RTOL)

    # Non-causal too.
    expected_nc, _ = ref(x, x, x, need_weights=False)
    torch.testing.assert_close(attn(x, is_causal=False), expected_nc, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- grouping semantics
def _reference_grouped(attn: GroupedQueryAttention, x: torch.Tensor) -> torch.Tensor:
    """Loop over query heads explicitly; head h uses KV head h // n_rep."""
    b, t, _ = x.shape
    hd = attn.head_dim
    q = attn.q_proj(x).view(b, t, attn.n_heads, hd)
    k = attn.k_proj(x).view(b, t, attn.n_kv_heads, hd)
    v = attn.v_proj(x).view(b, t, attn.n_kv_heads, hd)
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    heads = []
    for h in range(attn.n_heads):
        g = h // attn.n_rep
        s = q[:, :, h] @ k[:, :, g].transpose(1, 2) / math.sqrt(hd)
        s = s.masked_fill(~causal, float("-inf"))
        heads.append(torch.softmax(s, -1) @ v[:, :, g])
    return attn.o_proj(torch.cat(heads, dim=-1))


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
def test_matches_explicit_per_head_reference(n_kv_heads: int):
    attn = make_attn(n_kv_heads, rope=False)
    x = x64(2, 7, D_MODEL)
    torch.testing.assert_close(attn(x), _reference_grouped(attn, x), atol=ATOL, rtol=RTOL)


def test_repeat_kv_interleaves_groups():
    x = torch.arange(2.0).view(1, 2, 1, 1)  # two KV heads: 0, 1
    assert repeat_kv(x, 3).flatten().tolist() == [0, 0, 0, 1, 1, 1]
    assert repeat_kv(x, 1) is x


# --------------------------------------------------------------------------- masking
def test_causal_mask_with_start_pos_offset():
    q_pos = torch.arange(5, 8)  # three new queries at positions 5,6,7
    k_pos = torch.arange(8)  # eight cached keys
    mask = build_attention_mask(q_pos, k_pos)[0, 0]
    expected = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_decode_step_ignores_future_tokens():
    """Output at position t must not change when later tokens exist in the sequence."""
    attn = make_attn(2)
    x = x64(1, 10, D_MODEL)
    y = x.clone()
    y[:, 7:] = x64(1, 3, D_MODEL)  # perturb the future
    torch.testing.assert_close(attn(x)[:, :7], attn(y)[:, :7], atol=ATOL, rtol=RTOL)
    assert not torch.allclose(attn(x)[:, 7:], attn(y)[:, 7:])


def test_start_pos_without_cache_is_rejected():
    attn = make_attn(2)
    with pytest.raises(ValueError, match="requires a cache"):
        attn(x64(1, 1, D_MODEL), start_pos=3)


def test_key_padding_mask_left_padded_batch():
    """A left-padded short sequence must equal the same sequence run alone."""
    attn = make_attn(2)
    short = x64(1, 4, D_MODEL)
    long_ = x64(1, 6, D_MODEL)
    pad = torch.zeros(1, 2, D_MODEL, dtype=torch.float64)
    batch = torch.cat([torch.cat([pad, short], dim=1), long_], dim=0)  # (2, 6, D)
    key_mask = torch.tensor([[False, False, True, True, True, True], [True] * 6])

    out = attn(batch, attn_mask=key_mask)
    # Padding tokens must not change the real tokens of the padded row. The padded
    # row sits at RoPE positions 2..5 and the reference at 0..3; RoPE only depends on
    # relative offsets, so the outputs must still match exactly.
    torch.testing.assert_close(out[0, 2:], attn(short)[0], atol=ATOL, rtol=RTOL)
    # The unpadded row is unaffected by sharing a batch with a padded one.
    torch.testing.assert_close(out[1], attn(long_)[0], atol=ATOL, rtol=RTOL)
    # Fully-masked padding queries produce finite output (no NaN).
    assert torch.isfinite(out).all()


def test_padding_mask_with_cached_decode():
    """Key padding mask over the cache during decode, matching an unpadded run."""
    attn = make_attn(2)
    prompt = x64(1, 3, D_MODEL)
    nxt = x64(1, 1, D_MODEL)
    # Unpadded reference.
    ref_cache = KVCache(1, 2, MAX_SEQ, attn.head_dim, dtype=torch.float64)
    attn(prompt, cache=ref_cache)
    ref = attn(nxt, cache=ref_cache, start_pos=3)
    # Left-padded by 2, with a cache-wide key mask.
    cache = KVCache(1, 2, MAX_SEQ, attn.head_dim, dtype=torch.float64)
    padded = torch.cat([torch.zeros(1, 2, D_MODEL, dtype=torch.float64), prompt], dim=1)
    mask5 = torch.tensor([[False, False, True, True, True]])
    attn(padded, cache=cache, attn_mask=mask5)
    mask6 = torch.cat([mask5, torch.tensor([[True]])], dim=1)
    out = attn(nxt, cache=cache, start_pos=5, attn_mask=mask6)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_sliding_window_cache_matches_windowed_reference():
    attn = make_attn(2)
    window = 4
    x = x64(1, 9, D_MODEL)
    # Reference: full sequence with an explicit band mask (each query sees <= window keys).
    t = torch.arange(9)
    band = (t[None, :] <= t[:, None]) & (t[None, :] > t[:, None] - window)
    ref = attn(x, attn_mask=band[None])
    cache = SlidingWindowKVCache(1, 2, MAX_SEQ, attn.head_dim, window=window, dtype=torch.float64)
    outs = [attn(x[:, :3], cache=cache, start_pos=0)]
    for pos in range(3, 9):
        outs.append(attn(x[:, pos : pos + 1], cache=cache, start_pos=pos))
    torch.testing.assert_close(torch.cat(outs, dim=1), ref, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- training / shapes
def test_gradients_flow_in_training_mode():
    attn = make_attn(2, dropout=0.1).train()
    x = x64(2, 6, D_MODEL).requires_grad_(True)
    attn(x).pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in attn.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_dropout_only_active_in_training():
    attn = make_attn(2, dropout=0.5)
    x = x64(1, 5, D_MODEL)
    torch.testing.assert_close(attn(x), attn(x))  # eval: deterministic
    attn.train()
    assert not torch.allclose(attn(x), attn(x))


@pytest.mark.parametrize("batch,seq", [(1, 1), (3, 5), (7, 2)])
def test_output_shapes(batch: int, seq: int):
    attn = make_attn(2)
    assert attn(x64(batch, seq, D_MODEL)).shape == (batch, seq, D_MODEL)


def test_constructor_validation():
    with pytest.raises(ValueError, match="divisible by n_kv_heads"):
        GroupedQueryAttention(32, 4, 3)
    with pytest.raises(ValueError, match="divisible by n_heads"):
        GroupedQueryAttention(30, 4, 2)
    with pytest.raises(ValueError, match="rope.head_dim"):
        GroupedQueryAttention(32, 4, 2, rope=RotaryEmbedding(16, 8))
    with pytest.raises(TypeError, match="boolean"):
        make_attn(2)(x64(1, 2, D_MODEL), attn_mask=torch.ones(1, 2))


def test_n_kv_heads_defaults_to_n_heads():
    assert GroupedQueryAttention(32, 4).n_kv_heads == 4
