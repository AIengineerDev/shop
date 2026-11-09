from __future__ import annotations

import pytest
import torch

from gqa.generate import TinyDecoder, generate
from tests.gqa.conftest import ATOL, RTOL

VOCAB = 50


def make_model(n_kv_heads: int = 2, max_seq_len: int = 32, n_layers: int = 2) -> TinyDecoder:
    return TinyDecoder(VOCAB, d_model=32, n_layers=n_layers, n_heads=4, n_kv_heads=n_kv_heads, max_seq_len=max_seq_len).double()


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
def test_cached_generation_equals_uncached(n_kv_heads: int):
    model = make_model(n_kv_heads)
    prompt = torch.randint(0, VOCAB, (2, 5))
    ids_c, logits_c = generate(model, prompt, 6, use_cache=True, return_logits=True)
    ids_u, logits_u = generate(model, prompt, 6, use_cache=False, return_logits=True)
    assert ids_c.shape == (2, 11)
    assert torch.equal(ids_c[:, :5], prompt)
    assert torch.equal(ids_c, ids_u)
    assert logits_c.shape == (2, 6, VOCAB)
    torch.testing.assert_close(logits_c, logits_u, atol=ATOL, rtol=RTOL)


def test_prefill_then_decode_matches_one_shot_forward():
    model = make_model()
    ids = torch.randint(0, VOCAB, (3, 8))
    full = model(ids)
    caches = model.make_caches(3)
    assert len(caches) == 2 and caches[0].k.shape == (3, 2, 32, 8)
    out = [model(ids[:, :4], caches=caches, start_pos=0)]
    for t in range(4, 8):
        out.append(model(ids[:, t : t + 1], caches=caches, start_pos=t))
    torch.testing.assert_close(torch.cat(out, dim=1), full, atol=ATOL, rtol=RTOL)
    assert all(c.seq_len == 8 for c in caches)


def test_generate_overflow_raises():
    model = make_model(max_seq_len=8)
    with pytest.raises(ValueError, match="overflow.*max_seq_len=8"):
        generate(model, torch.randint(0, VOCAB, (1, 5)), 4)


def test_cache_overflow_raises_during_decode():
    model = make_model(max_seq_len=8)
    caches = model.make_caches(1)
    model(torch.randint(0, VOCAB, (1, 6)), caches=caches)
    # The RoPE table (sized to max_seq_len) is checked first; without RoPE the
    # cache itself raises. Either way the error names the limit.
    with pytest.raises(ValueError, match="overflow.*max_seq_len=8"):
        model(torch.randint(0, VOCAB, (1, 3)), caches=caches, start_pos=6)
    cache = caches[0]
    with pytest.raises(ValueError, match="KV cache overflow.*max_seq_len=8"):
        cache.update(6, torch.zeros(1, 2, 3, 8, dtype=torch.float64), torch.zeros(1, 2, 3, 8, dtype=torch.float64))


def test_generate_restores_training_mode_and_sampling_runs():
    model = make_model().train()
    out = generate(model, torch.randint(0, VOCAB, (1, 3)), 2, greedy=False)
    assert out.shape == (1, 5) and model.training


def test_training_step_without_cache_backprops():
    model = make_model().train()
    ids = torch.randint(0, VOCAB, (2, 6))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
