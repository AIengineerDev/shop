from __future__ import annotations

import pytest
import torch

from gqa.kv_cache import KVCache, SlidingWindowKVCache, kv_cache_bytes


def _kv(b: int, h: int, t: int, d: int, fill: float):
    return torch.full((b, h, t, d), fill), torch.full((b, h, t, d), -fill)


def test_shapes_and_seq_len_bookkeeping():
    cache = KVCache(batch=3, n_kv_heads=2, max_seq_len=10, head_dim=4)
    assert cache.k.shape == (3, 2, 10, 4)
    assert cache.v.shape == (3, 2, 10, 4)
    assert cache.seq_len == 0

    k, v = cache.update(0, *_kv(3, 2, 4, 4, 1.0))  # prefill 4
    assert cache.seq_len == 4 and k.shape == (3, 2, 4, 4) and v.shape == k.shape

    k, v = cache.update(4, *_kv(3, 2, 1, 4, 2.0))  # decode 1
    assert cache.seq_len == 5 and k.shape[2] == 5

    k, v = cache.update(5, *_kv(3, 2, 3, 4, 3.0))  # multi-token chunk
    assert cache.seq_len == 8 and k.shape[2] == 8
    assert torch.equal(k[0, 0, :, 0], torch.tensor([1.0, 1, 1, 1, 2, 3, 3, 3]))
    assert torch.equal(v[0, 0, :, 0], -torch.tensor([1.0, 1, 1, 1, 2, 3, 3, 3]))
    assert torch.equal(cache.key_positions(), torch.arange(8))


def test_returned_views_alias_storage():
    cache = KVCache(1, 1, 8, 2)
    k, _ = cache.update(0, *_kv(1, 1, 3, 2, 1.0))
    assert k.data_ptr() == cache.k.data_ptr()


def test_reset_keeps_allocation():
    cache = KVCache(1, 1, 8, 2)
    cache.update(0, *_kv(1, 1, 3, 2, 1.0))
    ptr = cache.k.data_ptr()
    cache.reset()
    assert cache.seq_len == 0 and cache.k.data_ptr() == ptr
    k, _ = cache.update(0, *_kv(1, 1, 1, 2, 5.0))
    assert k.shape[2] == 1


def test_rewind_overwrites():
    cache = KVCache(1, 1, 8, 2)
    cache.update(0, *_kv(1, 1, 4, 2, 1.0))
    k, _ = cache.update(2, *_kv(1, 1, 1, 2, 9.0))
    assert cache.seq_len == 3 and k[0, 0, 2, 0].item() == 9.0


def test_overflow_raises_clear_error():
    cache = KVCache(1, 1, max_seq_len=4, head_dim=2)
    cache.update(0, *_kv(1, 1, 3, 2, 1.0))
    with pytest.raises(ValueError, match="overflow.*max_seq_len=4"):
        cache.update(3, *_kv(1, 1, 2, 2, 1.0))
    assert cache.seq_len == 3  # failed update leaves state untouched


def test_shape_and_position_validation():
    cache = KVCache(2, 2, 8, 4)
    with pytest.raises(ValueError, match="n_kv_heads=2"):
        cache.update(0, *_kv(2, 3, 1, 4, 1.0))
    with pytest.raises(ValueError, match="batch=2"):
        cache.update(0, *_kv(1, 2, 1, 4, 1.0))
    with pytest.raises(ValueError, match="skipped"):
        cache.update(1, *_kv(2, 2, 1, 4, 1.0))


def test_dtype_cast_and_memory_bytes():
    cache = KVCache(2, 4, 16, 8, dtype=torch.bfloat16)
    assert cache.memory_bytes() == 2 * 2 * 4 * 16 * 8 * 2
    k, _ = cache.update(0, *_kv(2, 4, 1, 8, 1.0))
    assert k.dtype == torch.bfloat16


def test_kv_cache_bytes_worked_example():
    # Llama-style: 32 layers, 8 KV heads, head_dim 128, 4k context, bf16, batch 1.
    gqa = kv_cache_bytes(n_layers=32, batch=1, n_kv_heads=8, seq_len=4096, head_dim=128, bytes_per_elem=2)
    assert gqa == 512 * 1024**2
    mha = kv_cache_bytes(n_layers=32, batch=1, n_kv_heads=64, seq_len=4096, head_dim=128, bytes_per_elem=2)
    assert mha == 8 * gqa == 4 * 1024**3


class TestSlidingWindow:
    def test_retains_last_window_positions(self):
        cache = SlidingWindowKVCache(1, 1, max_seq_len=64, head_dim=2, window=4)
        assert cache.k.shape == (1, 1, 4, 2)
        for pos in range(3):
            k, _ = cache.update(pos, *_kv(1, 1, 1, 2, float(pos)))
        assert k.shape[2] == 3 and torch.equal(cache.key_positions(), torch.arange(3))

        for pos in range(3, 7):
            k, _ = cache.update(pos, *_kv(1, 1, 1, 2, float(pos)))
        assert cache.seq_len == 7 and k.shape[2] == 4
        stored = k[0, 0, :, 0]
        positions = cache.key_positions()
        # Storage order is not position order, but values line up with positions.
        assert torch.equal(stored, positions.to(stored.dtype))
        assert sorted(positions.tolist()) == [3, 4, 5, 6]

    def test_chunk_larger_than_window(self):
        cache = SlidingWindowKVCache(1, 1, max_seq_len=64, head_dim=2, window=3)
        k, _ = cache.update(0, torch.arange(6.0).view(1, 1, 6, 1).expand(1, 1, 6, 2), torch.zeros(1, 1, 6, 2))
        assert cache.seq_len == 6
        assert sorted(cache.key_positions().tolist()) == [3, 4, 5]
        assert torch.equal(k[0, 0, :, 0], cache.key_positions().to(k.dtype))

    def test_logical_overflow_and_rewind_limits(self):
        cache = SlidingWindowKVCache(1, 1, max_seq_len=5, head_dim=2, window=2)
        cache.update(0, *_kv(1, 1, 4, 2, 1.0))
        with pytest.raises(ValueError, match="overflow"):
            cache.update(4, *_kv(1, 1, 2, 2, 1.0))
        with pytest.raises(ValueError, match="rewind"):
            cache.update(0, *_kv(1, 1, 1, 2, 1.0))
