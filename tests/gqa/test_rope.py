from __future__ import annotations

import pytest
import torch

from gqa.rope import RotaryEmbedding, apply_rope
from tests.gqa.conftest import ATOL, RTOL


def test_relative_position_invariance():
    """<R(p+s) q, R(p'+s) k> must not depend on the shift s."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=128).double()
    q = torch.randn(1, 2, 1, 16, dtype=torch.float64)
    k = torch.randn(1, 2, 1, 16, dtype=torch.float64)

    def score(pq: int, pk: int) -> torch.Tensor:
        rq = rope(q, torch.tensor([pq]))
        rk = rope(k, torch.tensor([pk]))
        return (rq * rk).sum(-1)

    base = score(3, 7)
    for shift in (0, 1, 10, 50):
        torch.testing.assert_close(score(3 + shift, 7 + shift), base, atol=ATOL, rtol=RTOL)
    # And a *different* relative offset gives a different score.
    assert not torch.allclose(score(3, 8), base, atol=1e-6)


def test_batched_positions_equal_per_token_with_start_pos():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=32)
    x = torch.randn(2, 3, 5, 8)
    full = rope(x, torch.arange(5))
    start_pos = 2
    chunk = rope(x[:, :, start_pos:], torch.arange(start_pos, 5))
    torch.testing.assert_close(chunk, full[:, :, start_pos:])


def test_position_zero_is_identity():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=4)
    x = torch.randn(1, 1, 1, 8)
    torch.testing.assert_close(rope(x, torch.tensor([0])), x)


def test_rotation_preserves_norm():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 2, 4, 8)
    torch.testing.assert_close(rope(x, torch.arange(4)).norm(dim=-1), x.norm(dim=-1))


def test_apply_rope_matches_module():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    x = torch.randn(1, 1, 4, 8)
    pos = torch.tensor([5, 6, 7, 8])
    torch.testing.assert_close(apply_rope(x, rope.cos[pos], rope.sin[pos]), rope(x, pos))


def test_errors():
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(head_dim=7, max_seq_len=8)
    rope = RotaryEmbedding(head_dim=8, max_seq_len=8)
    x = torch.randn(1, 1, 2, 8)
    with pytest.raises(ValueError, match="max_seq_len"):
        rope(x, torch.tensor([7, 8]))
    with pytest.raises(ValueError, match="length T"):
        rope(x, torch.tensor([0, 1, 2]))
