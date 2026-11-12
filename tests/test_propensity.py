"""Propensity curve (spec §5.4) and IPW-NDCG (§8.3)."""

import numpy as np
import pandas as pd
import pytest

from shopfeed.eval.metrics import ipw_ndcg_at_k, ndcg_at_k
from shopfeed.eval.propensity import estimate_from_randomised, lookup


def _rows(ctr_by_pos, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"displayed_pos": p, "clicked": int(rng.random() < c)}
                         for p, c in ctr_by_pos.items() for _ in range(n)])


def test_curve_is_monotone_normalised_and_clipped():
    rows = _rows({1: 0.30, 2: 0.24, 3: 0.25, 4: 0.12, 5: 0.01})
    c = estimate_from_randomised(rows)
    p = c["propensity"]
    assert p[0] == 1.0
    assert all(a >= b for a, b in zip(p, p[1:], strict=False))
    assert min(p) == pytest.approx(0.1)  # position 5 (~0.03 raw) clipped
    assert c["raw"][4] < 0.1


def test_lookup_extends_last_value():
    c = {"positions": [1, 2, 3], "propensity": [1.0, 0.5, 0.25]}
    assert list(lookup(c, [1, 3, 9])) == [1.0, 0.25, 0.25]


def test_ipw_ndcg_equals_ndcg_with_unit_propensity():
    g, s = [3, 2, 0, 1], [4, 3, 2, 1]
    assert ipw_ndcg_at_k(g, s, [1, 1, 1, 1]) == pytest.approx(ndcg_at_k(g, s))


def test_ipw_upweights_deep_clicks():
    # Two clicked items; the one logged deep (low propensity) gets more weight, so ranking
    # it first scores higher under IPW than ranking the shallow one first.
    g, prop = [1, 1, 0], [1.0, 0.2, 1.0]
    deep_first = ipw_ndcg_at_k(g, [1, 2, 0], prop)
    shallow_first = ipw_ndcg_at_k(g, [2, 1, 0], prop)
    assert deep_first > shallow_first
    assert ndcg_at_k(g, [1, 2, 0]) == ndcg_at_k(g, [2, 1, 0])
