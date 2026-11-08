import numpy as np
import pandas as pd
import pytest

from shopfeed.config import DAY, HEADS
from shopfeed.features.streaming import StreamingStore
from shopfeed.training.dataset import build_labels, build_training_set, time_split

H = {h: i for i, h in enumerate(HEADS)}
W = 7 * DAY


def _imps():
    return pd.DataFrame({"impression_id": ["i1", "i2", "i3"], "ts": [1000.0, 2000.0, 3000.0], "card_type": "product"})


def test_label_window_boundaries():
    ev = pd.DataFrame({
        "impression_id": ["i1", "i1", "i2", "i3", "i3"],
        "type": ["purchase", "return", "hide", "view", "save"],
        "ts": [1000.0 + W - 1, 1000.0 + W + 1, 2000.0 + 10, 3000.0 - 1, 3000.0 + 5],
    })
    Y = build_labels(_imps(), ev, W)
    assert Y[0, H["purchase"]] == 1 and Y[0, H["return"]] == 0     # +7d-1s counts, +7d+1s does not
    assert Y[1, H["hide"]] == 1
    assert Y[2, H["click"]] == 0 and Y[2, H["save"]] == 1           # event before the impression is not a label


def test_time_split_ordered_with_gap_and_disjoint():
    meta = pd.DataFrame({"ts": np.linspace(0, 100 * DAY, 1000)})
    tr, va = time_split(meta, 0.2, W)
    assert len(set(tr) & set(va)) == 0
    assert meta.ts.values[tr].max() + W <= meta.ts.values[va].min()
    assert len(va) > 0 and len(tr) > 0


def test_training_set_negatives_have_no_positive_and_order_status_excluded(world, streaming, cfg):
    tables = {k: v.copy() for k, v in world.items()}
    extra = tables["impressions"].head(5).copy()
    extra["impression_id"] = [f"os_{i}" for i in range(5)]
    extra["card_type"] = "order_status"
    tables["impressions"] = pd.concat([tables["impressions"], extra], ignore_index=True)
    ts = build_training_set(tables, streaming, 1_750_000_000.0, W)
    assert not ts.meta.impression_id.str.startswith("os_").any()
    neg = ts.any_positive == 0
    pos_cols = [H[h] for h in ("click", "dwell", "save", "atc", "purchase")]
    assert ts.Y[neg][:, pos_cols].sum() == 0
    assert neg.sum() > 0 and (~neg).sum() > 0
    # impressions whose label window has not closed are excluded
    assert (ts.meta.ts + W <= 1_750_000_000.0).all()


def test_empty_log_raises_clear_error(world, streaming):
    tables = {k: v.copy() for k, v in world.items()}
    tables["impressions"] = tables["impressions"].iloc[0:0]
    with pytest.raises(ValueError, match="empty impression log"):
        build_training_set(tables, streaming, 1_750_000_000.0, W)


def test_features_in_training_set_have_no_nan(trained):
    assert not np.isnan(trained["ts"].X).any()
