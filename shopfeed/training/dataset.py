"""Load the ranking CSV, attach model features and the temporal split in one place."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from shopfeed import config
from shopfeed.data.build import Sources
from shopfeed.features.history import HistoryStore
from shopfeed.features.ranking import build_features
from shopfeed.schema import read_csv
from shopfeed.training.splits import SplitMeta, seen_user_flag, temporal_split

# Feature columns also copied next to the rows for evaluation slices and exposure metrics.
EVAL_COLUMNS = ["item_purchases_30d", "item_age_days", "is_category_top20",
                "seller_item_count", "is_new_user"]


@dataclass
class Dataset:
    rows: pd.DataFrame  # CSV columns + EVAL_COLUMNS + partition + user_seen_in_train
    X: pd.DataFrame
    feature_names: list[str]
    split: SplitMeta
    store: HistoryStore

    def part(self, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        mask = (self.rows["partition"] == name).to_numpy()
        return self.rows[mask].reset_index(drop=True), self.X[mask].reset_index(drop=True)


def load_store(data_dir=config.DATA_DIR, src: Sources | None = None) -> HistoryStore:
    src = src or Sources.load(data_dir)
    merchants = pd.read_parquet(data_dir / "merchants.parquet")
    return HistoryStore.from_sources(src.impressions, src.events, src.products, merchants)


def load_dataset(csv=config.RANKING_CSV, store: HistoryStore | None = None) -> Dataset:
    rows = read_csv(csv).sort_values(["qid", "displayed_pos"]).reset_index(drop=True)
    store = store or load_store()
    X, names = build_features(rows, store)
    partition, meta = temporal_split(rows)
    rows = pd.concat([rows, X[EVAL_COLUMNS]], axis=1)
    rows["partition"] = partition
    rows["user_seen_in_train"] = seen_user_flag(rows, partition)
    return Dataset(rows, X, names, meta, store)
