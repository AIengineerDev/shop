"""CSV column contract for data/ranking_items.csv (spec §2)."""

from __future__ import annotations

import pandas as pd

COLUMNS: list[str] = [
    "session_id",
    "qid",
    "query_text",
    "user_id",
    "item_id",
    "device",
    "user_aov_tier",
    "bm25",
    "semantic_sim",
    "price",
    "hist_ctr_7d",
    "rating",
    "displayed_pos",
    "clicked",
    "added_to_cart",
    "purchased",
    "relevance_grade",
    "impression_ts",
    "wishlisted",
    "seller_id",
    "category",
]

DTYPES: dict[str, str] = {
    "session_id": "string",
    "qid": "int64",
    "query_text": "string",
    "user_id": "string",
    "item_id": "string",
    "device": "string",
    "user_aov_tier": "string",
    "bm25": "float64",
    "semantic_sim": "float64",
    "price": "float64",
    "hist_ctr_7d": "float64",
    "rating": "float64",
    "displayed_pos": "int64",
    "clicked": "int64",
    "added_to_cart": "int64",
    "purchased": "int64",
    "relevance_grade": "int64",
    "impression_ts": "float64",
    "wishlisted": "int64",
    "seller_id": "string",
    "category": "string",
}

GROUP_KEYS = ["session_id", "qid", "user_id", "item_id"]
LABELS = ["clicked", "wishlisted", "added_to_cart", "purchased", "relevance_grade"]


class SchemaError(ValueError):
    pass


def read_csv(path) -> pd.DataFrame:
    """Read a ranking CSV with the contract dtypes. Empty query_text stays an empty string."""
    df = pd.read_csv(path, dtype={c: t for c, t in DTYPES.items()}, keep_default_na=True)
    df["query_text"] = df["query_text"].fillna("")
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise SchemaError if columns, order, or dtypes break the contract."""
    if list(df.columns) != COLUMNS:
        missing = [c for c in COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in COLUMNS]
        raise SchemaError(f"header mismatch; missing={missing} extra={extra} or wrong order")
    bad = {c: str(df[c].dtype) for c, t in DTYPES.items() if str(df[c].dtype) != t}
    if bad:
        raise SchemaError(f"dtype mismatch: {bad}")
    null_keys = [c for c in GROUP_KEYS + LABELS if df[c].isna().any()]
    if null_keys:
        raise SchemaError(f"nulls in keys/labels: {null_keys}")
