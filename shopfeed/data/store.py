"""Parquet persistence for the synthetic tables and the impression log."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

TABLES = ["users", "merchants", "products", "follows", "impressions", "events"]


def save_tables(tables: Dict[str, pd.DataFrame], data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(data_dir / f"{name}.parquet", index=False)


def load_tables(data_dir: Path) -> Dict[str, pd.DataFrame]:
    out = {}
    for name in TABLES:
        p = data_dir / f"{name}.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
    missing = [t for t in TABLES if t not in out]
    if missing:
        raise FileNotFoundError(f"missing tables {missing} in {data_dir}; run scripts/generate_data.py")
    return out
