"""Impression log: every served card with the features, scores, position, propensity and model
version that produced it. This is the training data for the next model."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from shopfeed.schema import Impression


class ImpressionLogger:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self.rows: List[dict] = []

    def log(self, impressions: List[Impression]) -> None:
        recs = [i.model_dump() for i in impressions]
        self.rows.extend(recs)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                for r in recs:
                    f.write(pd.Series(r).to_json() + "\n")

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=list(Impression.model_fields))
        return pd.DataFrame(self.rows)

    def __len__(self) -> int:
        return len(self.rows)
