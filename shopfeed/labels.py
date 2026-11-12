"""Graded relevance label (spec §4.2) and §4.3 consistency rules.

Grade ladder: 0 no click, 1 click, 2 wishlist, 3 add to cart, 4 purchase.
`hide` demotes a click-only row (grade 1) to 0. Wishlist, ATC and purchase outrank a hide,
so the §4.3 implications always hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GRADE_COLUMNS = ["clicked", "wishlisted", "added_to_cart", "purchased"]


def grade(df: pd.DataFrame, hidden: pd.Series | None = None) -> pd.Series:
    """Vectorised relevance_grade from binary columns (and an optional `hidden` flag)."""
    g = np.select(
        [df["purchased"] == 1, df["added_to_cart"] == 1, df["wishlisted"] == 1, df["clicked"] == 1],
        [4, 3, 2, 1],
        default=0,
    )
    if hidden is not None:
        g = np.where((hidden.to_numpy() == 1) & (g < 2), 0, g)
    return pd.Series(g, index=df.index, name="relevance_grade", dtype="int64")


def backfill_clicked(df: pd.DataFrame) -> pd.Series:
    """A wishlist / ATC / purchase without a logged click still counts as a click (§4.3)."""
    downstream = (df["wishlisted"] == 1) | (df["added_to_cart"] == 1) | (df["purchased"] == 1)
    return (df["clicked"].astype(bool) | downstream).astype("int64")


def consistency_violations(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that break a §4.3 implication."""
    g = df["relevance_grade"]
    return (
        ((df["purchased"] == 1) & (g != 4))
        | ((df["added_to_cart"] == 1) & (g < 3))
        | ((df["wishlisted"] == 1) & (g < 2))
        | ((df["clicked"] == 0) & (g != 0))
    )
