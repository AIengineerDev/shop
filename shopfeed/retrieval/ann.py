"""Brute-force cosine index behind the same interface a FAISS/ScaNN index would expose."""
from __future__ import annotations

from typing import Tuple

import numpy as np


class BruteForceIndex:
    def __init__(self) -> None:
        self.ids = np.zeros(0, dtype=object)
        self.vecs = np.zeros((0, 0), dtype=np.float32)

    def build(self, ids: np.ndarray, vecs: np.ndarray) -> "BruteForceIndex":
        if len(ids) != len(vecs):
            raise ValueError("ids and vecs length mismatch")
        self.ids = np.asarray(ids, dtype=object)
        v = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        self.vecs = v / np.where(norms == 0, 1, norms)
        return self

    def add(self, ids: np.ndarray, vecs: np.ndarray) -> "BruteForceIndex":
        """Insert newly published items. Called from the publish path so cold items are retrievable at once."""
        if len(self.ids) == 0:
            return self.build(ids, vecs)
        v = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        self.ids = np.concatenate([self.ids, np.asarray(ids, dtype=object)])
        self.vecs = np.vstack([self.vecs, v / np.where(norms == 0, 1, norms)])
        return self

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: np.ndarray, k: int, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k by cosine. `mask` (bool, len = index) restricts the search to eligible items."""
        if len(self.ids) == 0 or k <= 0:
            return np.zeros(0, dtype=object), np.zeros(0, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.vecs @ q
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        top = top[np.isfinite(scores[top])]
        return self.ids[top], scores[top]
