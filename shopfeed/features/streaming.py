"""Rolling counters with an explicit clock. Backed by per-product sorted timestamp arrays.

The same class serves two roles: the online store (add_events as they arrive, query at `now`)
and the offline point-in-time store for training (query at each impression's own ts).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from shopfeed.config import HOUR


class StreamingStore:
    def __init__(self) -> None:
        self._product_ts: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)  # type -> pid -> sorted ts
        self._pending: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._seen_event_ids: Set[str] = set()
        self._user_session: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._user_product_last: Dict[str, Dict[str, float]] = defaultdict(dict)  # (user, type)->pid->last ts

    # ------------------------------------------------------------------ ingest
    def add_events(self, events: pd.DataFrame, now: Optional[float] = None) -> Dict[str, int]:
        """Idempotent by event_id. Events with ts > now are ignored (clock skew / bad clients)."""
        accepted = duplicates = ignored_future = 0
        for row in events.itertuples(index=False):
            eid = getattr(row, "event_id")
            if eid in self._seen_event_ids:
                duplicates += 1
                continue
            if now is not None and row.ts > now:
                ignored_future += 1
                continue
            self._seen_event_ids.add(eid)
            accepted += 1
            pid = getattr(row, "product_id", None)
            if isinstance(pid, str):
                self._pending[row.type][pid].append(float(row.ts))
                self._user_product_last[f"{row.user_id}|{row.type}"][pid] = max(
                    float(row.ts), self._user_product_last[f"{row.user_id}|{row.type}"].get(pid, -np.inf))
            self._user_session[row.user_id][row.type] += 1
        self._flush()
        return {"accepted": accepted, "duplicates": duplicates, "ignored_future": ignored_future}

    def _flush(self) -> None:
        for etype, by_pid in self._pending.items():
            for pid, ts_list in by_pid.items():
                old = self._product_ts[etype].get(pid)
                new = np.sort(np.asarray(ts_list, dtype=float))
                self._product_ts[etype][pid] = new if old is None else np.sort(np.concatenate([old, new]))
        self._pending.clear()

    # ------------------------------------------------------------------ queries
    def product_count(self, etype: str, product_ids: Sequence[str], at: np.ndarray, window_s: float = HOUR) -> np.ndarray:
        """Count of `etype` events for each product in (at - window, at]. Vectorised per product."""
        at = np.asarray(at, dtype=float)
        out = np.zeros(len(product_ids), dtype=float)
        table = self._product_ts.get(etype, {})
        pids = np.asarray(product_ids, dtype=object)
        for pid in np.unique(pids):
            ts = table.get(pid)
            if ts is None:
                continue
            mask = pids == pid
            hi = np.searchsorted(ts, at[mask], side="right")
            lo = np.searchsorted(ts, at[mask] - window_s, side="right")
            out[mask] = hi - lo
        return out

    def product_count_now(self, etype: str, product_ids: Sequence[str], now: float, window_s: float = HOUR) -> np.ndarray:
        return self.product_count(etype, product_ids, np.full(len(product_ids), now), window_s)

    def user_products_since(self, user_id: str, etype: str, since: float) -> Set[str]:
        d = self._user_product_last.get(f"{user_id}|{etype}", {})
        return {pid for pid, ts in d.items() if ts >= since}

    def user_last_ts(self, user_id: str, etype: str) -> Dict[str, float]:
        return dict(self._user_product_last.get(f"{user_id}|{etype}", {}))

    def session_counts(self, user_id: str) -> Dict[str, int]:
        return dict(self._user_session.get(user_id, {}))
