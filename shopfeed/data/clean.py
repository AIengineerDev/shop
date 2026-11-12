"""Spec §9.1 regex cleaning guardrails.

`clean(raw)` takes the CSV as all-string columns, applies every rule in order, and returns
(clean typed frame, report). Each rule counts what it touched; `CleanError` is raised when a
drop rule exceeds its threshold, because a high drop rate means the upstream log is broken.
Raw query text is never written to the report, only rule names and counts.

Feed-mode deviation: an empty query_text is allowed (the feed log has no query); rules 5-10
apply only to non-empty queries.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shopfeed import labels
from shopfeed.schema import COLUMNS, DTYPES


class CleanError(RuntimeError):
    def __init__(self, msg: str, report: dict):
        super().__init__(msg)
        self.report = report


@dataclass(frozen=True)
class Rule:
    name: str
    action: str  # normalise | drop_row | drop_group | nan | clamp
    unit: str = "rows"  # rows | groups (denominator for the threshold)
    threshold: float | None = None  # fail-build if rate > threshold


ID_RULES = {
    "session_id": (re.compile(r"s_[0-9]{1,12}"), "drop_group", 0.005),
    "qid": (re.compile(r"[0-9]{1,12}"), "drop_group", 0.001),
    "user_id": (re.compile(r"u_[0-9]{1,12}|anon_[0-9a-f]{8,32}"), "drop_group", 0.005),
    "item_id": (re.compile(r"p_[0-9]{1,12}"), "drop_row", 0.005),
    "seller_id": (re.compile(r"m_[0-9]{1,12}"), "drop_row", 0.005),
}

DEVICE_MAP = {"iphone": "mobile", "android": "mobile", "phone": "mobile",
              "pc": "desktop", "web": "desktop", "laptop": "desktop",
              "ios_app": "app", "android_app": "app"}
DEVICE_RE = re.compile(r"mobile|desktop|app|unknown")
TIER_RE = re.compile(r"Low|Mid|High")
CATEGORY_RE = re.compile(r"[a-z][a-z0-9_]{0,39}")

QUERY_STEPS = [  # (name, pattern, action) for non-empty queries, in order
    ("query_html", re.compile(r"<[^>]{1,200}>"), "normalise"),
    ("query_control_chars", re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"), "normalise"),
    ("query_pii_email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "drop_group"),
    ("query_pii_phone", re.compile(r"\+?\d[\d\s().-]{7,}\d"), "drop_group"),
    ("query_pii_card", re.compile(r"\b(?:\d[ -]?){13,19}\b"), "drop_group"),
    ("query_injection_url", re.compile(r"https?://|www\.|javascript:|data:", re.I), "drop_group"),
]
QUERY_ALLOWED = re.compile(r"[\w '&+./#-]{1,128}")  # \w covers unicode letters and digits
QUERY_HAS_CONTENT = re.compile(r"[^\W_]")

FLOAT_RE = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")
NUMERIC_RULES = {  # column: (regex, lo, hi, out_of_range_action)
    "bm25": (FLOAT_RE, 0.0, np.inf, "nan"),
    "semantic_sim": (FLOAT_RE, -1.0, 1.0, "nan"),
    "price": (re.compile(r"\d+(\.\d{1,2})?"), 0.0, 1e6, "drop_row"),
    "hist_ctr_7d": (re.compile(r"\d+(\.\d+)?([eE][-+]?\d+)?"), 0.0, 1.0, "nan"),
    "rating": (re.compile(r"\d(\.\d+)?"), 0.0, 5.0, "nan"),
    "impression_ts": (re.compile(r"\d{9,10}(\.\d+)?"), 1e9, 1e10, "drop_group"),
}
BINARY_COLS = ["clicked", "added_to_cart", "purchased", "wishlisted"]

DEFAULT_ROW_THRESHOLD = 0.01
DEFAULT_GROUP_THRESHOLD = 0.01


class _Ctx:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.rows_in = len(df)
        self.groups_in = df["qid"].nunique()
        self.report: dict[str, dict] = {}
        self.drop_rows = pd.Series(False, index=df.index)
        self.drop_groups: set[str] = set()

    def record(self, rule: Rule, mask: pd.Series) -> None:
        mask = mask.fillna(False).astype(bool)
        if rule.action == "drop_group":
            groups = set(self.df.loc[mask, "qid"])
            self.drop_groups |= groups
            count, denom = len(groups), self.groups_in
        else:
            if rule.action == "drop_row":
                self.drop_rows |= mask
            count, denom = int(mask.sum()), self.rows_in
        threshold = rule.threshold
        if threshold is None and rule.action in ("drop_row", "drop_group"):
            threshold = DEFAULT_GROUP_THRESHOLD if rule.unit == "groups" else DEFAULT_ROW_THRESHOLD
        entry = self.report.setdefault(rule.name, {
            "action": rule.action, "unit": "groups" if rule.action == "drop_group" else "rows",
            "count": 0, "threshold": threshold})
        entry["count"] += count
        entry["rate"] = entry["count"] / denom if denom else 0.0


def _full(pattern: re.Pattern, s: pd.Series) -> pd.Series:
    return s.map(lambda v: bool(pattern.fullmatch(v)) if isinstance(v, str) else False)


def _normalise_id(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.strip()
    return s.str.replace(r"^([A-Za-z]+)_", lambda m: m.group(1).lower() + "_", regex=True)


def _clean_ids(c: _Ctx) -> None:
    for col, (pattern, action, threshold) in ID_RULES.items():
        c.df[col] = _normalise_id(c.df[col])
        c.record(Rule(f"id_{col}", action, threshold=threshold), ~_full(pattern, c.df[col]))


def _clean_categoricals(c: _Ctx) -> None:
    d = c.df["device"].fillna("").str.strip().str.lower().replace(DEVICE_MAP)
    c.df["device"] = d
    c.record(Rule("cat_device", "drop_row"), ~_full(DEVICE_RE, d))
    t = c.df["user_aov_tier"].fillna("").str.strip().str.title()
    c.df["user_aov_tier"] = t
    c.record(Rule("cat_user_aov_tier", "drop_group"), ~_full(TIER_RE, t))
    cat = c.df["category"].fillna("").str.strip().str.lower()
    c.df["category"] = cat
    c.record(Rule("cat_category", "drop_row"), ~_full(CATEGORY_RE, cat))


def normalise_query(q: str) -> str:
    q = unicodedata.normalize("NFKC", q).strip().lower()
    return re.sub(r"\s+", " ", q)


def _clean_query(c: _Ctx) -> None:
    q = c.df["query_text"].fillna("").map(normalise_query)
    dropped = pd.Series(False, index=q.index)  # each group is counted by the first rule only
    for name, pattern, action in QUERY_STEPS:
        hit = q.map(lambda v, p=pattern: bool(p.search(v)))
        if action == "normalise":
            c.record(Rule(name, "normalise"), hit)
            q = q.map(lambda v, p=pattern: p.sub("", v)).map(normalise_query)
        else:
            c.record(Rule(name, action), hit & ~dropped)
            dropped |= hit
    live = (q != "") & ~dropped
    bad_charset = live & ~_full(QUERY_ALLOWED, q)
    c.record(Rule("query_charset", "drop_group"), bad_charset)
    c.record(Rule("query_min_content", "drop_group"),
             live & ~bad_charset & ~q.map(lambda v: bool(QUERY_HAS_CONTENT.search(v))))
    # PII text is blanked so it cannot leak downstream even before the group is dropped.
    c.df["query_text"] = q.mask(dropped, "")


def _clean_numeric(c: _Ctx) -> None:
    for col, (pattern, lo, hi, action) in NUMERIC_RULES.items():
        raw = c.df[col].fillna("").astype(str).str.strip()
        if col == "price":
            raw = raw.str.replace(r"[$€£,]", "", regex=True)
        empty = raw.isin(["", "nan", "NaN"])
        parse_ok = _full(pattern, raw)
        bad_format = ~empty & ~parse_ok
        val = pd.to_numeric(raw.where(parse_ok), errors="coerce")
        out_of_range = val.notna() & ((val < lo) | (val > hi))
        if col == "price":
            # Price is required: unparseable, missing or out of range drops the row.
            c.record(Rule("num_price", "drop_row"), bad_format | empty | out_of_range)
        elif action == "drop_group":
            c.record(Rule(f"num_{col}", "drop_group"), bad_format | empty | out_of_range)
        else:
            c.record(Rule(f"num_{col}_unparseable", "nan"), bad_format)
            c.record(Rule(f"num_{col}_out_of_range", "nan"), out_of_range)
            val = val.mask(out_of_range)
        c.df[col] = val


def _clean_labels(c: _Ctx) -> None:
    for col in BINARY_COLS:
        ok = _full(re.compile(r"[01]"), c.df[col].fillna("").astype(str).str.strip())
        c.record(Rule(f"label_{col}", "drop_row"), ~ok)
        c.df[col] = pd.to_numeric(c.df[col].where(ok), errors="coerce")
    g_ok = _full(re.compile(r"[0-4]"), c.df["relevance_grade"].fillna("").astype(str).str.strip())
    c.df["relevance_grade"] = pd.to_numeric(c.df["relevance_grade"].where(g_ok), errors="coerce")
    c.record(Rule("label_relevance_grade_format", "drop_row"), ~g_ok)
    complete = c.df[BINARY_COLS + ["relevance_grade"]].notna().all(axis=1)
    sub = c.df[complete]
    inconsistent = pd.Series(False, index=c.df.index)
    inconsistent[complete] = labels.consistency_violations(sub)
    c.record(Rule("label_consistency", "drop_row"), inconsistent)
    pos_ok = _full(re.compile(r"[1-9]\d{0,2}"), c.df["displayed_pos"].fillna("").astype(str))
    c.record(Rule("num_displayed_pos", "drop_group"), ~pos_ok)
    c.df["displayed_pos"] = pd.to_numeric(c.df["displayed_pos"].where(pos_ok), errors="coerce")


def _structural(c: _Ctx) -> pd.DataFrame:
    df = c.df[~c.drop_rows & ~c.df["qid"].isin(c.drop_groups)]
    exact = df.duplicated(keep="first")
    c.record(Rule("dup_exact_row", "drop_row"), exact.reindex(c.df.index, fill_value=False))
    df = df[~exact]
    key = df.duplicated(["qid", "item_id"], keep=False)
    c.record(Rule("dup_qid_item", "drop_group"), key.reindex(c.df.index, fill_value=False))
    pos = df.duplicated(["qid", "displayed_pos"], keep=False)
    c.record(Rule("dup_qid_position", "drop_group"), pos.reindex(c.df.index, fill_value=False))
    df = df[~df["qid"].isin(c.drop_groups)]
    small = df.groupby("qid")["item_id"].transform("size") < 2
    c.record(Rule("group_lt_2_rows", "drop_group"), small.reindex(c.df.index, fill_value=False))
    return df[~df["qid"].isin(c.drop_groups)]


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply §9.1 to an all-string frame. Returns (typed clean frame, report)."""
    if list(raw.columns) != COLUMNS:
        raise CleanError("header does not match schema.COLUMNS", {})
    c = _Ctx(raw.copy())
    _clean_ids(c)
    _clean_categoricals(c)
    _clean_query(c)
    _clean_numeric(c)
    _clean_labels(c)
    out = _structural(c)
    out = out.astype({k: v for k, v in DTYPES.items()}).reset_index(drop=True)
    out["query_text"] = out["query_text"].fillna("")

    report = {
        "rows_in": c.rows_in, "rows_out": len(out),
        "groups_in": int(c.groups_in), "groups_out": int(out["qid"].nunique()),
        "rules": c.report,
    }
    failed = [n for n, r in c.report.items()
              if r["threshold"] is not None and r["rate"] > r["threshold"]]
    report["failed_rules"] = failed
    if failed:
        raise CleanError(f"drop threshold exceeded: {failed}", report)
    return out, report


def read_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
