"""§9.1 regex guardrails: one positive and one negative per rule, plus a dirty-CSV count test."""

import pandas as pd
import pytest

from shopfeed import config
from shopfeed.data import clean as C


@pytest.fixture
def raw():
    return C.read_raw(config.FIXTURE_CSV)


def test_clean_fixture_has_zero_drops(raw):
    df, report = C.clean(raw)
    assert len(df) == 10
    assert report["failed_rules"] == []
    assert all(r["count"] == 0 for r in report["rules"].values())


@pytest.mark.parametrize("col,good,bad", [
    ("session_id", "s_101", "sess-101"),
    ("qid", "12", "q12"),
    ("user_id", "u_89", "user89"),
    ("user_id", "anon_deadbeef", "anon_xyz"),
    ("item_id", "p_301", "p301"),
    ("seller_id", "m_3", "merchant_3"),
])
def test_id_regex(col, good, bad):
    pattern = C.ID_RULES[col][0]
    assert pattern.fullmatch(good)
    assert not pattern.fullmatch(bad)


def test_id_prefix_is_lowercased_before_match():
    assert C._normalise_id(pd.Series([" P_301 "])).iloc[0] == "p_301"


@pytest.mark.parametrize("pattern,good,bad", [
    (C.DEVICE_RE, "unknown", "tablet"),
    (C.TIER_RE, "High", "VIP"),
    (C.CATEGORY_RE, "home_decor", "Home Decor!"),
    (C.QUERY_ALLOWED, "usb-c cable 2m", "<script>"),
])
def test_categorical_and_charset_regex(pattern, good, bad):
    assert pattern.fullmatch(good)
    assert not pattern.fullmatch(bad)


@pytest.mark.parametrize("name,hit,miss", [
    ("query_html", "red <b>shoes</b>", "red shoes"),
    ("query_control_chars", "red\x07shoes", "red shoes"),
    ("query_pii_email", "mail me a@b.com", "email marketing book"),
    ("query_pii_phone", "call +1 415 555 0100", "iphone 15"),
    ("query_pii_card", "4111 1111 1111 1111", "size 42"),
    ("query_injection_url", "https://evil.example", "web cam"),
])
def test_query_rules(name, hit, miss):
    pattern = next(p for n, p, _ in C.QUERY_STEPS if n == name)
    assert pattern.search(hit)
    assert not pattern.search(miss)


@pytest.mark.parametrize("col,good,bad", [
    ("bm25", "14.8", "high"),
    ("semantic_sim", "-0.25", "0,88"),
    ("price", "199.99", "199.999"),
    ("hist_ctr_7d", "0.082", "-0.1"),
    ("rating", "4.6", "45"),
    ("impression_ts", "1746100000.0", "2025-05-01"),
])
def test_numeric_regex(col, good, bad):
    pattern = C.NUMERIC_RULES[col][0]
    assert pattern.fullmatch(good)
    assert not pattern.fullmatch(bad)


def test_feed_mode_allows_empty_query(raw):
    raw["query_text"] = ""
    df, report = C.clean(raw)
    assert len(df) == 10
    assert report["rules"]["query_min_content"]["count"] == 0


def _dirty(raw: pd.DataFrame) -> pd.DataFrame:
    """Fixture + 8 extra groups, each carrying exactly one violation."""
    groups = []
    base = raw[raw["qid"] == "2"]
    for i, mutate in enumerate([
        lambda g: g.assign(session_id="bad session"),          # id_session_id (group)
        lambda g: g.assign(user_aov_tier="VIP"),                # cat_user_aov_tier (group)
        lambda g: g.assign(query_text="mail a@b.com"),          # query_pii_email (group)
        lambda g: g.assign(query_text="see https://x.io"),      # query_injection_url (group)
        lambda g: g.assign(impression_ts="yesterday"),          # num_impression_ts (group)
        lambda g: g.assign(displayed_pos=["1", "1", "3", "4", "5"]),  # dup_qid_position
        lambda g: g.iloc[:1],                                  # group_lt_2_rows
        lambda g: g.assign(price=["$79.99", "x", "45.00", "299.99", "15.99"]),  # num_price
    ]):
        g = mutate(base.copy()).assign(qid=str(100 + i))
        groups.append(g)
    dirty = pd.concat([raw, *groups], ignore_index=True)
    dup = dirty.iloc[[0]]
    return pd.concat([dirty, dup], ignore_index=True)  # dup_exact_row


def test_dirty_csv_exact_counts(raw, monkeypatch):
    monkeypatch.setattr(C, "DEFAULT_ROW_THRESHOLD", 1.0)
    monkeypatch.setattr(C, "DEFAULT_GROUP_THRESHOLD", 1.0)
    monkeypatch.setitem(C.ID_RULES, "session_id", (C.ID_RULES["session_id"][0], "drop_group", 1.0))
    df, report = C.clean(_dirty(raw))
    counts = {k: v["count"] for k, v in report["rules"].items() if v["count"]}
    assert counts == {
        "id_session_id": 1, "cat_user_aov_tier": 1, "query_pii_email": 1,
        "query_injection_url": 1, "num_impression_ts": 1, "dup_qid_position": 1,
        "group_lt_2_rows": 1, "num_price": 1, "dup_exact_row": 1,
    }
    assert set(df["qid"]) == {1, 2, 107}  # fixture groups + the price group minus one row
    assert len(df[df["qid"] == 107]) == 4
    assert "a@b.com" not in str(report)


def test_threshold_breach_raises(raw):
    raw.loc[raw["qid"] == "1", "session_id"] = "broken"
    with pytest.raises(C.CleanError) as e:
        C.clean(raw)
    assert "id_session_id" in e.value.report["failed_rules"]
