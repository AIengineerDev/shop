from shopfeed import config
from shopfeed.data.checks import run_all
from shopfeed.schema import read_csv


def _by_name(results):
    return {r.name: r for r in results}


def test_fixture_fails_scale_by_design():
    r = _by_name(run_all(read_csv(config.FIXTURE_CSV)))
    assert not r["scale"].passed
    for name in ("header", "nulls", "label_consistency", "groups", "semantic_sim_direction",
                 "temporal_column"):
        assert r[name].passed, (name, r[name].detail)


def test_duplicate_position_fails_groups():
    df = read_csv(config.FIXTURE_CSV)
    df.loc[1, "displayed_pos"] = 1
    assert not _by_name(run_all(df))["groups"].passed


def test_inconsistent_label_fails():
    df = read_csv(config.FIXTURE_CSV)
    df.loc[0, "relevance_grade"] = 3  # purchased row
    assert not _by_name(run_all(df))["label_consistency"].passed


def test_results_are_json_safe():
    import json

    from shopfeed.data.checks import as_dicts
    json.dumps(as_dicts(run_all(read_csv(config.FIXTURE_CSV))))
