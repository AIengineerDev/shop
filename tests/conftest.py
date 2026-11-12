import pandas as pd
import pytest

from shopfeed import config
from shopfeed.features.history import HistoryStore
from shopfeed.schema import read_csv

EVENT_COLS = ["event_id", "user_id", "product_id", "merchant_id", "type", "ts", "impression_id",
              "value"]


@pytest.fixture(scope="session")
def fixture_rows() -> pd.DataFrame:
    return read_csv(config.FIXTURE_CSV)


@pytest.fixture(scope="session")
def fixture_store(fixture_rows) -> HistoryStore:
    rows = fixture_rows
    events = pd.DataFrame(columns=EVENT_COLS).astype({"ts": float})
    products = pd.DataFrame({"product_id": rows["item_id"], "merchant_id": rows["seller_id"],
                             "category": rows["category"],
                             "published_at": rows["impression_ts"] - 86400})
    merchants = pd.DataFrame({"merchant_id": rows["seller_id"].unique(), "return_rate": 0.1,
                              "fulfilment_days": 2.0})
    imps = pd.DataFrame({"product_id": rows["item_id"], "ts": rows["impression_ts"] - 1})
    return HistoryStore.from_sources(imps, events, products, merchants)


# ---- synthetic pipeline fixtures (integration / load) -------------------------------------

@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    """Synthetic data dir with ranking_items.csv built through build -> clean -> validate."""
    import io

    from shopfeed.data.build import BuildReport, Sources, build_rows
    from shopfeed.data.clean import clean, read_raw
    from shopfeed.schema import validate
    from shopfeed.testing.synth import generate

    root = tmp_path_factory.mktemp("synth")
    data = root / "data"
    generate(data)
    report = BuildReport()
    built = build_rows(Sources.load(data), report)
    buf = io.StringIO()
    built.to_csv(buf, index=False)
    buf.seek(0)
    df, clean_report = clean(read_raw(buf))
    validate(df)
    df.to_csv(data / "ranking_items.csv", index=False)
    return {"root": root, "data": data, "build_report": report.counts,
            "clean_report": clean_report}


@pytest.fixture(scope="session")
def synth_dataset(synth_root):
    from shopfeed.training.dataset import load_dataset, load_store

    store = load_store(synth_root["data"])
    return load_dataset(synth_root["data"] / "ranking_items.csv", store)


@pytest.fixture(scope="session")
def synth_model(synth_root, synth_dataset):
    from shopfeed.training.train_lambdamart import run

    out = synth_root["root"] / "artifacts" / "ranker_v2"
    result = run(synth_dataset, out, n_boot=100)
    return {"dir": out, **result}
