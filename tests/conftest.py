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
