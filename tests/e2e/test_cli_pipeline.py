"""End-to-end: every CLI, in operator order, as subprocesses on a synthetic data root.

generate -> build_user_aov -> build_ranking_dataset -> check_ranking_data -> baselines ->
train -> sweep -> bench_serving. Asserts exit codes, artefacts and the headline results.
"""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from shopfeed import config

pytestmark = pytest.mark.e2e
SCRIPTS = config.ROOT / "scripts"


@pytest.fixture(scope="module")
def env_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    env = {**os.environ, "SHOPFEED_DATA_DIR": str(root / "data"),
           "SHOPFEED_ARTIFACTS_DIR": str(root / "artifacts"),
           "SHOPFEED_REPORTS_DIR": str(root / "reports")}
    return root, env


def _run(env, *args, expect=0) -> subprocess.CompletedProcess:
    p = subprocess.run([sys.executable, *map(str, args)], env=env, capture_output=True,
                       text=True, timeout=600, cwd=config.ROOT)
    detail = f"{args}\nstdout:\n{p.stdout[-3000:]}\nstderr:\n{p.stderr[-3000:]}"
    assert p.returncode == expect, detail
    return p


@pytest.fixture(scope="module")
def pipeline(env_root):
    """Run the whole pipeline once; individual tests assert on its outputs."""
    root, env = env_root
    out = {}
    out["generate"] = _run(env, "-m", "shopfeed.testing.synth", root / "data")
    out["aov"] = _run(env, SCRIPTS / "build_user_aov.py")
    out["aov_rerun"] = _run(env, SCRIPTS / "build_user_aov.py", expect=1)
    out["build"] = _run(env, SCRIPTS / "build_ranking_dataset.py")
    out["check"] = _run(env, SCRIPTS / "check_ranking_data.py", "--profile", "ci")
    out["check_prod"] = _run(env, SCRIPTS / "check_ranking_data.py", expect=1)
    out["baselines"] = _run(env, SCRIPTS / "baselines.py")
    out["train"] = _run(env, SCRIPTS / "train.py", "--n-boot", "100")
    out["sweep"] = _run(env, SCRIPTS / "sweep.py", "--trials", "2")
    out["bench"] = _run(env, SCRIPTS / "bench_serving.py", "--requests", "50",
                        "--candidates", "200")
    return root, out


def test_data_stage(pipeline):
    root, out = pipeline
    users = pd.read_parquet(root / "data" / "users.parquet")
    assert {"aov_tier", "aov_value", "aov_order_count", "aov_asof_ts"} <= set(users.columns)
    assert (root / "data" / "users.pre_aov.parquet").exists()
    assert "refusing" in out["aov_rerun"].stderr
    clean = json.loads((root / "reports" / "clean_report.json").read_text())
    assert clean["failed_rules"] == [] and clean["rows_out"] > 10_000
    assert "FAIL  scale" in out["check_prod"].stdout  # production gates reject the sample
    readiness = json.loads((root / "reports" / "readiness_report.json").read_text())
    assert readiness["profile"] == "production"


def test_baselines_stage(pipeline):
    root, _ = pipeline
    b = json.loads((root / "reports" / "baselines.json").read_text())
    ndcg = {r["scorer"]: r["value"] for r in b["overall"] if r["metric"] == "ndcg@10"}
    assert ndcg["semantic"] > ndcg["logged"]
    assert (root / "reports" / "baselines.html").exists()


def test_training_and_sweep_stage(pipeline):
    root, _ = pipeline
    art = root / "artifacts" / "ranker_v2"
    assert {p.name for p in art.iterdir()} >= {"model.txt", "features.json",
                                               "propensities.json", "metrics.json",
                                               "meta.json", "importance.csv"}
    m = json.loads((art / "metrics.json").read_text())
    assert all(g["pass"] for g in m["gates"].values()), m["gates"]
    sweep = json.loads((root / "reports" / "sweep.json").read_text())
    assert sweep["trials"] == 6 and sweep["chosen"]["label_gain"][4] in (10, 20, 40)
    assert (root / "reports" / "ranker_v2_test.html").exists()


def test_serving_stage(pipeline):
    root, _ = pipeline
    bench = json.loads((root / "reports" / "serving_bench_200.json").read_text())
    assert bench["fallbacks"] == 0
    assert bench["total_p99_ms"] < config.RANK_TIMEOUT_S * 1000
