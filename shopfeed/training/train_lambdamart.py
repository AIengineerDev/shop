"""LambdaMART v1 training (spec §6.2, §7) -> artifacts/ranker_v2/.

Artefacts (§7.4): model.txt, features.json, propensities.json, metrics.json, meta.json,
importance.csv. Group filtering (§7.2): groups with < 2 rows or constant grades are
dropped from ranking train / validation. Test is never filtered and is read once.

Position debiasing (M6): pass `position` to `fit`; LightGBM >= 4.1 learns a position bias
term jointly (unbiased LambdaMART) when a Dataset has `position` set. There is no
`lambdarank_unbiased` parameter in LightGBM 4.x; spec §7.3 is corrected accordingly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from shopfeed import config
from shopfeed.eval.metrics import mean_ipw_ndcg, per_query
from shopfeed.eval.propensity import estimate_from_randomised, lookup
from shopfeed.eval.offline import evaluate, paired_delta
from shopfeed.features.history import UNAVAILABLE
from shopfeed.features.ranking import CATEGORICAL, EXCLUDED, WITHIN_GROUP
from shopfeed.training.baselines import heuristic_scores

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10, 5, 20],  # NDCG@10 first: early stopping uses the first metric only
    "label_gain": config.GAIN_PURCHASE_DOMINANT,
    "lambdarank_truncation_level": 20,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": config.SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 100
GATES = {"ndcg@10_rel": 0.02, "purchase_mrr_rel": 0.03}


def filter_groups(rows: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = rows.groupby("qid")["relevance_grade"]
    keep = (g.transform("size") >= 2) & (g.transform("nunique") > 1)
    return rows[keep.to_numpy()].reset_index(drop=True), X[keep.to_numpy()].reset_index(drop=True)


def group_sizes(rows: pd.DataFrame) -> np.ndarray:
    if not rows["qid"].is_monotonic_increasing:
        raise ValueError("rows must be sorted by qid for LightGBM groups")
    return rows.groupby("qid", sort=True).size().to_numpy()


def fit(tr_rows, tr_X, va_rows, va_X, params=None, num_boost_round=NUM_BOOST_ROUND,
        early_stopping=EARLY_STOPPING, position: bool = False) -> lgb.Booster:
    params = {**PARAMS, **(params or {})}
    pos_tr = tr_rows["displayed_pos"].to_numpy() - 1 if position else None
    train = lgb.Dataset(tr_X, tr_rows["relevance_grade"], group=group_sizes(tr_rows),
                        position=pos_tr, free_raw_data=False)
    val = lgb.Dataset(va_X, va_rows["relevance_grade"], group=group_sizes(va_rows),
                      reference=train)
    callbacks = [lgb.early_stopping(early_stopping, first_metric_only=True, verbose=False)]
    return lgb.train(params, train, num_boost_round=num_boost_round, valid_sets=[val],
                     valid_names=["val"], callbacks=callbacks)


def predict(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(X, num_iteration=booster.best_iteration or None)


def importance(booster: lgb.Booster) -> pd.DataFrame:
    gain = booster.feature_importance("gain")
    split = booster.feature_importance("split")
    df = pd.DataFrame({"feature": booster.feature_name(), "gain": gain, "split": split})
    df["gain_share"] = df["gain"] / max(df["gain"].sum(), 1e-12)
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


def evaluate_test(te_rows: pd.DataFrame, model_scores: np.ndarray,
                  n_boot: int = 1000) -> dict:
    scores = heuristic_scores(te_rows)[["logged", "semantic"]]
    scores["lambdamart"] = model_scores
    df = pd.concat([te_rows, scores], axis=1)
    res = evaluate(df, list(scores.columns), reference="logged", n_boot=n_boot)
    pq_model, pq_sem = per_query(df, "lambdamart"), per_query(df, "semantic")
    vs_sem = [paired_delta(pq_model, pq_sem, m, n_boot)
              for m in ("ndcg@10", "purchase_mrr", "purchase@10")]
    d = res["deltas"].set_index(["scorer", "metric"]).loc["lambdamart"]
    o = res["overall"].set_index(["scorer", "metric"])["value"]
    gates = {
        "ndcg@10_rel_vs_logged": {"value": d.loc["ndcg@10", "rel_delta"],
                                  "ci_lo": d.loc["ndcg@10", "rel_lo"],
                                  "pass": bool(d.loc["ndcg@10", "rel_delta"] >= GATES["ndcg@10_rel"]
                                               and d.loc["ndcg@10", "delta_lo"] > 0)},
        "purchase_mrr_rel_vs_logged": {
            "value": d.loc["purchase_mrr", "rel_delta"],
            "pass": bool(d.loc["purchase_mrr", "rel_delta"] >= GATES["purchase_mrr_rel"])},
        "purchase@10_vs_logged": {
            "value": o.loc[("lambdamart", "purchase@10")] - o.loc[("logged", "purchase@10")],
            "pass": bool(o.loc[("lambdamart", "purchase@10")] >= o.loc[("logged", "purchase@10")])},
    }
    return {"overall": res["overall"], "deltas": res["deltas"], "slices": res["slices"],
            "vs_semantic": pd.DataFrame(vs_sem), "gates": gates}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def features_spec(names: list[str]) -> dict:
    return {
        "features": names,
        "dtype": "float64",
        "categorical_vocab": CATEGORICAL,
        "within_group": {"columns": WITHIN_GROUP, "transforms": ["rank", "z", "ratio_to_max"],
                         "group_key": "qid"},
        "excluded": sorted(EXCLUDED),
        "unavailable": UNAVAILABLE,
        "notes": {
            "semantic_sim": "USER-ITEM proxy: cosine(user t0..t7, product c0..c7); "
                            "replace with query-item similarity when search logs exist",
            "bm25": "NaN in the feed log",
            "seller_return_rate": "merchants.parquet snapshot; slow-moving, accepted for v1",
            "seller_fulfilment_days": "merchants.parquet snapshot; slow-moving, accepted for v1",
        },
    }


def write_artifacts(out: Path, booster: lgb.Booster, names: list[str], metrics: dict,
                    meta: dict, propensities: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"), num_iteration=booster.best_iteration or None)
    (out / "features.json").write_text(json.dumps(features_spec(names), indent=2))
    (out / "propensities.json").write_text(json.dumps(
        propensities or {"method": "none", "note": "position debiasing lands in M6"}, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    importance(booster).to_csv(out / "importance.csv", index=False)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def round_metrics(d: dict, nd: int = 6) -> dict:
    """Round floats so reproducibility can be compared exactly at the stored precision."""
    def r(x):
        if isinstance(x, float):
            return None if np.isnan(x) else round(x, nd)
        if isinstance(x, dict):
            return {k: r(v) for k, v in x.items()}
        if isinstance(x, list):
            return [r(v) for v in x]
        return x
    return r(d)


def run(ds, out: Path, params: dict | None = None, position: bool = False,
        n_boot: int = 1000, num_boost_round: int = NUM_BOOST_ROUND,
        early_stopping: int = EARLY_STOPPING, propensities: dict | None = None) -> dict:
    """Train on ds train/val, evaluate once on test, write the six §7.4 artefacts."""
    tr_rows, tr_X = filter_groups(*ds.part("train"))
    va_rows, va_X = filter_groups(*ds.part("val"))
    te_rows, te_X = ds.part("test")
    booster = fit(tr_rows, tr_X, va_rows, va_X, params=params, position=position,
                  num_boost_round=num_boost_round, early_stopping=early_stopping)
    scores = predict(booster, te_X)
    res = evaluate_test(te_rows, scores, n_boot=n_boot)
    imp = importance(booster)
    if propensities is None:
        propensities = estimate_from_randomised(ds.part("train")[0])
    te = te_rows.assign(lambdamart=scores, propensity_exam=lookup(propensities,
                                                                  te_rows["displayed_pos"]))
    te = te.assign(**heuristic_scores(te_rows)[["logged", "semantic"]])
    ipw = {s: mean_ipw_ndcg(te, s) for s in ("lambdamart", "logged", "semantic")}
    metrics = round_metrics({
        "best_iteration": booster.best_iteration,
        "val": {k: float(v) for k, v in booster.best_score["val"].items()},
        "test": {
            "overall": res["overall"].to_dict(orient="records"),
            "deltas_vs_logged": res["deltas"].to_dict(orient="records"),
            "deltas_vs_semantic": res["vs_semantic"].to_dict(orient="records"),
            "slices": res["slices"][res["slices"]["scorer"] == "lambdamart"]
            .to_dict(orient="records"),
        },
        "gates": res["gates"],
        "ipw_ndcg@10": ipw,
        "randomised_slice_note": "the whole log is randomised traffic, so raw test NDCG is "
                                 "the §8.3 randomised-slice number",
        "leakage_check": {
            "max_gain_share": float(imp["gain_share"].max()),
            "top_feature": imp["feature"].iloc[0],
            "flag": bool(imp["gain_share"].max() > 0.5),
        },
    })
    meta = {
        "git_sha": git_sha(),
        "seed": config.SEED,
        "data_window": {"min_ts": float(ds.rows["impression_ts"].min()),
                        "max_ts": float(ds.rows["impression_ts"].max())},
        "split": ds.split.to_dict(),
        "rows_after_group_filter": {"train": len(tr_rows), "val": len(va_rows),
                                    "test": len(te_rows)},
        "groups_after_group_filter": {"train": int(tr_rows["qid"].nunique()),
                                      "val": int(va_rows["qid"].nunique()),
                                      "test": int(te_rows["qid"].nunique())},
        "label_gain": {**PARAMS, **(params or {})}["label_gain"],
        "params": {**PARAMS, **(params or {})},
        "position_debias": position,
        "attribution": {"stamped_max_lag_s": config.STAMPED_MAX_LAG_S,
                        "reconstruct_window_s": config.ATTRIBUTION_WINDOW_S},
    }
    write_artifacts(out, booster, ds.feature_names, metrics, meta, propensities)
    return {"metrics": metrics, "meta": meta, "result": res, "booster": booster,
            "importance": imp}
