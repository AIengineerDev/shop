from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from shopfeed.config import Config
from shopfeed.ranking.heavy import MultiTaskRanker, RankerArtifacts, train_ranker
from shopfeed.ranking.light import LightRanker
from shopfeed.retrieval.two_tower import RetrievalArtifacts, TwoTower
from shopfeed.training.dataset import TrainingSet


def fit_rankers(ts: TrainingSet, train_idx: np.ndarray, cfg: Config, version: str = "ranker_v1", steps: int = None) -> Tuple[LightRanker, RankerArtifacts]:
    if len(train_idx) == 0:
        raise ValueError("empty training split")
    light = LightRanker().fit(ts.X[train_idx], ts.any_positive[train_idx])
    heavy = train_ranker(ts.X[train_idx], ts.Y[train_idx], cfg.model, version=version, steps=steps)
    return light, heavy


def save_artifacts(dir_: Path, retrieval: RetrievalArtifacts, light: LightRanker, heavy: RankerArtifacts) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    torch.save({"state": retrieval.model.state_dict(),
                "user_dim": retrieval.model.user_net[0].in_features,
                "item_dim": retrieval.model.item_net[0].in_features - 8,
                "emb_dim": retrieval.model.user_net[-1].out_features,
                "hidden": retrieval.model.user_net[0].out_features}, dir_ / "two_tower.pt")
    np.save(dir_ / "item_ids.npy", retrieval.item_ids.astype(str))
    np.save(dir_ / "item_vecs.npy", retrieval.item_vecs)
    np.savez(dir_ / "light.npz", **light.to_state())  # arrays, not a pickled estimator
    torch.save({"state": heavy.model.state_dict(), "n_features": heavy.model.norm.num_features,
                "hidden": heavy.model.bottom[0].out_features, "n_experts": len(heavy.model.experts)}, dir_ / "heavy.pt")
    with open(dir_ / "meta.json", "w") as f:
        json.dump({"version": heavy.version, "pos_weight": heavy.pos_weight,
                   "retrieval_final_loss": retrieval.history[-1] if retrieval.history else None,
                   "ranker_final_loss": heavy.history[-1] if heavy.history else None}, f, indent=2)


def load_artifacts(dir_: Path) -> Tuple[RetrievalArtifacts, LightRanker, RankerArtifacts]:
    tt = torch.load(dir_ / "two_tower.pt", weights_only=False)
    model = TwoTower(tt["user_dim"], tt["item_dim"], tt["emb_dim"], tt["hidden"])
    model.load_state_dict(tt["state"])
    retrieval = RetrievalArtifacts(model=model, item_ids=np.load(dir_ / "item_ids.npy", allow_pickle=True).astype(object),
                                   item_vecs=np.load(dir_ / "item_vecs.npy"))
    with np.load(dir_ / "light.npz", allow_pickle=False) as z:
        light = LightRanker.from_state({k: z[k] for k in z.files})
    hv = torch.load(dir_ / "heavy.pt", weights_only=False)
    hm = MultiTaskRanker(hv["n_features"], hv["hidden"], hv["n_experts"])
    hm.load_state_dict(hv["state"])
    meta = json.loads((dir_ / "meta.json").read_text())
    heavy = RankerArtifacts(model=hm, version=meta["version"], pos_weight=meta.get("pos_weight", {}))
    return retrieval, light, heavy
