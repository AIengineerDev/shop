"""Two-tower retrieval model.

User tower: batch user features (affinity, content profile, counts).
Item tower: content vector + category one-hot + log price + hashed merchant embedding.
Because the item tower uses only catalog content, a product with zero interactions gets a
usable embedding the moment it is published (cold-start path).

Loss: in-batch sampled softmax with logQ correction (Yi et al. 2019) so popular items are
not over-rewarded merely for appearing often as in-batch negatives.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from shopfeed.config import CATEGORIES, ModelConfig
from shopfeed.features.batch import AFF_COLS, CONTENT_COLS, PROFILE_COLS

N_MERCHANT_BUCKETS = 128
USER_COLS: List[str] = AFF_COLS + PROFILE_COLS + ["n_purchases", "n_views_7d", "shop_pay_history"]
_CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
POSITIVE_EVENTS = {"dwell": 1.0, "save": 2.0, "add_to_cart": 2.0, "purchase": 3.0}


def merchant_bucket(merchant_id: str) -> int:
    return zlib.crc32(merchant_id.encode()) % N_MERCHANT_BUCKETS


def user_matrix(batch_users: pd.DataFrame) -> np.ndarray:
    X = batch_users[USER_COLS].values.astype(np.float32)
    X[:, len(AFF_COLS) + len(PROFILE_COLS)] = np.log1p(X[:, len(AFF_COLS) + len(PROFILE_COLS)])
    X[:, len(AFF_COLS) + len(PROFILE_COLS) + 1] = np.log1p(X[:, len(AFF_COLS) + len(PROFILE_COLS) + 1])
    return np.nan_to_num(X)


def item_matrix(products: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    n = len(products)
    onehot = np.zeros((n, len(CATEGORIES)), dtype=np.float32)
    onehot[np.arange(n), products.category.map(_CAT_INDEX).fillna(0).astype(int).values] = 1.0
    X = np.concatenate([
        products[CONTENT_COLS].values.astype(np.float32),
        onehot,
        np.log(products.price.values.astype(np.float32))[:, None],
    ], axis=1)
    buckets = np.array([merchant_bucket(m) for m in products.merchant_id.values], dtype=np.int64)
    return np.nan_to_num(X), buckets


class TwoTower(nn.Module):
    def __init__(self, user_dim: int, item_dim: int, emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.user_net = nn.Sequential(nn.Linear(user_dim, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim))
        self.merchant_emb = nn.Embedding(N_MERCHANT_BUCKETS, 8)
        self.item_net = nn.Sequential(nn.Linear(item_dim + 8, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim))
        self.temperature = 0.1

    def user_embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_net(x), dim=-1)

    def item_embed(self, x: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, self.merchant_emb(bucket)], dim=-1)
        return F.normalize(self.item_net(z), dim=-1)


def in_batch_softmax_loss(u: torch.Tensor, v: torch.Tensor, log_q: Optional[torch.Tensor], temperature: float) -> torch.Tensor:
    logits = u @ v.T / temperature
    if log_q is not None:
        logits = logits - log_q[None, :]  # logQ correction: subtract log sampling prob of each in-batch item
    labels = torch.arange(len(u), device=u.device)
    return F.cross_entropy(logits, labels)


@dataclass
class RetrievalArtifacts:
    model: TwoTower
    item_ids: np.ndarray
    item_vecs: np.ndarray
    history: List[float] = field(default_factory=list)
    item_freq: Dict[str, float] = field(default_factory=dict)

    def embed_users(self, batch_users: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model.user_embed(torch.from_numpy(user_matrix(batch_users))).numpy()

    def embed_items(self, products: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X, b = item_matrix(products)
        with torch.no_grad():
            return self.model.item_embed(torch.from_numpy(X), torch.from_numpy(b)).numpy()


def positive_pairs(events: pd.DataFrame) -> pd.DataFrame:
    """(user_id, product_id, weight) from meaningful engagements; purchases weigh most."""
    ev = events[events.type.isin(POSITIVE_EVENTS) & events.product_id.notna()]
    w = ev.type.map(POSITIVE_EVENTS)
    return pd.DataFrame({"user_id": ev.user_id.values, "product_id": ev.product_id.values, "weight": w.values})


def train_two_tower(
    pairs: pd.DataFrame,
    batch_users: pd.DataFrame,
    products: pd.DataFrame,
    cfg: ModelConfig,
    log_q_correction: bool = True,
    steps: Optional[int] = None,
) -> RetrievalArtifacts:
    if len(pairs) == 0:
        raise ValueError("no positive pairs to train retrieval on")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    U = user_matrix(batch_users)
    uid_to_row = {u: i for i, u in enumerate(batch_users.user_id.values)}
    I, B = item_matrix(products)
    pid_to_row = {p: i for i, p in enumerate(products.product_id.values)}

    pairs = pairs[pairs.user_id.isin(uid_to_row) & pairs.product_id.isin(pid_to_row)]
    u_rows = pairs.user_id.map(uid_to_row).values
    i_rows = pairs.product_id.map(pid_to_row).values
    freq = pairs.product_id.value_counts(normalize=True)
    log_q_all = np.log(freq.reindex(products.product_id.values).fillna(freq.min()).values.astype(np.float32))

    model = TwoTower(U.shape[1], I.shape[1], cfg.emb_dim, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = len(pairs)
    bs = min(cfg.batch_size, n)
    total_steps = steps if steps is not None else max(1, cfg.epochs * (n // bs))
    history: List[float] = []
    Ut, It, Bt, Lq = map(torch.from_numpy, (U, I, B, log_q_all))
    model.train()
    for _ in range(total_steps):
        idx = rng.choice(n, size=bs, replace=False)
        # de-duplicate items within the batch so the diagonal label is unambiguous
        _, keep = np.unique(i_rows[idx], return_index=True)
        idx = idx[keep]
        u = model.user_embed(Ut[u_rows[idx]])
        v = model.item_embed(It[i_rows[idx]], Bt[i_rows[idx]])
        loss = in_batch_softmax_loss(u, v, Lq[i_rows[idx]] if log_q_correction else None, model.temperature)
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(loss.item())
    art = RetrievalArtifacts(model=model, item_ids=products.product_id.values.copy(), item_vecs=np.zeros((0, cfg.emb_dim)),
                             history=history, item_freq=freq.to_dict())
    art.item_vecs = art.embed_items(products)
    return art
