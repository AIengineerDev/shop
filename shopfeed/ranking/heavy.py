"""Heavy ranker: shared-bottom multi-task network with per-head gating (MMoE-lite).

One sigmoid head per action in config.HEADS. Loss is BCE per head with a positive-class
weight equal to neg/pos prevalence so rare heads (purchase, return) are not drowned out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shopfeed.config import HEADS, ModelConfig
from shopfeed.ranking.position import neutralize_position


class MultiTaskRanker(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_experts: int = 3, heads: List[str] = HEADS):
        super().__init__()
        self.heads = list(heads)
        self.norm = nn.BatchNorm1d(n_features)
        self.bottom = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU()) for _ in range(n_experts)])
        self.gates = nn.ModuleList([nn.Linear(hidden, n_experts) for _ in self.heads])
        self.towers = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)) for _ in self.heads])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bottom(self.norm(x))
        E = torch.stack([e(h) for e in self.experts], dim=1)  # (B, n_experts, hidden)
        outs = []
        for gate, tower in zip(self.gates, self.towers):
            g = F.softmax(gate(h), dim=-1).unsqueeze(-1)       # (B, n_experts, 1)
            outs.append(tower((E * g).sum(dim=1)))
        return torch.cat(outs, dim=1)  # logits (B, n_heads)


@dataclass
class RankerArtifacts:
    model: MultiTaskRanker
    version: str
    history: List[float] = field(default_factory=list)
    pos_weight: Dict[str, float] = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilities (n, n_heads) with position neutralised to the inference slot."""
        if len(X) == 0:
            return np.zeros((0, len(self.model.heads)), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(neutralize_position(X).astype(np.float32)))
        return torch.sigmoid(logits).numpy()


def train_ranker(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: ModelConfig,
    version: str = "ranker_v1",
    head_mask: Optional[np.ndarray] = None,
    steps: Optional[int] = None,
) -> RankerArtifacts:
    """Train on logged impressions. Y is (n, len(HEADS)) in {0,1}. `head_mask` zeros a head's loss
    (used in tests to show what happens when e.g. hide is ignored)."""
    if len(X) == 0:
        raise ValueError("empty training set: impression log has no labelled rows")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    Xt = torch.from_numpy(X.astype(np.float32))
    Yt = torch.from_numpy(Y.astype(np.float32))
    pos = Y.sum(axis=0)
    neg = len(Y) - pos
    pw = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0).astype(np.float32)
    pw = np.clip(pw, 1.0, 50.0)
    pos_weight = torch.from_numpy(pw)
    hm = torch.ones(len(HEADS)) if head_mask is None else torch.from_numpy(head_mask.astype(np.float32))

    model = MultiTaskRanker(X.shape[1], cfg.hidden, cfg.n_experts)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr * 0.3)
    n = len(X)
    bs = min(cfg.batch_size, n)
    total = steps if steps is not None else max(1, cfg.epochs * (n // bs))
    history: List[float] = []
    model.train()
    for _ in range(total):
        idx = rng.choice(n, size=bs, replace=False) if bs > 1 else np.array([0])
        logits = model(Xt[idx])
        per_head = F.binary_cross_entropy_with_logits(logits, Yt[idx], pos_weight=pos_weight, reduction="none").mean(dim=0)
        loss = (per_head * hm).sum() / hm.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(loss.item())
    return RankerArtifacts(model=model, version=version, history=history, pos_weight=dict(zip(HEADS, pw.tolist())))
