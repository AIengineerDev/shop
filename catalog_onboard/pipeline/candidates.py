"""Deterministic candidate shortlist: BM25 over leaf categories (name + ancestor path)."""

from __future__ import annotations

import math
import re
from collections import Counter

from catalog_onboard.models import Candidate
from catalog_onboard.taxonomy import Taxonomy

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset({"and", "the", "with", "for", "of", "a", "an", "in", "to"})
# Light stemming so "shorts" ~ "short", "knives" ~ "knife", "leggings" ~ "legging".
_IRREGULAR = {"knives": "knife", "dresses": "dress", "boxes": "box", "glasses": "glass"}
# Domain synonyms applied to both documents and queries. Small on purpose: the LLM reranks within top-K.
SYNONYMS: dict[str, str] = {
    "tee": "shirt",
    "tshirt": "shirt",
    "tshirts": "shirt",
    "sundress": "dress",
    "skillet": "pan",
    "saucepan": "pan",
    "earbud": "headphone",
    "earbuds": "headphone",
    "earphones": "headphone",
    "headphones": "headphone",
    "daypack": "backpack",
    "rucksack": "backpack",
    "basketball": "ball",
    "football": "ball",
    "volleyball": "ball",
    "soccer": "ball",
    "container": "storage",
    "containers": "storage",
    "tumbler": "mug",
    "sneaker": "shoe",
    "sneakers": "shoe",
    "trainers": "shoe",
    "sweatpants": "jogger",
    "sweatshirt": "hoodie",
    "hooded": "hoodie",
    "charger": "cable",
    "charging": "cable",
    "denim": "jean",
    "trunks": "short",
    "cutlery": "knife",
    "boys": "kid",
    "girls": "kid",
    "toddler": "kid",
    "baby": "kid",
    "ladies": "women",
}


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP or len(tok) < 2:
            continue
        tok = SYNONYMS.get(tok, tok)
        tok = _IRREGULAR.get(tok, tok)
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(SYNONYMS.get(tok, tok))
    return out


class CandidateIndex:
    def __init__(self, taxonomy: Taxonomy, *, k1: float = 1.5, b: float = 0.75):
        self.taxonomy = taxonomy
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = list(taxonomy.leaf_ids)
        self.docs: list[Counter[str]] = []
        self.doc_len: list[int] = []
        df: Counter[str] = Counter()
        for cid in self.doc_ids:
            # Leaf name counts double so that the leaf term dominates its ancestors.
            text = f"{taxonomy.get(cid).name} {taxonomy.get(cid).name} {taxonomy.path(cid)}"
            toks = tokenize(text)
            c = Counter(toks)
            self.docs.append(c)
            self.doc_len.append(len(toks))
            df.update(c.keys())
        n = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        doc = self.docs[i]
        dl = self.doc_len[i]
        s = 0.0
        for t in query_tokens:
            tf = doc.get(t)
            if not tf:
                continue
            idf = self.idf.get(t, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += idf * tf * (self.k1 + 1) / denom
        return s

    def shortlist(self, text: str, k: int) -> list[Candidate]:
        """Top-k leaves with a positive BM25 score. Empty when the text has no usable tokens or no overlap."""
        q = tokenize(text)
        if not q:
            return []
        scored = [(self.score(q, i), cid) for i, cid in enumerate(self.doc_ids)]
        scored = [(s, cid) for s, cid in scored if s > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [Candidate(category_id=cid, path=self.taxonomy.path(cid), score=round(s, 4)) for s, cid in scored[:k]]

    def all_leaves(self) -> list[Candidate]:
        """Fallback shortlist when lexical overlap is zero: every leaf, score 0, in id order."""
        return [Candidate(category_id=cid, path=self.taxonomy.path(cid), score=0.0) for cid in self.doc_ids]


def deterministic_confidence(cands: list[Candidate]) -> float:
    """s1/(s1+s2) in [0.5, 1]; 1.0 when there is a single candidate."""
    if not cands:
        return 0.0
    if len(cands) == 1:
        return 1.0
    s1, s2 = cands[0].score, cands[1].score
    return s1 / (s1 + s2) if (s1 + s2) > 0 else 0.0
