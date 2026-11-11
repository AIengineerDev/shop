from catalog_onboard.pipeline.candidates import CandidateIndex, deterministic_confidence, tokenize
from catalog_onboard.taxonomy import Taxonomy


def test_tokenize_synonyms_and_stem():
    assert tokenize("Men's Fitted Tee") == ["men", "fitted", "shirt"]
    assert tokenize("Chef Knives Set") == ["chef", "knife", "set"]
    assert tokenize("Boys Toddler Tee") == ["kid", "kid", "shirt"]


def test_sample_ranks_shorts_first(taxonomy: Taxonomy):
    idx = CandidateIndex(taxonomy)
    cands = idx.shortlist("Men Dry-Fit Athletic Gym Shorts With Pockets Moisture wicking fabric", 8)
    assert cands[0].category_id == "apparel.men.bottoms.shorts"
    assert len(cands) <= 8
    assert all(taxonomy.is_leaf(c.category_id) for c in cands)


def test_no_overlap_returns_empty(taxonomy: Taxonomy):
    idx = CandidateIndex(taxonomy)
    assert idx.shortlist("zzzz qqqq", 8) == []
    assert idx.shortlist("", 8) == []
    assert len(idx.all_leaves()) == len(taxonomy.leaf_ids)


def test_deterministic_confidence(taxonomy: Taxonomy):
    idx = CandidateIndex(taxonomy)
    c = idx.shortlist("Cast Iron Skillet", 8)
    assert 0.5 <= deterministic_confidence(c) <= 1.0
    assert deterministic_confidence([]) == 0.0
    assert deterministic_confidence(c[:1]) == 1.0
