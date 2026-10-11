import math

from eval.ir_metrics import dedupe_ranking, evaluate, mrr_at_k, ndcg_at_k, recall_at_k
from rag.chunking import chunk_text


def test_chunking_windows_and_overlap():
    text = "abcdefghij" * 25  # 250 chars
    chunks = chunk_text(text, 100, 20)
    assert chunks[0] == text[:100]
    assert chunks[1] == text[80:180]
    assert "".join(c[:80] for c in chunks[:-1]) + chunks[-1] == text


def test_chunking_drops_blank():
    assert chunk_text("   ", 100, 10) == []


def test_ir_metrics():
    rels = {"d1": 1, "d2": 1}
    assert ndcg_at_k(["d1", "d2", "x"], rels) == 1.0
    assert math.isclose(ndcg_at_k(["x", "d1"], {"d1": 1}), 1 / math.log2(3))
    assert recall_at_k(["d1", "x"], rels, 10) == 0.5
    assert mrr_at_k(["x", "y", "d2"], rels) == 1 / 3
    assert dedupe_ranking(["a", "b", "a", "c"]) == ["a", "b", "c"]
    out = evaluate({"q1": ["d1"], "q2": ["x"]}, {"q1": {"d1": 1}, "q2": {"d9": 1}})
    assert out["queries"] == 2 and out["ndcg@10"] == 0.5
