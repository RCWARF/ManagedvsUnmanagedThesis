"""Standard IR metrics on doc-level rankings (BEIR conventions: graded relevance, gain = rel)."""
import math


def dedupe_ranking(doc_ids: list[str]) -> list[str]:
    """Chunk hits -> doc ranking: keep each doc at its best (first) rank."""
    seen, out = set(), []
    for d in doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def ndcg_at_k(ranking: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranking: list[str], rels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    return len(relevant & set(ranking[:k])) / len(relevant) if relevant else 0.0


def mrr_at_k(ranking: list[str], rels: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranking[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], ks=(1, 10, 100)) -> dict:
    """run: qid -> ranked doc ids (already deduped). Means over queries that have qrels."""
    qids = [q for q in qrels if q in run]
    if not qids:
        return {"queries": 0}
    out = {"queries": len(qids), "ndcg@10": sum(ndcg_at_k(run[q], qrels[q], 10) for q in qids) / len(qids),
           "mrr@10": sum(mrr_at_k(run[q], qrels[q], 10) for q in qids) / len(qids)}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(run[q], qrels[q], k) for q in qids) / len(qids)
    return {k: round(v, 5) if isinstance(v, float) else v for k, v in out.items()}
