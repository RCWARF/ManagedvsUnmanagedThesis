"""Retrieval-quality control: every BEIR test query against the run's index, top-100 chunks.

Runs from the harness (not through the query Lambda) so it never mixes with load data.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from . import beir, manifest
from .aws import client
from .paths import RAW, ROOT

sys.path.insert(0, str(ROOT))
from eval.ir_metrics import dedupe_ranking, evaluate  # noqa: E402


def run(run_id: str, top_k: int = 100, workers: int = 8) -> dict:
    m = manifest.load(run_id)
    cfg, out = m["config"], m["outputs"]
    d = beir.local_dir(cfg["dataset"]["name"])
    queries = beir.load_queries(d / "queries.jsonl")
    qrels = beir.load_qrels(d / "qrels" / f"{cfg['dataset']['split']}.tsv")
    br, s3v = client("bedrock-runtime"), client("s3vectors")

    def one(qid):
        body = json.dumps({"inputText": queries[qid], "dimensions": cfg["embedding"]["dimensions"],
                           "normalize": cfg["embedding"]["normalize"]})
        emb = json.loads(br.invoke_model(modelId=out["EmbedModel"], body=body)["body"].read())["embedding"]
        resp = s3v.query_vectors(vectorBucketName=out["VectorBucket"], indexName=out["VectorIndex"],
                                 topK=top_k, queryVector={"float32": emb}, returnMetadata=True)
        return qid, dedupe_ranking([(v.get("metadata") or {}).get("doc_id", "") for v in resp["vectors"]])

    qids = [q for q in qrels if q in queries]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ranking = dict(pool.map(one, qids))
    (RAW / run_id).mkdir(parents=True, exist_ok=True)
    (RAW / run_id / "retrieval_run.json").write_text(json.dumps(ranking))
    metrics = evaluate(ranking, qrels)
    manifest.update(run_id, retrieval_quality=metrics)
    return metrics
