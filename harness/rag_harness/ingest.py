"""Ingestion phase: burst-upload one S3 object per BEIR doc, wait for the queue to drain,
then verify that every expected chunk is in the index (completeness check)."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

from . import beir, manifest
from .aws import client
from .paths import CONFIGS, ROOT

sys.path.insert(0, str(CONFIGS))
sys.path.insert(0, str(ROOT / "lambdas" / "src"))
from rag.chunking import chunk_text  # noqa: E402  same code the Lambda runs
from rag_config import RunConfig  # noqa: E402


def _doc_text(doc: dict) -> str:
    title = (doc.get("title") or "").strip()
    body = (doc.get("text") or "").strip()
    return f"{title}\n\n{body}" if title else body


def expected_chunks(cfg: RunConfig, docs: list[dict]) -> int:
    return sum(len(chunk_text(_doc_text(d), cfg.chunking.size, cfg.chunking.overlap)) for d in docs)


def index_arn(out: dict) -> str:
    """Stacks deployed before IndexArn was an output expose the bucket ARN + index name."""
    return out.get("IndexArn") or f"{out['VectorBucket']}/index/{out['VectorIndex']}"


def count_vectors(arn: str) -> int:
    s3v = client("s3vectors")
    n, token = 0, None
    while True:
        kw = dict(indexArn=arn, maxResults=1000)
        if token:
            kw["nextToken"] = token
        resp = s3v.list_vectors(**kw)
        n += len(resp.get("vectors", []))
        token = resp.get("nextToken")
        if not token:
            return n


def _queue_depth(sqs, url: str) -> int:
    a = sqs.get_queue_attributes(QueueUrl=url, AttributeNames=[
        "ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
    return int(a["ApproximateNumberOfMessages"]) + int(a["ApproximateNumberOfMessagesNotVisible"])


def run(run_id: str) -> dict:
    m = manifest.load(run_id)
    cfg = RunConfig.model_validate(m["config"])
    out = m["outputs"]
    corpus = beir.local_dir(cfg.dataset.name) / "corpus.jsonl"
    if beir.sha256_file(corpus) != cfg.dataset.sha256:
        raise SystemExit("local corpus checksum differs from dataset.sha256 in the config")
    docs = list(beir.iter_corpus(corpus, cfg.dataset.doc_limit))
    n_expected = expected_chunks(cfg, docs)
    s3, sqs = client("s3"), client("sqs")

    def put(doc):
        key = f"{cfg.dataset.name}/{quote(str(doc['_id']), safe='')}.json"
        s3.put_object(Bucket=out["IngestBucket"], Key=key, Body=json.dumps(doc).encode(),
                      ContentType="application/json")

    manifest.update(run_id, phases={"ingest_upload_start": manifest.now()},
                    ingest_expected={"docs": len(docs), "chunks": n_expected})
    with ThreadPoolExecutor(max_workers=cfg.ingest.upload_parallelism) as pool:
        list(pool.map(put, docs))
    manifest.update(run_id, phases={"ingest_upload_end": manifest.now()})

    # Drained = queue empty on 3 consecutive polls. Exact end time comes from the logs in `collect`.
    deadline = time.time() + cfg.ingest.timeout_min * 60
    empty_polls = 0
    time.sleep(10)
    while empty_polls < 3:
        if time.time() > deadline:
            manifest.update(run_id, ingest_status="timeout")
            raise SystemExit("ingestion timed out; collect anyway to see where it stopped")
        empty_polls = empty_polls + 1 if _queue_depth(sqs, out["QueueUrl"]) == 0 else 0
        time.sleep(5)
    manifest.update(run_id, phases={"ingest_drained": manifest.now()})

    # The DLQ belongs to the shared stack, so count only what this run added.
    result = {"dlq_messages": _queue_depth(sqs, out["DlqUrl"]) - m.get("dlq_depth_before", 0)}
    if cfg.ingest.verify_completeness:
        stored = count_vectors(index_arn(out))
        result |= {"vectors_stored": stored, "vectors_expected": n_expected,
                   "completeness": round(stored / n_expected, 6) if n_expected else None}
    manifest.update(run_id, ingest_result=result, ingest_status="done")
    return result
