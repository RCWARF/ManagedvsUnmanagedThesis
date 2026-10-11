"""SQS FIFO -> read doc from S3 -> chunk -> embed (thread fan-out) -> vector store.

Per-stage wall times are recorded so the thesis can attribute time to S3, chunking,
embedding and the vector store, and every throttle is counted explicitly.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from rag.chunking import chunk_text
from rag.clients import s3
from rag.embedder import embed
from rag.metrics import emit, summarize
from rag.settings import SETTINGS
from rag.stores.base import Chunk, get_store
from saaf.Inspector import Inspector

STORE = get_store(SETTINGS.vector_store)
POOL = ThreadPoolExecutor(max_workers=SETTINGS.embed_workers)


def _doc_text(raw: bytes) -> tuple[str, str]:
    """BEIR corpus line {"_id", "title", "text"} -> (doc_id, text). Plain text falls back to the key."""
    try:
        doc = json.loads(raw)
        title = (doc.get("title") or "").strip()
        body = (doc.get("text") or "").strip()
        return str(doc["_id"]), f"{title}\n\n{body}" if title else body
    except (ValueError, KeyError):
        return "", raw.decode("utf-8", errors="ignore")


def process_message(msg: dict, acc: dict) -> None:
    bucket, key = msg["bucket"], msg["key"]
    t_file = time.perf_counter()

    t0 = time.perf_counter()
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    acc["s3_read_ms"] += (time.perf_counter() - t0) * 1000.0

    doc_id, text = _doc_text(raw)
    doc_id = doc_id or key

    t0 = time.perf_counter()
    pieces = chunk_text(text, SETTINGS.chunk_size, SETTINGS.chunk_overlap)
    acc["chunk_ms"] += (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    results = list(POOL.map(embed, pieces))  # preserves chunk order
    acc["embed_stage_ms"] += (time.perf_counter() - t0) * 1000.0

    chunks = [
        Chunk(key=f"{doc_id}#{i:05d}", embedding=r.embedding, doc_id=doc_id,
              chunk_index=i, content=p, source_key=key)
        for i, (p, r) in enumerate(zip(pieces, results))
    ]
    t0 = time.perf_counter()
    for start in range(0, len(chunks), SETTINGS.put_batch_size):
        tb = time.perf_counter()
        STORE.put(chunks[start:start + SETTINGS.put_batch_size])
        acc["put_call_ms"].append((time.perf_counter() - tb) * 1000.0)
    acc["put_stage_ms"] += (time.perf_counter() - t0) * 1000.0

    acc["docs"] += 1
    acc["chunks"] += len(chunks)
    acc["chars"] += len(text)
    acc["embed_call_ms"].extend(r.latency_ms for r in results)
    acc["embed_tokens"] += sum(r.input_tokens for r in results)
    acc["embed_throttles"] += sum(r.throttles for r in results)
    acc["embed_other_errors"] += sum(r.other_errors for r in results)
    acc["doc_ms"].append((time.perf_counter() - t_file) * 1000.0)


def lambda_handler(event, context):
    inspector = Inspector()
    inspector.inspectAll()
    records = event.get("Records", [])
    acc = {
        "docs": 0, "chunks": 0, "chars": 0, "embed_tokens": 0,
        "embed_throttles": 0, "embed_other_errors": 0,
        "s3_read_ms": 0.0, "chunk_ms": 0.0, "embed_stage_ms": 0.0, "put_stage_ms": 0.0,
        "embed_call_ms": [], "put_call_ms": [], "doc_ms": [],
    }
    failures = []
    t_inv = time.perf_counter()
    try:
        for rec in records:
            try:
                process_message(json.loads(rec["body"]), acc)
            except Exception as exc:  # noqa: BLE001 - reported per message, SQS retries it
                failures.append({"itemIdentifier": rec["messageId"]})
                inspector.addAttribute("last_error", f"{type(exc).__name__}: {exc}"[:500])
        inspector.addAttribute("num_sqs_records", len(records))
        inspector.addAttribute("sqs_receive_count_max",
                               max((int(r.get("attributes", {}).get("ApproximateReceiveCount", 1)) for r in records), default=0))
        inspector.addAttribute("invocation_ms", round((time.perf_counter() - t_inv) * 1000.0, 2))
        for k in ("docs", "chunks", "chars", "embed_tokens", "embed_throttles", "embed_other_errors"):
            inspector.addAttribute(k, acc[k])
        for k in ("s3_read_ms", "chunk_ms", "embed_stage_ms", "put_stage_ms"):
            inspector.addAttribute(k, round(acc[k], 2))
        for name, values in (("embed_call_ms", acc["embed_call_ms"]), ("put_call_ms", acc["put_call_ms"]), ("doc_ms", acc["doc_ms"])):
            for k, v in summarize(name, values).items():
                inspector.addAttribute(k, v)
        inspector.addAttribute("embed_workers", SETTINGS.embed_workers)
        inspector.addAttribute("embed_dims", SETTINGS.embed_dims)
        inspector.addAttribute("messages_failed", len(failures))
        inspector.addAttribute("status", "partial_failure" if failures else "success")
    finally:
        inspector.inspectAllDeltas()
        emit(inspector, SETTINGS, "embed")
    return {"batchItemFailures": failures}
