"""Closed-loop query load: N workers each send one request at a time to the query Lambda.

Built in so a run is reproducible from the repo alone. Queries are BEIR test queries
(never one repeated query), sampled with a fixed seed. Client-side records go to
results/raw/<run_id>/client_queries.jsonl; server-side SAAF records come from `collect`.
"""
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from botocore.config import Config
import boto3

from . import beir, manifest
from .paths import CONFIGS, RAW, REGION

sys.path.insert(0, str(CONFIGS))
from rag_config import RunConfig  # noqa: E402


def _sample(cfg: RunConfig) -> list[tuple[str, str]]:
    d = beir.local_dir(cfg.dataset.name)
    queries = beir.load_queries(d / "queries.jsonl")
    qrels = beir.load_qrels(d / "qrels" / f"{cfg.dataset.split}.tsv")
    pool = sorted((qid, queries[qid]) for qid in qrels if qid in queries)
    rng = random.Random(cfg.query.seed)
    total = cfg.query.warmup_requests + cfg.query.requests
    if cfg.query.sampling == "with_replacement":
        return [rng.choice(pool) for _ in range(total)]
    out = []
    while len(out) < total:  # reshuffle passes when requests exceed the query set
        batch = pool[:]
        rng.shuffle(batch)
        out.extend(batch)
    return out[:total]


def run(run_id: str) -> dict:
    m = manifest.load(run_id)
    cfg = RunConfig.model_validate(m["config"])
    fn = m["outputs"]["QueryFn"]
    lam = boto3.client("lambda", config=Config(region_name=REGION, retries={"max_attempts": 1},
                                               read_timeout=cfg.lambdas.query.timeout_s + 10,
                                               max_pool_connections=cfg.query.concurrency + 4))
    work = _sample(cfg)
    out_dir = RAW / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    sink = (out_dir / "client_queries.jsonl").open("w")
    idx = iter(enumerate(work))

    def worker():
        while True:
            with lock:
                try:
                    i, (qid, text) = next(idx)
                except StopIteration:
                    return
            t0 = time.time()
            p0 = time.perf_counter()
            try:
                r = lam.invoke(FunctionName=fn, Payload=json.dumps({"qid": qid, "question": text}).encode())
                payload = r["Payload"].read()
                err = r.get("FunctionError")
                status = "error" if err else "ok"
                detail = payload[:300].decode(errors="ignore") if err else None
            except Exception as exc:  # noqa: BLE001 - client-side failure is data
                status, detail = "client_error", f"{type(exc).__name__}: {exc}"[:300]
            rec = {"i": i, "qid": qid, "warmup": i < cfg.query.warmup_requests, "t_start": t0,
                   "rtt_ms": round((time.perf_counter() - p0) * 1000.0, 2), "status": status, "detail": detail}
            with lock:
                sink.write(json.dumps(rec) + "\n")

    manifest.update(run_id, phases={"query_start": manifest.now()})
    with ThreadPoolExecutor(max_workers=cfg.query.concurrency) as pool:
        for _ in range(cfg.query.concurrency):
            pool.submit(worker)
    sink.close()
    manifest.update(run_id, phases={"query_end": manifest.now()})
    recs = [json.loads(line) for line in (out_dir / "client_queries.jsonl").open()]
    summary = {"requests": len(recs), "errors": sum(r["status"] != "ok" for r in recs)}
    manifest.update(run_id, query_result=summary)
    return summary
