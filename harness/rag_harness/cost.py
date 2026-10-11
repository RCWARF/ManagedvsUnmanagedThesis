"""Per-run cost estimate from collected records (reconcile with Cost Explorer later by RunId tag).

Prices: AWS Price List API, us-west-2, retrieved 2026-10-11.
"""
import json

from . import manifest
from .paths import RAW

PRICES = {
    "retrieved": "2026-10-11",
    "titan_v2_per_1k_tokens": 0.00002,
    "nova_lite_in_per_1k": 0.00006,
    "nova_lite_out_per_1k": 0.00024,
    "lambda_arm_gb_s": 0.0000133334,
    "lambda_x86_gb_s": 0.0000166667,
    "lambda_request": 0.0000002,
    "sqs_fifo_request": 0.0000005,
    "s3v_put_per_gb": 0.20,
    "s3v_query_per_request": 0.0000025,
    "s3_put_request": 0.000005,
}


def _read(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def estimate(run_id: str) -> dict:
    m = manifest.load(run_id)
    cfg = m["config"]
    d = RAW / run_id
    embed = _read(d / "embed_saaf.jsonl")
    query = _read(d / "query_saaf.jsonl")
    c = {}
    embed_tokens = sum(r.get("embed_tokens", 0) for r in embed) + sum(r.get("embed_tokens", 0) for r in query)
    c["titan"] = embed_tokens / 1000 * PRICES["titan_v2_per_1k_tokens"]
    c["nova"] = (sum(r.get("gen_input_tokens", 0) for r in query) / 1000 * PRICES["nova_lite_in_per_1k"]
                 + sum(r.get("gen_output_tokens", 0) for r in query) / 1000 * PRICES["nova_lite_out_per_1k"])
    lam = 0.0
    n_inv = 0
    for role in ("enqueue", "embed", "query"):
        spec = cfg["lambdas"][role]
        rate = PRICES["lambda_arm_gb_s"] if spec["arch"] == "arm64" else PRICES["lambda_x86_gb_s"]
        for r in _read(d / f"{role}_reports.jsonl"):
            lam += int(r["billed_ms"]) / 1000 * int(r["memory_mb"]) / 1024 * rate
            n_inv += 1
    c["lambda"] = lam + n_inv * PRICES["lambda_request"]
    chunks = sum(r.get("chunks", 0) for r in embed)
    avg_chars = (sum(r.get("chars", 0) for r in embed) / chunks) if chunks else 0
    vec_bytes = cfg["embedding"]["dimensions"] * 4 + min(avg_chars, cfg["chunking"]["size"]) + 200
    c["s3v_put"] = chunks * vec_bytes / 1e9 * PRICES["s3v_put_per_gb"]
    c["s3v_query"] = len(query) * PRICES["s3v_query_per_request"]
    docs = m.get("ingest_expected", {}).get("docs", 0)
    c["sqs"] = docs * 3 * PRICES["sqs_fifo_request"]  # send + receive + delete per doc
    c["s3_put"] = docs * PRICES["s3_put_request"]
    c = {k: round(v, 6) for k, v in c.items()}
    c["total"] = round(sum(c.values()), 4)
    c["prices"] = PRICES
    c["excluded"] = "CloudWatch Logs, VPC endpoints (network stack), S3 Vectors storage"
    manifest.update(run_id, cost_estimate_usd=c)
    return c
