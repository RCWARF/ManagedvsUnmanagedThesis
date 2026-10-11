import json

import boto3
import botocore
from statistics import mean


def pct(values, p):
    if not values:
        return None
    v = sorted(values)
    k = int(round((p / 100.0) * (len(v) - 1)))
    return v[k]


def summarize(prefix: str, values: list[float]) -> dict:
    if not values:
        return {}
    return {
        f"{prefix}_n": len(values),
        f"{prefix}_avg": round(mean(values), 2),
        f"{prefix}_p50": round(pct(values, 50), 2),
        f"{prefix}_p95": round(pct(values, 95), 2),
        f"{prefix}_max": round(max(values), 2),
    }


def emit(inspector, settings, stage: str) -> dict:
    """Finish SAAF and print one JSON line tagged for `rag collect`."""
    inspector.addAttribute("rag_metric", 1)
    inspector.addAttribute("stage", stage)
    inspector.addAttribute("run_id", settings.run_id)
    inspector.addAttribute("config_hash", settings.config_hash)
    inspector.addAttribute("git_sha", settings.git_sha)
    inspector.addAttribute("boto3_version", boto3.__version__)
    inspector.addAttribute("botocore_version", botocore.__version__)
    out = inspector.finish()
    print(json.dumps(out, default=str))
    return out
