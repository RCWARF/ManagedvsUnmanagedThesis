"""Pull raw SAAF records, Lambda REPORT lines and CloudWatch metrics for one run.

Everything lands in results/raw/<run_id>/ as JSONL (and parquet when pandas+pyarrow
are installed). Timestamps stay as epoch milliseconds: no spreadsheet round-trips.
"""
import json
import re
from datetime import datetime, timedelta, timezone

from . import manifest
from .aws import client
from .paths import RAW

REPORT_RE = re.compile(
    r"REPORT RequestId: (?P<request_id>\S+)\s+Duration: (?P<duration_ms>[\d.]+) ms\s+"
    r"Billed Duration: (?P<billed_ms>\d+) ms\s+Memory Size: (?P<memory_mb>\d+) MB\s+"
    r"Max Memory Used: (?P<max_mem_mb>\d+) MB(?:\s+Init Duration: (?P<init_ms>[\d.]+) ms)?")


def _events(group: str, start_ms: int, pattern: str):
    logs = client("logs")
    kw = dict(logGroupName=group, startTime=start_ms, filterPattern=pattern)
    try:
        for page in logs.get_paginator("filter_log_events").paginate(**kw):
            yield from page.get("events", [])
    except logs.exceptions.ResourceNotFoundException:
        return  # function never invoked, so its log group was never created


def _metrics(fn: str, start: datetime, end: datetime) -> list[dict]:
    cw = client("cloudwatch")
    queries = [
        {"Id": f"m{i}", "ReturnData": True, "MetricStat": {
            "Metric": {"Namespace": "AWS/Lambda", "MetricName": name,
                       "Dimensions": [{"Name": "FunctionName", "Value": fn}]},
            "Period": 60, "Stat": stat}}
        for i, (name, stat) in enumerate([("Invocations", "Sum"), ("Throttles", "Sum"), ("Errors", "Sum"),
                                          ("ConcurrentExecutions", "Maximum"), ("Duration", "Average")])
    ]
    resp = cw.get_metric_data(MetricDataQueries=queries, StartTime=start, EndTime=end)
    names = ["Invocations", "Throttles", "Errors", "ConcurrentExecutions", "Duration"]
    rows = []
    for i, res in enumerate(resp["MetricDataResults"]):
        for ts, val in zip(res["Timestamps"], res["Values"]):
            rows.append({"function": fn, "metric": names[i], "ts": ts.isoformat(), "value": val})
    return rows


def _write(path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    try:
        import pandas as pd
        if rows:
            pd.json_normalize(rows).to_parquet(path.with_suffix(".parquet"), index=False)
    except ImportError:
        pass


def run(run_id: str) -> dict:
    m = manifest.load(run_id)
    out_dir = RAW / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.fromisoformat(m["phases"]["deploy_start"])
    start_ms = int(start.timestamp() * 1000)
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    counts = {}
    fns = {"enqueue": m["outputs"]["EnqueueFn"], "embed": m["outputs"]["EmbedFn"], "query": m["outputs"]["QueryFn"]}
    for role, fn_name in fns.items():
        group = f"/aws/lambda/{fn_name}"
        saaf, reports = [], []
        for ev in _events(group, start_ms, '"rag_metric"'):
            try:
                rec = json.loads(ev["message"])
            except ValueError:
                continue
            rec["log_ts_ms"] = ev["timestamp"]
            saaf.append(rec)
        for ev in _events(group, start_ms, '"REPORT RequestId"'):
            mt = REPORT_RE.search(ev["message"])
            if mt:
                reports.append({**{k: v for k, v in mt.groupdict().items()}, "log_ts_ms": ev["timestamp"]})
        _write(out_dir / f"{role}_saaf.jsonl", saaf)
        _write(out_dir / f"{role}_reports.jsonl", reports)
        counts[role] = {"saaf": len(saaf), "reports": len(reports)}
    metric_rows = []
    for role, key in (("enqueue", "EnqueueFn"), ("embed", "EmbedFn"), ("query", "QueryFn")):
        metric_rows += _metrics(m["outputs"][key], start, end)
    _write(out_dir / "cloudwatch_metrics.jsonl", metric_rows)
    manifest.update(run_id, collected=counts, phases={"collect_end": manifest.now()})
    return counts
