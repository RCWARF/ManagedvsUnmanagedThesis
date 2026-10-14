"""Pull raw SAAF records, Lambda REPORT lines and CloudWatch metrics for one run.

Everything lands in results/raw/<run_id>/ as JSONL (and parquet when pandas+pyarrow
are installed). Timestamps stay as epoch milliseconds: no spreadsheet round-trips.
"""
import json
import re
import time
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


def _expected(m: dict, out_dir) -> dict:
    """Minimum record counts a complete collection must reach."""
    docs = m.get("ingest_expected", {}).get("docs", 0)
    client_log = out_dir / "client_queries.jsonl"
    queries = sum(1 for _ in client_log.open()) if client_log.exists() else 0
    return {"enqueue": ("num_s3_records", docs), "embed": ("docs", docs - m.get("ingest_result", {}).get("dlq_messages", 0)),
            "query": (None, queries)}


def _complete(role: str, saaf: list, reports: list, want: tuple) -> bool:
    field, n = want
    got = len(saaf) if field is None else sum(int(r.get(field) or 0) for r in saaf)
    return got >= n and len(reports) >= len(saaf)


def _pull(group: str, run_id: str, start_ms: int) -> tuple[list, list]:
    saaf, reports = [], []
    for ev in _events(group, start_ms, '"rag_metric"'):
        try:
            rec = json.loads(ev["message"])
        except ValueError:
            continue
        if rec.get("run_id") != run_id:
            continue
        rec["log_ts_ms"] = ev["timestamp"]
        saaf.append(rec)
    for ev in _events(group, start_ms, '"REPORT RequestId"'):
        mt = REPORT_RE.search(ev["message"])
        if mt:
            reports.append({**{k: v for k, v in mt.groupdict().items()}, "log_ts_ms": ev["timestamp"]})
    return saaf, reports


def run(run_id: str, max_wait_s: int = 300) -> dict:
    m = manifest.load(run_id)
    out_dir = RAW / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Log groups are shared by the runs on one stack: SAAF records are matched on run_id, and
    # REPORT lines and metrics are limited to the window after this run's settings were applied.
    start = datetime.fromisoformat(m["phases"].get("deploy_end") or m["phases"]["deploy_start"])
    start_ms = int(start.timestamp() * 1000)
    fns = {"enqueue": m["outputs"]["EnqueueFn"], "embed": m["outputs"]["EmbedFn"], "query": m["outputs"]["QueryFn"]}
    want = _expected(m, out_dir)
    # Log events can take a while to become searchable, so poll until every stage has at least
    # the records the run produced (one per S3 event, per doc, per client request) or time runs out.
    deadline = time.time() + max_wait_s
    while True:
        pulled = {role: _pull(f"/aws/lambda/{fn}", run_id, start_ms) for role, fn in fns.items()}
        complete = {role: _complete(role, *pulled[role], want[role]) for role in fns}
        if all(complete.values()) or time.time() > deadline:
            break
        time.sleep(15)
    counts = {}
    for role, (saaf, reports) in pulled.items():
        _write(out_dir / f"{role}_saaf.jsonl", saaf)
        _write(out_dir / f"{role}_reports.jsonl", reports)
        counts[role] = {"saaf": len(saaf), "reports": len(reports), "complete": complete[role]}
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    metric_rows = []
    for fn in fns.values():
        metric_rows += _metrics(fn, start, end)
    _write(out_dir / "cloudwatch_metrics.jsonl", metric_rows)
    manifest.update(run_id, collected=counts, phases={"collect_end": manifest.now()})
    return counts
