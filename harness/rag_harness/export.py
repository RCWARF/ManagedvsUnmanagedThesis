"""Excel workbook of every collected run: `rag export` -> results/rag_results.xlsx.

Sheets
  Runs           one row per run: config, ingest/query/quality/cost summary
  Embed          one row per embed Lambda invocation (one doc at ESM batch size 1)
  Queries        one row per query request (client timing joined with the Lambda's record)
  Enqueue        one row per enqueue invocation
  Lambda reports one row per Lambda REPORT line (billed duration, memory, init)
  Fields         what every column means

Runs-sheet means and counts are Excel formulas over the detail sheets, so they update if
rows are filtered out or corrected. Percentiles are computed here (Excel has no
percentile-with-criteria function that LibreOffice also evaluates) and marked as values.
The source of truth stays results/manifests and results/raw; the workbook is rebuilt from them.
"""
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .paths import MANIFESTS, RAW, RESULTS

FONT = "Arial"
HEAD = PatternFill("solid", start_color="1F3A5F")
COMPUTED = PatternFill("solid", start_color="EEF2F7")
THIN = Border(bottom=Side(style="thin", color="C9D1DC"))


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def _pct(values: list[float], p: float):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return v[int(round(p / 100 * (len(v) - 1)))]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# (header, key or function, number format, description) per sheet
EMBED_COLS = [
    ("run_id", "run_id", None, "Run the invocation belongs to"),
    ("t_s", "t_s", "0.000", "Start time in seconds after the run's first embed invocation"),
    ("runtime_ms", "runtime", "0", "SAAF wall time of the invocation"),
    ("new_container", "newcontainer", "0", "1 = cold start (first invocation in a new container)"),
    ("docs", "docs", "0", "Docs processed successfully in this invocation"),
    ("chunks", "chunks", "0", "Chunks embedded and written"),
    ("chars", "chars", "0", "Characters of document text"),
    ("embed_tokens", "embed_tokens", "0", "Titan input tokens billed for this invocation"),
    ("s3_read_ms", "s3_read_ms", "0.0", "Time reading the doc from S3"),
    ("chunk_ms", "chunk_ms", "0.0", "Time splitting the doc into chunks"),
    ("embed_stage_ms", "embed_stage_ms", "0.0", "Time for all Titan calls of the doc (parallel threads)"),
    ("embed_call_p50_ms", "embed_call_ms_p50", "0.0", "Median single Titan call inside the invocation"),
    ("put_stage_ms", "put_stage_ms", "0.0", "Time writing the doc's vectors to S3 Vectors"),
    ("put_calls", "put_call_ms_n", "0", "PutVectors calls made"),
    ("embed_throttles", "embed_throttles", "0", "Titan throttling errors (each retried by our code, counted here)"),
    ("embed_other_errors", "embed_other_errors", "0", "Other Titan errors"),
    ("messages_failed", "messages_failed", "0", "SQS messages returned for retry"),
    ("sqs_receive_count", "sqs_receive_count_max", "0", "Delivery attempt of the message (1 = first)"),
    ("embed_workers", "embed_workers", "0", "Threads per invocation"),
    ("cpu_user_delta", "cpuUserDelta", "0", "CPU time in user mode during the invocation (1/100 s)"),
    ("cpu_idle_delta", "cpuIdleDelta", "0", "CPU idle time during the invocation (1/100 s)"),
    ("container_id", "containerID", None, "Container identifier from SAAF"),
]
QUERY_COLS = [
    ("run_id", "run_id", None, "Run the request belongs to"),
    ("request", "i", "0", "Request number in the load (0-based)"),
    ("warmup", "warmup", None, "TRUE = warm-up request, excluded from warm statistics"),
    ("qid", "qid", None, "BEIR query id"),
    ("status", "status", None, "Client view: ok, error or client_error"),
    ("client_rtt_ms", "rtt_ms", "0.0", "Round trip measured by the load generator"),
    ("handler_ms", "handler_ms", "0.0", "Time inside the query Lambda handler"),
    ("new_container", "newcontainer", "0", "1 = cold start"),
    ("embed_ms", "embed_ms", "0.0", "Titan call for the question"),
    ("search_ms", "search_ms", "0.0", "S3 Vectors top-k query"),
    ("ttft_ms", "gen_ttft_ms", "0.0", "Nova time to first token"),
    ("gen_total_ms", "gen_total_ms", "0.0", "Nova total generation time"),
    ("gen_server_ms", "gen_server_latency_ms", "0", "Nova latency reported by Bedrock"),
    ("gen_input_tokens", "gen_input_tokens", "0", "Nova input tokens"),
    ("gen_output_tokens", "gen_output_tokens", "0", "Nova output tokens"),
    ("embed_tokens", "embed_tokens", "0", "Titan tokens for the question"),
    ("docs_retrieved", "docs_retrieved", "0", "Chunks retrieved"),
    ("stop_reason", "gen_stop_reason", None, "Why Nova stopped"),
]
ENQUEUE_COLS = [
    ("run_id", "run_id", None, "Run"),
    ("t_s", "t_s", "0.000", "Start time in seconds after the run's first enqueue invocation"),
    ("runtime_ms", "runtime", "0", "SAAF wall time"),
    ("new_container", "newcontainer", "0", "1 = cold start (blank for runs before this was recorded)"),
    ("s3_records", "num_s3_records", "0", "S3 events in the invocation"),
    ("messages_sent", "messages_sent", "0", "SQS messages sent"),
    ("messages_failed", "messages_failed", "0", "SQS sends that failed"),
]
REPORT_COLS = [
    ("run_id", "run_id", None, "Run"),
    ("function", "function", None, "enqueue, embed or query"),
    ("duration_ms", "duration_ms", "0.00", "Lambda-reported duration"),
    ("billed_ms", "billed_ms", "0", "Billed duration"),
    ("memory_mb", "memory_mb", "0", "Configured memory"),
    ("max_memory_used_mb", "max_mem_mb", "0", "Peak memory used"),
    ("init_ms", "init_ms", "0.00", "Init duration; present only on cold starts"),
]


def _sheet(wb, title: str, cols, rows: list[dict]):
    ws = wb.create_sheet(title)
    for j, (h, _, _, desc) in enumerate(cols, 1):
        c = ws.cell(row=1, column=j, value=h)
        c.font = Font(name=FONT, bold=True, color="FFFFFF")
        c.fill = HEAD
        c.alignment = Alignment(vertical="center")
    for i, r in enumerate(rows, 2):
        for j, (_, key, fmt, _) in enumerate(cols, 1):
            v = r.get(key)
            if fmt and v is not None and not isinstance(v, bool):
                v = _num(v)
            c = ws.cell(row=i, column=j, value=v)
            c.font = Font(name=FONT)
            if fmt:
                c.number_format = fmt
    for j, (h, _, _, _) in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(j)].width = max(11, min(34, len(h) + 4))
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{max(1, len(rows) + 1)}"
    return ws


def build(out: Path | None = None) -> Path:
    out = out or RESULTS / "rag_results.xlsx"
    manifests = sorted((json.loads(p.read_text()) for p in MANIFESTS.glob("*.json")),
                       key=lambda m: m.get("phases", {}).get("deploy_start", ""))
    embed_rows, query_rows, enq_rows, rep_rows, runs = [], [], [], [], []
    for m in manifests:
        rid = m["run_id"]
        d = RAW / rid
        e = sorted(_read(d / "embed_saaf.jsonl"), key=lambda r: r.get("startTime", 0))
        en = sorted(_read(d / "enqueue_saaf.jsonl"), key=lambda r: r.get("startTime", 0))
        q = {r.get("qid"): r for r in _read(d / "query_saaf.jsonl")}
        client = sorted(_read(d / "client_queries.jsonl"), key=lambda r: r["i"])
        for rows, src in ((embed_rows, e), (enq_rows, en)):
            t0 = src[0]["startTime"] if src else 0
            rows += [{**r, "run_id": rid, "t_s": (r["startTime"] - t0) / 1000} for r in src]
        query_rows += [{**q.get(c["qid"], {}), **c, "run_id": rid} for c in client]
        for fn in ("enqueue", "embed", "query"):
            rep_rows += [{**r, "run_id": rid, "function": fn} for r in _read(d / f"{fn}_reports.jsonl")]
        warm = [r for r in query_rows if r["run_id"] == rid and not r["warmup"] and r["status"] == "ok"]
        gaps = [b["startTime"] - a["endTime"] for a, b in zip(e, e[1:])]
        runs.append({"m": m, "has_raw": d.exists(),
                     "ingest_s": (e[-1]["endTime"] - e[0]["startTime"]) / 1000 if e else None,
                     "gap_p50": _pct(gaps, 50),
                     "inv_p50": _pct([r["runtime"] for r in e], 50), "inv_p95": _pct([r["runtime"] for r in e], 95),
                     "rtt_p50": _pct([r["rtt_ms"] for r in warm], 50), "rtt_p95": _pct([r["rtt_ms"] for r in warm], 95),
                     "ttft_p50": _pct([r.get("gen_ttft_ms") for r in warm], 50)})

    wb = Workbook()
    readme = wb.active
    readme.title = "README"
    ws_e = _sheet(wb, "Embed", EMBED_COLS, embed_rows)
    ws_q = _sheet(wb, "Queries", QUERY_COLS, query_rows)
    _sheet(wb, "Enqueue", ENQUEUE_COLS, enq_rows)
    _sheet(wb, "Lambda reports", REPORT_COLS, rep_rows)

    def col(cols, header):  # absolute column range on a detail sheet, e.g. Embed!$C:$C
        j = [h for h, *_ in cols].index(header) + 1
        L = get_column_letter(j)
        return f"${L}:${L}"

    E = lambda h: f"Embed!{col(EMBED_COLS, h)}"
    Q = lambda h: f"Queries!{col(QUERY_COLS, h)}"
    # Runs sheet: (header, value or formula builder, number format, description, computed-in-python?)
    run_cols = [
        ("run_id", lambda r, i: r["m"]["run_id"], None, "Run id: experiment-dataset-concurrency-repetition-confighash", False),
        ("experiment", lambda r, i: r["m"]["experiment_id"], None, "Experiment (SMOKE, E1...)", False),
        ("dataset", lambda r, i: r["m"]["config"]["dataset"]["name"], None, "BEIR dataset", False),
        ("concurrency", lambda r, i: r["m"]["config"]["ingest"]["concurrency"], "0", "Embed Lambda reserved concurrency (the swept variable in E1)", False),
        ("repetition", lambda r, i: r["m"]["repetition"], "0", "Repetition number", False),
        ("threads", lambda r, i: r["m"]["config"]["embedding"]["workers_per_invocation"], "0", "Titan threads per embed invocation", False),
        ("dims", lambda r, i: r["m"]["config"]["embedding"]["dimensions"], "0", "Embedding dimensions", False),
        ("embed_memory_mb", lambda r, i: r["m"]["config"]["lambdas"]["embed"]["memory_mb"], "0", "Embed Lambda memory", False),
        ("started_utc", lambda r, i: r["m"].get("phases", {}).get("deploy_end", "")[:19].replace("T", " "), None, "When the run's settings were applied", False),
        ("docs_expected", lambda r, i: r["m"].get("ingest_expected", {}).get("docs"), "#,##0", "Docs uploaded", False),
        ("vectors_expected", lambda r, i: r["m"].get("ingest_result", {}).get("vectors_expected"), "#,##0", "Chunks the corpus should produce", False),
        ("vectors_stored", lambda r, i: r["m"].get("ingest_result", {}).get("vectors_stored"), "#,##0", "Vectors counted in the index after ingest", False),
        ("completeness", lambda r, i: f"=IFERROR(L{i}/K{i},\"\")", "0.0%", "vectors_stored / vectors_expected", False),
        ("dlq_messages", lambda r, i: r["m"].get("ingest_result", {}).get("dlq_messages"), "0", "Messages that failed 3 times", False),
        ("ingest_s", lambda r, i: r["ingest_s"], "0.0", "First embed start to last embed end (seconds)", True),
        ("docs_per_s", lambda r, i: f"=IFERROR(J{i}/O{i},\"\")", "0.00", "Ingest throughput", False),
        ("embed_invocations", lambda r, i: f"=COUNTIFS({E('run_id')},A{i})", "#,##0", "Rows on the Embed sheet", False),
        ("embed_cold_starts", lambda r, i: f"=SUMIFS({E('new_container')},{E('run_id')},A{i})", "0", "Embed containers started", False),
        ("embed_mean_ms", lambda r, i: f"=IFERROR(AVERAGEIFS({E('runtime_ms')},{E('run_id')},A{i}),\"\")", "0", "Mean embed invocation time", False),
        ("embed_p50_ms", lambda r, i: r["inv_p50"], "0", "Median embed invocation time", True),
        ("embed_p95_ms", lambda r, i: r["inv_p95"], "0", "95th percentile embed invocation time", True),
        ("s3_read_mean_ms", lambda r, i: f"=IFERROR(AVERAGEIFS({E('s3_read_ms')},{E('run_id')},A{i}),\"\")", "0", "Mean S3 read per invocation", False),
        ("titan_mean_ms", lambda r, i: f"=IFERROR(AVERAGEIFS({E('embed_stage_ms')},{E('run_id')},A{i}),\"\")", "0", "Mean Titan stage per invocation", False),
        ("put_mean_ms", lambda r, i: f"=IFERROR(AVERAGEIFS({E('put_stage_ms')},{E('run_id')},A{i}),\"\")", "0", "Mean S3 Vectors write per invocation", False),
        ("queue_gap_p50_ms", lambda r, i: r["gap_p50"], "0", "Median idle time between consecutive embed invocations (meaningful at concurrency 1)", True),
        ("titan_throttles", lambda r, i: f"=SUMIFS({E('embed_throttles')},{E('run_id')},A{i})", "0", "Titan throttles during ingest", False),
        ("ingest_tokens", lambda r, i: f"=SUMIFS({E('embed_tokens')},{E('run_id')},A{i})", "#,##0", "Titan tokens during ingest", False),
        ("query_requests", lambda r, i: f"=COUNTIFS({Q('run_id')},A{i})", "0", "Requests sent by the load generator", False),
        ("query_errors", lambda r, i: f"=COUNTIFS({Q('run_id')},A{i},{Q('status')},\"<>ok\")", "0", "Requests that failed", False),
        ("warm_rtt_mean_ms", lambda r, i: f"=IFERROR(AVERAGEIFS({Q('client_rtt_ms')},{Q('run_id')},A{i},{Q('warmup')},FALSE),\"\")", "0", "Mean client round trip, warm-up excluded", False),
        ("warm_rtt_p50_ms", lambda r, i: r["rtt_p50"], "0", "Median warm round trip", True),
        ("warm_rtt_p95_ms", lambda r, i: r["rtt_p95"], "0", "95th percentile warm round trip", True),
        ("ttft_p50_ms", lambda r, i: r["ttft_p50"], "0", "Median Nova time to first token (warm)", True),
        ("search_mean_ms", lambda r, i: f"=IFERROR(AVERAGEIFS({Q('search_ms')},{Q('run_id')},A{i},{Q('warmup')},FALSE),\"\")", "0", "Mean S3 Vectors query time (warm)", False),
        ("ndcg@10", lambda r, i: r["m"].get("retrieval_quality", {}).get("ndcg@10"), "0.000", "Retrieval quality over the BEIR test queries", False),
        ("recall@10", lambda r, i: r["m"].get("retrieval_quality", {}).get("recall@10"), "0.000", "Share of relevant docs in the top 10", False),
        ("quality_queries", lambda r, i: r["m"].get("retrieval_quality", {}).get("queries"), "0", "Queries scored (subset runs score only loaded docs)", False),
    ]
    cost_keys = ["titan", "nova", "lambda", "s3v_put", "s3v_query", "sqs", "s3_put"]
    first_cost = len(run_cols) + 1
    for k in cost_keys:
        run_cols.append((f"cost_{k}_usd", lambda r, i, k=k: r["m"].get("cost_estimate_usd", {}).get(k), "$0.000000",
                         f"Estimated {k} cost (prices retrieved 2026-10-11, see cost.py)", False))
    a, b = get_column_letter(first_cost), get_column_letter(first_cost + len(cost_keys) - 1)
    run_cols.append(("cost_total_usd", lambda r, i: f"=SUM({a}{i}:{b}{i})", "$0.0000", "Sum of the cost columns (excludes VPC endpoints, logs, storage)", False))
    tot = get_column_letter(len(run_cols))
    run_cols.append(("cost_per_1k_docs_usd", lambda r, i: f"=IFERROR({tot}{i}/J{i}*1000,\"\")", "$0.0000", "Total cost per 1,000 docs ingested", False))

    ws = wb.create_sheet("Runs", 1)
    for j, (h, _, _, _, computed) in enumerate(run_cols, 1):
        c = ws.cell(row=1, column=j, value=h)
        c.font = Font(name=FONT, bold=True, color="FFFFFF")
        c.fill = HEAD
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = 30 if h == "run_id" else max(11, min(20, len(h) + 2))
    for i, r in enumerate(runs, 2):
        for j, (_, f, fmt, _, computed) in enumerate(run_cols, 1):
            c = ws.cell(row=i, column=j, value=f(r, i))
            c.font = Font(name=FONT)
            c.border = THIN
            if fmt:
                c.number_format = fmt
            if computed:
                c.fill = COMPUTED
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(run_cols))}{len(runs) + 1}"

    fields = wb.create_sheet("Fields")
    for j, h in enumerate(("sheet", "column", "meaning"), 1):
        c = fields.cell(row=1, column=j, value=h)
        c.font = Font(name=FONT, bold=True, color="FFFFFF")
        c.fill = HEAD
    n = 2
    for sheet, cols in (("Runs", [(h, None, None, d) for h, _, _, d, _ in run_cols]), ("Embed", EMBED_COLS),
                        ("Queries", QUERY_COLS), ("Enqueue", ENQUEUE_COLS), ("Lambda reports", REPORT_COLS)):
        for h, _, _, d in cols:
            for j, v in enumerate((sheet, h, d), 1):
                fields.cell(row=n, column=j, value=v).font = Font(name=FONT)
            n += 1
    fields.column_dimensions["A"].width, fields.column_dimensions["B"].width, fields.column_dimensions["C"].width = 16, 24, 90
    fields.freeze_panes = "A2"

    lines = [
        ("RAG thesis results", True),
        ("Rebuilt from results/manifests and results/raw by `rag export`. Do not edit here; edit the source and re-export.", False),
        ("", False),
        ("Runs: one row per run. White cells are config values or Excel formulas over the detail sheets; "
         "shaded cells (percentiles, ingest time) were computed by the exporter.", False),
        ("Embed: one row per embed Lambda invocation (one doc each at ESM batch size 1), with per-stage times.", False),
        ("Queries: one row per query request, client round trip joined with the Lambda's own timings.", False),
        ("Enqueue: one row per enqueue Lambda invocation.", False),
        ("Lambda reports: one row per Lambda REPORT log line (billed duration, memory, init on cold starts).", False),
        ("Fields: what every column means.", False),
        ("", False),
        (f"Runs included: {len(runs)}. Runs without raw data still appear on Runs, with blanks for the detail columns.", False),
        ("Costs exclude VPC endpoints (about $0.06 per hour while the network is up), CloudWatch Logs and storage.", False),
    ]
    for i, (text, bold) in enumerate(lines, 1):
        c = readme.cell(row=i, column=1, value=text)
        c.font = Font(name=FONT, bold=bold, size=14 if bold else 10)
    readme.column_dimensions["A"].width = 120
    wb.move_sheet("Runs", offset=-wb.index(wb["Runs"]) + 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out
