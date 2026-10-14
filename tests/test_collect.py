import json

from rag_harness import collect, manifest


def test_collect_waits_for_late_log_events(tmp_path, monkeypatch):
    run_id = "SMOKE-scifact-c1-r1-abc123"
    monkeypatch.setattr(collect, "RAW", tmp_path)
    m = {"run_id": run_id, "phases": {"deploy_end": "2026-10-14T00:00:00+00:00"},
         "outputs": {"EnqueueFn": "enq", "EmbedFn": "emb", "QueryFn": "qry"},
         "ingest_expected": {"docs": 2}, "ingest_result": {"dlq_messages": 0}}
    saved = {}
    monkeypatch.setattr(manifest, "load", lambda r: m)
    monkeypatch.setattr(manifest, "update", lambda r, **kw: saved.update(kw))
    (tmp_path / run_id).mkdir()
    (tmp_path / run_id / "client_queries.jsonl").write_text("{}\n" * 4)
    monkeypatch.setattr(collect, "_metrics", lambda *a: [])
    monkeypatch.setattr(collect.time, "sleep", lambda s: None)
    polls = {"n": 0}

    def events(group, start_ms, pattern):
        fn = group.rsplit("/", 1)[1]
        if fn == "qry" and pattern == '"rag_metric"':
            polls["n"] += 1
        n_query = 2 if polls["n"] < 2 else 4  # half of the query records arrive late
        recs = {"enq": [{"num_s3_records": 1}] * 2, "emb": [{"docs": 1}] * 2, "qry": [{}] * n_query}[fn]
        for i, r in enumerate(recs):
            if pattern == '"rag_metric"':
                yield {"timestamp": i, "message": json.dumps({"run_id": run_id, **r})}
            else:
                yield {"timestamp": i, "message": f"REPORT RequestId: r{i}\tDuration: 1.0 ms\t"
                                                  "Billed Duration: 1 ms\tMemory Size: 128 MB\tMax Memory Used: 50 MB"}

    monkeypatch.setattr(collect, "_events", events)
    counts = collect.run(run_id, max_wait_s=60)
    assert counts["query"] == {"saaf": 4, "reports": 4, "complete": True}
    assert polls["n"] == 2
