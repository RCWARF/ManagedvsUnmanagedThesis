"""Query path: Titan query embedding -> vector search -> Nova answer.

Request: {"question": str, "qid": str (optional), "top_k": int (optional)}
Failures raise, so the load generator sees an invocation error instead of a 200.
"""
import time

from rag.embedder import embed
from rag.generation import generate
from rag.metrics import emit
from rag.prompt import build_prompt
from rag.settings import SETTINGS
from rag.stores.base import get_store
from saaf.Inspector import Inspector

STORE = get_store(SETTINGS.vector_store)


def lambda_handler(event, context):
    inspector = Inspector()
    inspector.inspectContainer()  # newcontainer flag: cold starts per stage
    t_inv = time.perf_counter()
    question = (event.get("question") or "").strip()
    top_k = int(event.get("top_k") or SETTINGS.gen_top_k)
    inspector.addAttribute("qid", event.get("qid"))
    inspector.addAttribute("top_k", top_k)
    status = "failed"
    try:
        if not question:
            raise ValueError("question missing")

        # Same dimensionality as ingestion; query-time throttles are not retried silently either.
        e = embed(question, dims=SETTINGS.embed_dims)
        inspector.addAttribute("embed_ms", round(e.latency_ms, 2))
        inspector.addAttribute("embed_wall_ms", round(e.wall_ms, 2))
        inspector.addAttribute("embed_tokens", e.input_tokens)
        inspector.addAttribute("embed_throttles", e.throttles)

        t0 = time.perf_counter()
        hits = STORE.search(e.embedding, top_k)
        inspector.addAttribute("search_ms", round((time.perf_counter() - t0) * 1000.0, 2))
        inspector.addAttribute("docs_retrieved", len(hits))
        inspector.addAttribute("retrieved_doc_ids", [h.doc_id for h in hits])
        inspector.addAttribute("retrieved_scores", [h.score for h in hits])
        inspector.addAttribute("score_kind", STORE.score_kind)

        prompt = build_prompt(question, [h.content for h in hits])
        g = generate(prompt)
        inspector.addAttribute("prompt_chars", len(prompt))
        inspector.addAttribute("gen_ttft_ms", round(g.ttft_ms, 2) if g.ttft_ms is not None else None)
        inspector.addAttribute("gen_total_ms", round(g.total_ms, 2))
        inspector.addAttribute("gen_server_latency_ms", g.server_latency_ms)
        inspector.addAttribute("gen_input_tokens", g.input_tokens)
        inspector.addAttribute("gen_output_tokens", g.output_tokens)
        inspector.addAttribute("gen_stop_reason", g.stop_reason)
        inspector.addAttribute("answer_chars", len(g.text))
        status = "success"
        return {"qid": event.get("qid"), "answer": g.text, "doc_ids": [h.doc_id for h in hits]}
    except Exception as exc:
        inspector.addAttribute("error", f"{type(exc).__name__}: {exc}"[:500])
        raise
    finally:
        inspector.addAttribute("status", status)
        inspector.addAttribute("handler_ms", round((time.perf_counter() - t_inv) * 1000.0, 2))
        emit(inspector, SETTINGS, "query")
