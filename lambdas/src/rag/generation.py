"""Nova (Converse API). Streaming captures time-to-first-token."""
import time
from dataclasses import dataclass

from .clients import bedrock
from .settings import SETTINGS


@dataclass
class GenResult:
    text: str
    ttft_ms: float | None
    total_ms: float
    input_tokens: int
    output_tokens: int
    stop_reason: str
    server_latency_ms: int | None  # Bedrock-reported latency, when present


def generate(prompt: str) -> GenResult:
    kwargs = dict(
        modelId=SETTINGS.gen_model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": SETTINGS.gen_max_tokens, "temperature": SETTINGS.gen_temperature},
    )
    t0 = time.perf_counter()
    if not SETTINGS.gen_streaming:
        resp = bedrock.converse(**kwargs)
        t1 = time.perf_counter()
        usage = resp.get("usage", {})
        return GenResult(
            text=resp["output"]["message"]["content"][0]["text"],
            ttft_ms=None,
            total_ms=(t1 - t0) * 1000.0,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            stop_reason=resp.get("stopReason", ""),
            server_latency_ms=resp.get("metrics", {}).get("latencyMs"),
        )

    resp = bedrock.converse_stream(**kwargs)
    parts, ttft, usage, stop, server_ms = [], None, {}, "", None
    for event in resp["stream"]:
        if "contentBlockDelta" in event:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000.0
            parts.append(event["contentBlockDelta"]["delta"].get("text", ""))
        elif "messageStop" in event:
            stop = event["messageStop"].get("stopReason", "")
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            server_ms = event["metadata"].get("metrics", {}).get("latencyMs")
    return GenResult(
        text="".join(parts),
        ttft_ms=ttft,
        total_ms=(time.perf_counter() - t0) * 1000.0,
        input_tokens=usage.get("inputTokens", 0),
        output_tokens=usage.get("outputTokens", 0),
        stop_reason=stop,
        server_latency_ms=server_ms,
    )
