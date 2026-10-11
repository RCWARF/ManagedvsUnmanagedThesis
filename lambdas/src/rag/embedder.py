"""Titan Text Embeddings V2 calls with explicit throttle accounting."""
import json
import random
import time
from dataclasses import dataclass

from botocore.exceptions import ClientError

from .clients import bedrock
from .settings import SETTINGS

THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"}


@dataclass
class EmbedResult:
    embedding: list[float]
    latency_ms: float  # successful attempt only
    wall_ms: float  # first attempt start to success, including backoff
    input_tokens: int
    throttles: int
    other_errors: int


def embed(text: str, *, dims: int | None = None, max_retries: int | None = None) -> EmbedResult:
    body = json.dumps({
        "inputText": text,
        "dimensions": dims or SETTINGS.embed_dims,
        "normalize": SETTINGS.embed_normalize,
    })
    retries = SETTINGS.embed_max_retries if max_retries is None else max_retries
    throttles = other = 0
    t_start = time.perf_counter()
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            resp = bedrock.invoke_model(modelId=SETTINGS.embed_model, body=body)
            payload = json.loads(resp["body"].read())
            t1 = time.perf_counter()
            return EmbedResult(
                embedding=payload["embedding"],
                latency_ms=(t1 - t0) * 1000.0,
                wall_ms=(t1 - t_start) * 1000.0,
                input_tokens=int(payload.get("inputTextTokenCount", 0)),
                throttles=throttles,
                other_errors=other,
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in THROTTLE_CODES:
                throttles += 1
            elif code in {"ValidationException", "AccessDeniedException", "ResourceNotFoundException"}:
                raise  # not transient: retrying would only hide a bug
            else:
                other += 1
        except (ConnectionError, TimeoutError):
            other += 1
        if attempt == retries:
            raise RuntimeError(f"embedding failed after {retries} retries ({throttles} throttles, {other} other)")
        # Exponential backoff with full jitter, capped at 8 s.
        time.sleep(random.uniform(0, min(8.0, 0.2 * 2 ** attempt)))
    raise AssertionError("unreachable")
