"""Runtime settings, read once per container from env vars that CDK sets from the RunConfig."""
import os
from dataclasses import dataclass


def _bool(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    region: str
    run_id: str
    config_hash: str
    git_sha: str
    dataset: str
    # embedding
    embed_model: str  # model id or application inference profile ARN
    embed_dims: int
    embed_normalize: bool
    embed_workers: int
    embed_max_retries: int
    # chunking
    chunk_size: int
    chunk_overlap: int
    # vector store
    vector_store: str
    vector_bucket: str
    vector_index: str
    put_batch_size: int
    # generation
    gen_model: str
    gen_temperature: float
    gen_max_tokens: int
    gen_top_k: int
    gen_streaming: bool
    # enqueue
    queue_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        return cls(
            region=e("AWS_REGION", "us-west-2"),
            run_id=e("RUN_ID", "unset"),
            config_hash=e("CONFIG_HASH", "unset"),
            git_sha=e("GIT_SHA", "unset"),
            dataset=e("DATASET", "unset"),
            embed_model=e("EMBED_MODEL", "amazon.titan-embed-text-v2:0"),
            embed_dims=int(e("EMBED_DIMS", "1024")),
            embed_normalize=_bool("EMBED_NORMALIZE", "true"),
            embed_workers=int(e("EMBED_WORKERS", "8")),
            embed_max_retries=int(e("EMBED_MAX_RETRIES", "6")),
            chunk_size=int(e("CHUNK_SIZE", "1000")),
            chunk_overlap=int(e("CHUNK_OVERLAP", "200")),
            vector_store=e("VECTOR_STORE", "s3v"),
            vector_bucket=e("VECTOR_BUCKET", ""),
            vector_index=e("VECTOR_INDEX", ""),
            put_batch_size=int(e("PUT_BATCH_SIZE", "32")),
            gen_model=e("GEN_MODEL", "amazon.nova-lite-v1:0"),
            gen_temperature=float(e("GEN_TEMPERATURE", "0")),
            gen_max_tokens=int(e("GEN_MAX_TOKENS", "256")),
            gen_top_k=int(e("GEN_TOP_K", "5")),
            gen_streaming=_bool("GEN_STREAMING", "true"),
            queue_url=e("QUEUE_URL", ""),
        )


SETTINGS = Settings.from_env()
