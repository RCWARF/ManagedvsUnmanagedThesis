"""Run-config schema. Every experiment run is fully described by one RunConfig.

The config hash (sha256 of the canonical JSON) goes into the stack name, every
Lambda's environment, every SAAF record and the run manifest, so any result can
be traced back to the exact configuration that produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CROSS_REGION_PREFIXES = {"us", "global", "eu", "apac", "jp", "us-gov"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Quotas(_Strict):
    """Expected quotas. The harness also snapshots live values into the manifest."""
    titan_rpm: int = 6000
    titan_tpm: int = 3_072_000
    nova_lite_rpm: int | None = None
    nova_lite_tpm: int | None = None
    lambda_account_concurrency: int | None = None

    @property
    def titan_crossover_tokens(self) -> float:
        # Above this average request size TPM binds before RPM.
        return self.titan_tpm / self.titan_rpm


class Dataset(_Strict):
    name: Literal["scifact", "nfcorpus", "fiqa", "nq_subset"]
    split: Literal["test", "dev", "train"] = "test"
    corpus_uri: str
    queries_uri: str
    qrels_uri: str
    doc_limit: int | None = None
    packing: Literal["one_doc_per_object"] = "one_doc_per_object"
    sha256: str = Field(description="checksum of corpus.jsonl, pinned so every run uses identical input")


class Chunking(_Strict):
    strategy: Literal["chars"] = "chars"
    size: Annotated[int, Field(ge=100, le=8000)] = 1000
    overlap: Annotated[int, Field(ge=0)] = 200

    @model_validator(mode="after")
    def _overlap_lt_size(self):
        if self.overlap >= self.size:
            raise ValueError("chunking.overlap must be smaller than chunking.size")
        return self


class Embedding(_Strict):
    provider: Literal["bedrock_titan_v2"] = "bedrock_titan_v2"
    model_id: str = "amazon.titan-embed-text-v2:0"
    mode: Literal["on_demand"] = "on_demand"  # "batch" comes with its own experiment
    dimensions: Literal[256, 512, 1024] = 1024  # single source of truth for ingest AND query
    normalize: bool = True
    workers_per_invocation: Annotated[int, Field(ge=1, le=16)] = 8
    max_retries: Annotated[int, Field(ge=0, le=10)] = 6
    use_app_profile: bool = True  # application inference profile = per-run cost tags


class S3VectorStore(_Strict):
    kind: Literal["s3vectors"] = "s3vectors"
    distance: Literal["cosine", "euclidean"] = "cosine"
    put_batch_size: Annotated[int, Field(ge=1, le=500)] = 32
    non_filterable_keys: tuple[str, ...] = ("content", "source_key", "ingested_at")

    @model_validator(mode="after")
    def _limit(self):
        if len(self.non_filterable_keys) > 10:
            raise ValueError("S3 Vectors allows at most 10 non-filterable metadata keys per index")
        return self


class Generation(_Strict):
    model_id: str = "amazon.nova-lite-v1:0"
    temperature: Annotated[float, Field(ge=0, le=1)] = 0.0
    max_tokens: Annotated[int, Field(ge=1, le=10000)] = 256
    top_k: Annotated[int, Field(ge=1, le=100)] = 5
    streaming: bool = True  # converse_stream, needed for time-to-first-token
    prompt_template: Literal["qa_v1"] = "qa_v1"
    use_app_profile: bool = True
    allow_cross_region: bool = False

    @model_validator(mode="after")
    def _in_region(self):
        prefix = self.model_id.split(".", 1)[0]
        if prefix in CROSS_REGION_PREFIXES and not self.allow_cross_region:
            raise ValueError(
                f"generation.model_id {self.model_id!r} is a cross-region inference profile; "
                "final runs are single-region. Set allow_cross_region and record the confound in notes."
            )
        return self


class LambdaFn(_Strict):
    memory_mb: Annotated[int, Field(ge=128, le=10240)]
    timeout_s: Annotated[int, Field(ge=1, le=900)]
    arch: Literal["arm64", "x86_64"] = "arm64"
    runtime: Literal["python3.13", "python3.14"] = "python3.14"


class Lambdas(_Strict):
    enqueue: LambdaFn = LambdaFn(memory_mb=256, timeout_s=60)
    embed: LambdaFn = LambdaFn(memory_mb=1769, timeout_s=900)
    query: LambdaFn = LambdaFn(memory_mb=1024, timeout_s=60)


class IngestWorkload(_Strict):
    concurrency: Annotated[int, Field(ge=1, le=1000)] = 1  # one value per run; sweeps live in sweep files
    esm_batch_size: Annotated[int, Field(ge=1, le=10)] = 1  # FIFO queues cap batches at 10
    upload_parallelism: Annotated[int, Field(ge=1, le=256)] = 64
    timeout_min: int = 120
    verify_completeness: bool = True


class QueryWorkload(_Strict):
    entrypoint: Literal["direct", "function_url"] = "direct"
    concurrency: Annotated[int, Field(ge=1, le=1000)] = 10
    requests: Annotated[int, Field(ge=1)] = 1000
    warmup_requests: Annotated[int, Field(ge=0)] = 50
    sampling: Literal["without_replacement", "with_replacement"] = "without_replacement"
    seed: int = 42


class Network(_Strict):
    in_vpc: bool = True  # every variant runs in the same VPC so the network path is constant


class RunConfig(_Strict):
    schema_version: Literal["1"] = "1"
    experiment_id: Literal["E1", "E2", "E3", "E4", "SMOKE"]
    variant_id: Literal["A-s3v-titan-nova"] = "A-s3v-titan-nova"
    region: Literal["us-west-2"] = "us-west-2"
    repetition: Annotated[int, Field(ge=1)] = 1
    quotas: Quotas = Quotas()
    dataset: Dataset
    chunking: Chunking = Chunking()
    embedding: Embedding = Embedding()
    vector_store: S3VectorStore = S3VectorStore()
    generation: Generation = Generation()
    lambdas: Lambdas = Lambdas()
    network: Network = Network()
    ingest: IngestWorkload = IngestWorkload()
    query: QueryWorkload = QueryWorkload()
    budget_cap_usd: Annotated[float, Field(gt=0)] = 25.0
    notes: str = ""

    @model_validator(mode="after")
    def _confounds_need_notes(self):
        if self.generation.allow_cross_region and "cross-region" not in self.notes.lower():
            raise ValueError("allow_cross_region is set: name the confound in notes ('cross-region ...')")
        return self

    # ---- derived values -------------------------------------------------
    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:12]

    def run_slug(self) -> str:
        """Stable, CloudFormation-safe name: E1-scifact-c8-r1-<hash6>."""
        return (
            f"{self.experiment_id}-{self.dataset.name.replace('_', '')}"
            f"-c{self.ingest.concurrency}-r{self.repetition}-{self.config_hash()[:6]}"
        )

    def peak_embed_inflight(self) -> int:
        return self.ingest.concurrency * self.embedding.workers_per_invocation


def load_config(path: str | Path, overrides: dict | None = None) -> RunConfig:
    """Load a YAML run config, optionally applying dotted overrides (e.g. {"ingest.concurrency": 8})."""
    # ${VARS} (e.g. DATASETS_BUCKET, a RagCommon output) are filled from the environment.
    data = yaml.safe_load(os.path.expandvars(Path(path).read_text()))
    for dotted, value in (overrides or {}).items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return RunConfig.model_validate(data)


class Sweep(_Strict):
    """A grid of overrides on one base config. Each cell x repetition is one run (one stack)."""
    base: str
    grid: dict[str, list] = Field(default_factory=dict)
    repetitions: Annotated[int, Field(ge=1)] = 3

    def expand(self, sweep_path: str | Path) -> list[RunConfig]:
        import itertools

        base_path = Path(sweep_path).parent / self.base
        keys = list(self.grid)
        runs = []
        for values in itertools.product(*(self.grid[k] for k in keys)) if keys else [()]:
            for rep in range(1, self.repetitions + 1):
                overrides = dict(zip(keys, values)) | {"repetition": rep}
                runs.append(load_config(base_path, overrides))
        return runs


def load_sweep(path: str | Path) -> list[RunConfig]:
    return Sweep.model_validate(yaml.safe_load(Path(path).read_text())).expand(path)
