from dataclasses import dataclass
from typing import Protocol


@dataclass
class Chunk:
    key: str
    embedding: list[float]
    doc_id: str
    chunk_index: int
    content: str
    source_key: str


@dataclass
class Hit:
    key: str
    score: float | None  # distance for S3 Vectors, similarity for OpenSearch; see `score_kind`
    doc_id: str
    content: str


class VectorStore(Protocol):
    score_kind: str

    def put(self, chunks: list[Chunk]) -> None: ...

    def search(self, embedding: list[float], top_k: int) -> list[Hit]: ...


def get_store(kind: str) -> VectorStore:
    if kind == "s3v":
        from .s3v import S3VectorsStore
        return S3VectorsStore()
    raise ValueError(f"unknown VECTOR_STORE {kind!r}")
