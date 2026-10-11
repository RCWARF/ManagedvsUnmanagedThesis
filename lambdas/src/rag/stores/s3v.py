import time

from ..clients import s3vectors
from ..settings import SETTINGS
from .base import Chunk, Hit


class S3VectorsStore:
    score_kind = "distance"

    def __init__(self, bucket: str | None = None, index: str | None = None):
        self.bucket = bucket or SETTINGS.vector_bucket
        self.index = index or SETTINGS.vector_index

    def put(self, chunks: list[Chunk]) -> None:
        now = int(time.time())
        s3vectors.put_vectors(
            vectorBucketName=self.bucket,
            indexName=self.index,
            vectors=[
                {
                    "key": c.key,
                    "data": {"float32": c.embedding},
                    "metadata": {
                        # filterable
                        "doc_id": c.doc_id,
                        "dataset": SETTINGS.dataset,
                        "chunk_index": c.chunk_index,
                        # non-filterable (declared on the index)
                        "content": c.content,
                        "source_key": c.source_key,
                        "ingested_at": now,
                    },
                }
                for c in chunks
            ],
        )

    def search(self, embedding: list[float], top_k: int) -> list[Hit]:
        resp = s3vectors.query_vectors(
            vectorBucketName=self.bucket,
            indexName=self.index,
            topK=top_k,
            queryVector={"float32": embedding},
            returnMetadata=True,
            returnDistance=True,
        )
        return [
            Hit(
                key=v["key"],
                score=v.get("distance"),
                doc_id=(v.get("metadata") or {}).get("doc_id", ""),
                content=(v.get("metadata") or {}).get("content", ""),
            )
            for v in resp.get("vectors", [])
        ]
