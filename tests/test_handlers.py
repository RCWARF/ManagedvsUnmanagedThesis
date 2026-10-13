"""Handler logic with AWS calls faked: throttle counting, chunk keys, failure reporting."""
import io
import json

import pytest
from botocore.exceptions import ClientError


class FakeBedrock:
    def __init__(self, throttle_first=0):
        self.calls = 0
        self.throttle_first = throttle_first

    def invoke_model(self, modelId, body):
        self.calls += 1
        if self.calls <= self.throttle_first:
            raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "InvokeModel")
        dims = json.loads(body)["dimensions"]
        return {"body": io.BytesIO(json.dumps({"embedding": [0.1] * dims, "inputTextTokenCount": 7}).encode())}


class FakeStore:
    score_kind = "distance"

    def __init__(self):
        self.put_batches = []

    def put(self, chunks):
        self.put_batches.append(chunks)


class FakeS3:
    def __init__(self, doc):
        self.doc = doc

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(json.dumps(self.doc).encode())}


@pytest.fixture
def embed_mod(monkeypatch):
    import rag.embedder as embedder
    import embed_handler
    monkeypatch.setattr(embedder.time, "sleep", lambda s: None)
    return embedder, embed_handler


def test_throttles_are_counted_not_hidden(embed_mod, monkeypatch):
    embedder, _ = embed_mod
    fake = FakeBedrock(throttle_first=2)
    monkeypatch.setattr(embedder, "bedrock", fake)
    r = embedder.embed("hello", dims=256)
    assert r.throttles == 2 and len(r.embedding) == 256 and r.input_tokens == 7


def test_validation_errors_are_not_retried(embed_mod, monkeypatch):
    embedder, _ = embed_mod

    class Bad:
        calls = 0

        def invoke_model(self, **kw):
            Bad.calls += 1
            raise ClientError({"Error": {"Code": "ValidationException", "Message": "bad"}}, "InvokeModel")

    monkeypatch.setattr(embedder, "bedrock", Bad())
    with pytest.raises(ClientError):
        embedder.embed("x")
    assert Bad.calls == 1


def test_embed_handler_chunks_keys_and_batches(embed_mod, monkeypatch, capsys):
    embedder, handler = embed_mod
    monkeypatch.setattr(embedder, "bedrock", FakeBedrock())
    store = FakeStore()
    monkeypatch.setattr(handler, "STORE", store)
    doc = {"_id": "MED-10", "title": "T", "text": "x" * 2500}
    monkeypatch.setattr(handler, "s3", FakeS3(doc))
    event = {"Records": [{"messageId": "m1", "body": json.dumps({"bucket": "b", "key": "scifact/MED-10.json"})}]}
    out = handler.lambda_handler(event, None)
    assert out == {"batchItemFailures": []}
    chunks = [c for batch in store.put_batches for c in batch]
    assert [c.key for c in chunks][:2] == ["MED-10#00000", "MED-10#00001"]
    assert all(c.doc_id == "MED-10" for c in chunks)
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["rag_metric"] == 1 and record["chunks"] == len(chunks) and record["embed_throttles"] == 0


def test_embed_handler_reports_failed_message(embed_mod, monkeypatch):
    _, handler = embed_mod

    class Boom:
        def get_object(self, **kw):
            raise RuntimeError("no such key")

    monkeypatch.setattr(handler, "s3", Boom())
    event = {"Records": [{"messageId": "m9", "body": json.dumps({"bucket": "b", "key": "k"})}]}
    assert handler.lambda_handler(event, None) == {"batchItemFailures": [{"itemIdentifier": "m9"}]}


def test_query_handler_raises_on_failure(monkeypatch):
    import query_handler
    with pytest.raises(ValueError):
        query_handler.lambda_handler({"question": ""}, None)


def test_s3v_store_requests_pass_botocore_validation(monkeypatch):
    """The store must send an index ARN botocore accepts (a bucket ARN is not a bucket name)."""
    import boto3
    from botocore.stub import Stubber

    from rag.stores import s3v as s3v_mod
    from rag.stores.base import Chunk

    arn = "arn:aws:s3vectors:us-west-2:123456789012:bucket/raga-smoke-vectorbucket-abc/index/idx-1024"
    real = boto3.client("s3vectors", region_name="us-west-2",
                        aws_access_key_id="x", aws_secret_access_key="x")
    stub = Stubber(real)
    stub.add_response("put_vectors", {}, {"indexArn": arn, "vectors": stub_any()})
    stub.add_response("query_vectors", {"vectors": [{"key": "d1#00000", "distance": 0.1,
                                                     "metadata": {"doc_id": "d1", "content": "c"}}],
                                        "distanceMetric": "cosine"}, None)
    monkeypatch.setattr(s3v_mod, "s3vectors", real)
    store = s3v_mod.S3VectorsStore(index_arn=arn)
    with stub:
        store.put([Chunk(key="d1#00000", embedding=[0.1] * 4, doc_id="d1", chunk_index=0,
                         content="c", source_key="scifact/d1.json")])
        hits = store.search([0.1] * 4, 5)
    assert hits[0].doc_id == "d1"


def stub_any():
    from botocore.stub import ANY
    return ANY
