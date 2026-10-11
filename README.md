# rag-thesis

Experiment framework for *Building Serverless RAG Pipelines: An Investigation of Performance and Cost Implications* (MS thesis, UW Tacoma; advisor Wes Lloyd).

Every run is one YAML config → one CloudFormation stack → one manifest → raw results. Nothing is configured by hand in the console.

## Layout

| Path | What |
|---|---|
| `configs/rag_config/schema.py` | pydantic `RunConfig` (single source of truth) and sweep expansion |
| `configs/e1/` | Variant A configs and sweeps (ingest concurrency, threads, query load) |
| `infra/` | AWS CDK (Python): `RagCommon` (long-lived buckets), `RagNetwork` (per-session VPC + endpoints), `RagA-<run>` (one per run) |
| `lambdas/src/` | One code asset for all three functions: `enqueue_handler`, `embed_handler`, `query_handler`, shared `rag/` package, SAAF `Inspector` |
| `harness/rag_harness/` | `rag` CLI: deploy → ingest → query → quality → collect → cost → destroy |
| `eval/` | IR metrics (nDCG@10, Recall@k, MRR@10) |
| `results/manifests/` | One JSON manifest per run (committed) |
| `results/raw/` | SAAF records, Lambda REPORT lines, CloudWatch metrics (not committed; sync to the results bucket) |
| `analysis/` | Scripts that regenerate every thesis figure from `results/` |

## Variant A

`S3 (one object per BEIR doc) → Enqueue λ → SQS FIFO (group per doc) → Embed λ (8 threads) → Titan Text Embeddings V2 → S3 Vectors`
`Query λ → Titan V2 (same dims) → S3 Vectors top-k → Nova Lite v1 (in-region, streaming for TTFT)`

All Lambdas run in the `RagNetwork` VPC (no NAT; endpoints for Bedrock runtime, SQS, S3 Vectors; S3 gateway), so the network path is the same for every variant.

## Measurement rules built into the code
- botocore retries are off in the Lambdas, so every Bedrock throttle is counted (`embed_throttles`).
- Ingest and query embedding dimensions come from one config field.
- Cross-region inference profiles are rejected unless the config names the confound in `notes`.
- Every SAAF record carries `run_id`, `config_hash`, `git_sha`, boto3 version.
- Token counts (Titan input, Nova input/output) and per-stage wall times are recorded, so cost per doc/query is computed from data.
- Query failures raise (the load generator sees an error, not a 200).
- Non-smoke runs refuse to deploy from a dirty git tree.
- Each run gets a fresh vector index. After ingest, the stored vector count is checked against the expected chunk count.

## Usage

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[infra,analysis,dev]"
npm install -g aws-cdk          # CDK CLI
export AWS_PROFILE=...          # account where runs happen; region is fixed to us-west-2
cd infra && cdk bootstrap aws://<account>/us-west-2 && cdk deploy RagCommon && cd ..

rag prep-beir scifact           # prints the sha256 to pin in configs/e1/a_scifact.yaml
rag run configs/smoke.yaml      # 50 docs, 20 queries, end to end
rag run configs/e1/sweep_ingest_scifact.yaml --skip-query
rag network down                # when the session is over
```

Run steps individually with `rag deploy|ingest|query|quality|collect|cost|destroy`, and inspect a run with `rag show <run_id>`.

## Tests

```bash
pytest
```

SAAF `Inspector.py` is from [wlloyduw/SAAF](https://github.com/wlloyduw/SAAF) (W. Lloyd, W. Shu, R. Cordingly).
