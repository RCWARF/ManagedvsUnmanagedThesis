"""Stacks and per-run setup.

A Variant A stack (RagA-<infra key>) holds the Lambdas, queue, buckets and model profiles and
is shared by every run with the same infrastructure spec. Each run then gets:
  * a fresh S3 Vectors index in the stack's vector bucket, and
  * its settings applied to the functions from the RunConfig (environment, reserved
    concurrency, ESM batch size and scaling), which also forces fresh execution environments.
`destroy` removes the run's index. `teardown` deletes a stack in the background; RagCommon's
EventBridge rule reports when the delete finishes (`rag notify EMAIL`, or `rag stacks`).
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

from botocore.exceptions import ClientError

from . import manifest
from .aws import client, stack_outputs
from .paths import CONFIGS, INFRA

sys.path.insert(0, str(CONFIGS))
from rag_config import RunConfig  # noqa: E402

_deployed_this_session: set[str] = set()


def _cdk(args: list[str]) -> None:
    subprocess.run(["cdk", *args], cwd=INFRA, check=True, env=os.environ.copy())


def ensure_stack(cfg: RunConfig, config_path: str, overrides: dict) -> str:
    """cdk deploy the run's infra stack (a no-op when nothing changed), once per CLI session."""
    stack = cfg.infra_stack_name()
    if stack not in _deployed_this_session:
        try:
            status = client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]["StackStatus"]
        except ClientError:
            status = None
        if status == "DELETE_IN_PROGRESS":
            raise SystemExit(f"{stack} is still being deleted; wait until `rag stacks` shows it gone, then rerun")
        _cdk(["deploy", stack, "--require-approval", "never",
              "-c", f"config={os.path.abspath(config_path)}", "-c", f"overrides={json.dumps(overrides)}"])
        _deployed_this_session.add(stack)
    return stack


def _queue_depth(url: str) -> int:
    a = client("sqs").get_queue_attributes(QueueUrl=url, AttributeNames=[
        "ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
    return int(a["ApproximateNumberOfMessages"]) + int(a["ApproximateNumberOfMessagesNotVisible"])


def _delete_index(arn: str) -> None:
    try:
        client("s3vectors").delete_index(indexArn=arn)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NotFoundException":
            raise


def _create_index(cfg: RunConfig, out: dict) -> str:
    s3v = client("s3vectors")
    name = cfg.index_name()
    arn = f"{out['VectorBucket']}/index/{name}"
    _delete_index(arn)  # left over from an aborted attempt of the same run
    s3v.create_index(
        vectorBucketArn=out["VectorBucket"], indexName=name, dataType="float32",
        dimension=cfg.embedding.dimensions, distanceMetric=cfg.vector_store.distance,
        metadataConfiguration={"nonFilterableMetadataKeys": list(cfg.vector_store.non_filterable_keys)},
    )
    return s3v.get_index(indexArn=arn)["index"]["indexArn"]


def _set_env(fn: str, extra: dict) -> None:
    lam = client("lambda")
    current = lam.get_function_configuration(FunctionName=fn).get("Environment", {}).get("Variables", {})
    lam.update_function_configuration(FunctionName=fn, Environment={"Variables": current | extra})
    lam.get_waiter("function_updated_v2").wait(FunctionName=fn)


def _set_concurrency(cfg: RunConfig, out: dict) -> None:
    """Reserved concurrency = c; ESM MaximumConcurrency = c when c >= 2 (its minimum), unset at c = 1.
    The ESM cap may never exceed reserved concurrency, so the order depends on the direction."""
    lam = client("lambda")
    c = cfg.ingest.concurrency
    esm = out["EmbedEsm"]
    current_cap = lam.get_event_source_mapping(UUID=esm).get("ScalingConfig", {}).get("MaximumConcurrency", 0)
    scaling = {"MaximumConcurrency": c} if c >= 2 else {}

    def set_esm():
        lam.update_event_source_mapping(UUID=esm, BatchSize=cfg.ingest.esm_batch_size, ScalingConfig=scaling)
        for _ in range(90):
            if lam.get_event_source_mapping(UUID=esm)["State"] == "Enabled":
                return
            time.sleep(2)
        raise SystemExit("embed event source mapping did not return to Enabled")

    lowering = current_cap > c
    if lowering:
        set_esm()
    lam.put_function_concurrency(FunctionName=out["EmbedFn"], ReservedConcurrentExecutions=c)
    if not lowering:
        set_esm()


def deploy(cfg: RunConfig, config_path: str, overrides: dict) -> str:
    run_id = cfg.run_slug()
    sha = manifest.git_sha()
    if sha.endswith("-dirty") and cfg.experiment_id != "SMOKE":
        raise SystemExit("refusing a non-smoke run from a dirty working tree: commit first")
    stack = cfg.infra_stack_name()
    m = manifest.new(cfg, stack, sha)
    m["stack_shared"] = True
    m["phases"]["deploy_start"] = manifest.now()
    manifest.save(m)
    ensure_stack(cfg, config_path, overrides)
    out = stack_outputs(stack)
    manifest.update(run_id, phases={"stack_ready": manifest.now()})

    if _queue_depth(out["QueueUrl"]):
        raise SystemExit(f"ingest queue of {stack} is not empty (an earlier run is still processing or was "
                         f"aborted). Wait for it to drain, or: aws sqs purge-queue --queue-url {out['QueueUrl']}")
    out["VectorIndex"] = cfg.index_name()
    out["IndexArn"] = _create_index(cfg, out)
    env = cfg.run_env() | {"GIT_SHA": sha, "VECTOR_INDEX_ARN": out["IndexArn"]}
    for key in ("EnqueueFn", "EmbedFn", "QueryFn"):
        _set_env(out[key], env)
    _set_concurrency(cfg, out)
    esm = client("lambda").get_event_source_mapping(UUID=out["EmbedEsm"])

    manifest.update(
        run_id,
        outputs=out,
        dlq_depth_before=_queue_depth(out["DlqUrl"]),
        phases={"deploy_end": manifest.now()},
        quotas_live=manifest.snapshot_quotas(),
        quotas_expected=cfg.quotas.model_dump(),
        model_lifecycle=manifest.model_lifecycle([cfg.embedding.model_id, cfg.generation.model_id]),
        lambdas_deployed=manifest.lambda_configs(
            {"enqueue": out["EnqueueFn"], "embed": out["EmbedFn"], "query": out["QueryFn"]}),
        esm_deployed={k: esm.get(k) for k in ("BatchSize", "ScalingConfig", "MaximumBatchingWindowInSeconds")},
    )
    return run_id


def destroy(run_id: str, force: bool = False) -> None:
    """Delete the run's vector index. The shared stack stays for the next run (see teardown)."""
    m = manifest.load(run_id)
    if "collect_end" not in m["phases"] and not force:
        raise SystemExit(f"{run_id} has not been collected; run `rag collect {run_id}` first (or --force)")
    if m.get("stack_shared"):
        if m.get("outputs", {}).get("IndexArn"):
            _delete_index(m["outputs"]["IndexArn"])
    else:  # one stack per run, from before stacks were shared
        _delete_stack(m["stack_name"])
    manifest.update(run_id, phases={"destroyed": manifest.now()})


def _empty_vector_bucket(bucket_arn: str) -> None:
    s3v = client("s3vectors")
    token = None
    while True:
        resp = s3v.list_indexes(vectorBucketArn=bucket_arn, **({"nextToken": token} if token else {}))
        for idx in resp.get("indexes", []):
            _delete_index(idx["indexArn"])
        token = resp.get("nextToken")
        if not token:
            return


def _delete_stack(stack: str) -> None:
    client("cloudformation").delete_stack(StackName=stack)
    print(f"{stack}: delete started. It finishes in the background (VPC Lambdas often take 10-40 min).\n"
          "  `rag stacks` shows progress; with `rag notify EMAIL` set up you also get an email.")


def _rag_stacks():
    for page in client("cloudformation").get_paginator("list_stacks").paginate():
        yield from (s for s in page["StackSummaries"] if s["StackName"].startswith("Rag"))


def live_stacks() -> list[str]:
    return [s["StackName"] for s in _rag_stacks() if s["StackStatus"] != "DELETE_COMPLETE"]


def teardown(stack: str | None = None) -> None:
    """Start deleting one RagA-* stack (or all of them) and return without waiting."""
    stacks = [stack] if stack else [s for s in live_stacks() if s.startswith("RagA-")]
    if not stacks:
        print("no RagA-* stacks to delete")
    for name in stacks:
        try:
            out = stack_outputs(name)
        except ClientError:
            print(f"{name}: not found")
            continue
        if out.get("VectorBucket"):
            _empty_vector_bucket(out["VectorBucket"])  # CloudFormation cannot delete a bucket with indexes
        _delete_stack(name)


def stacks_status(hours: int = 24) -> list[dict]:
    """Rag* stacks that exist, plus those deleted in the last `hours`."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = []
    for s in _rag_stacks():
        if s["StackStatus"] == "DELETE_COMPLETE" and s.get("DeletionTime", since) < since:
            continue
        changed = s.get("DeletionTime") or s.get("LastUpdatedTime") or s["CreationTime"]
        rows.append({"stack": s["StackName"], "status": s["StackStatus"],
                     "changed_utc": changed.strftime("%Y-%m-%d %H:%M"), "reason": s.get("StackStatusReason", "")})
    return sorted(rows, key=lambda r: r["changed_utc"], reverse=True)


def notify(email: str) -> None:
    arn = stack_outputs("RagCommon").get("NotifyTopicArn")
    if not arn:
        raise SystemExit("RagCommon has no NotifyTopicArn output yet: `cd infra && cdk deploy RagCommon` first")
    client("sns").subscribe(TopicArn=arn, Protocol="email", Endpoint=email)
    print(f"AWS sent a confirmation email to {email}. Click its link; after that, every Rag* stack delete "
          "that finishes or fails sends you an email.")


def network(action: str) -> None:
    if action == "up":
        _cdk(["deploy", "RagNetwork", "--require-approval", "never"])
        return
    blocking = [s for s in live_stacks() if s.startswith("RagA-")]
    if blocking:
        raise SystemExit(f"these stacks still use the network: {', '.join(blocking)}. "
                         "Run `rag teardown` and wait until `rag stacks` shows them deleted.")
    _delete_stack("RagNetwork")
