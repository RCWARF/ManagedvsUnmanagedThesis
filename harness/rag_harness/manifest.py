"""Run manifest: one JSON file per run, updated by every harness phase."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

from .aws import client
from .paths import CONFIGS, MANIFESTS, ROOT

sys.path.insert(0, str(CONFIGS))
from rag_config import RunConfig  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def path_for(run_id: str):
    return MANIFESTS / f"{run_id}.json"


def load(run_id: str) -> dict:
    return json.loads(path_for(run_id).read_text())


def save(m: dict) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path_for(m["run_id"]).write_text(json.dumps(m, indent=2, sort_keys=True, default=str))


def update(run_id: str, **fields) -> dict:
    m = load(run_id)
    for k, v in fields.items():
        if isinstance(v, dict) and isinstance(m.get(k), dict):
            m[k].update(v)
        else:
            m[k] = v
    save(m)
    return m


def snapshot_quotas() -> dict:
    """Live quota values, so every manifest records what was actually in force."""
    out = {}
    sq = client("service-quotas")
    wanted = ("titan text embeddings v2", "nova lite")
    try:
        for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="bedrock"):
            for q in page["Quotas"]:
                name = q["QuotaName"].lower()
                if any(w in name for w in wanted) and "on-demand" in name:
                    out[q["QuotaName"]] = q["Value"]
        lam = sq.get_service_quota(ServiceCode="lambda", QuotaCode="L-B99A9384")["Quota"]
        out["Lambda concurrent executions"] = lam["Value"]
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def model_lifecycle(model_ids: list[str]) -> dict:
    br = client("bedrock")
    out = {}
    for mid in model_ids:
        try:
            out[mid] = br.get_foundation_model(modelIdentifier=mid)["modelDetails"].get("modelLifecycle")
        except Exception as exc:  # noqa: BLE001
            out[mid] = {"error": str(exc)}
    return out


def lambda_configs(fn_names: dict[str, str]) -> dict:
    lam = client("lambda")
    out = {}
    for role, name in fn_names.items():
        c = lam.get_function(FunctionName=name)
        conf = c["Configuration"]
        out[role] = {
            "memory_mb": conf["MemorySize"], "timeout_s": conf["Timeout"],
            "architectures": conf.get("Architectures"), "runtime": conf.get("Runtime"),
            "runtime_version_arn": conf.get("RuntimeVersionConfig", {}).get("RuntimeVersionArn"),
            "code_sha256": conf.get("CodeSha256"),
            "reserved_concurrency": c.get("Concurrency", {}).get("ReservedConcurrentExecutions"),
            "vpc": bool(conf.get("VpcConfig", {}).get("SubnetIds")),
        }
    return out


def new(cfg: RunConfig, stack_name: str, sha: str) -> dict:
    return {
        "run_id": cfg.run_slug(),
        "stack_name": stack_name,
        "experiment_id": cfg.experiment_id,
        "variant_id": cfg.variant_id,
        "repetition": cfg.repetition,
        "config_hash": cfg.config_hash(),
        "config": cfg.model_dump(mode="json"),
        "git_sha": sha,
        "region": cfg.region,
        "tools": {"python": platform.python_version(), "harness_platform": platform.platform()},
        "phases": {},
        "notes": cfg.notes,
    }
