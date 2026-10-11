import json
import os
import subprocess
import sys

from . import manifest
from .aws import stack_outputs
from .paths import CONFIGS, INFRA

sys.path.insert(0, str(CONFIGS))
from rag_config import RunConfig  # noqa: E402


def _cdk(args: list[str]) -> None:
    subprocess.run(["cdk", *args], cwd=INFRA, check=True, env=os.environ.copy())


def deploy(cfg: RunConfig, config_path: str, overrides: dict) -> str:
    run_id = cfg.run_slug()
    stack = f"RagA-{run_id}"
    sha = manifest.git_sha()
    if sha.endswith("-dirty") and cfg.experiment_id != "SMOKE":
        raise SystemExit("refusing a non-smoke run from a dirty working tree: commit first")
    m = manifest.new(cfg, stack, sha)
    m["phases"]["deploy_start"] = manifest.now()
    manifest.save(m)
    _cdk(["deploy", stack, "--require-approval", "never",
          "-c", f"config={os.path.abspath(config_path)}",
          "-c", f"overrides={json.dumps(overrides)}", "-c", f"git_sha={sha}"])
    outputs = stack_outputs(stack)
    manifest.update(
        run_id,
        outputs=outputs,
        phases={"deploy_end": manifest.now()},
        quotas_live=manifest.snapshot_quotas(),
        quotas_expected=cfg.quotas.model_dump(),
        model_lifecycle=manifest.model_lifecycle([cfg.embedding.model_id, cfg.generation.model_id]),
        lambdas_deployed=manifest.lambda_configs(
            {"enqueue": outputs["EnqueueFn"], "embed": outputs["EmbedFn"], "query": outputs["QueryFn"]}),
    )
    return run_id


def destroy(run_id: str, force: bool = False) -> None:
    m = manifest.load(run_id)
    if "collect_end" not in m["phases"] and not force:
        raise SystemExit(f"{run_id} has not been collected; run `rag collect {run_id}` first (or --force)")
    cfgp = m["config"]
    # Re-synthesize from the stored config so destroy targets exactly this stack.
    tmp = INFRA / "cdk.out.destroy.json"
    tmp.write_text(json.dumps(cfgp))
    try:
        _cdk(["destroy", m["stack_name"], "--force", "-c", f"config={tmp}"])
    finally:
        tmp.unlink(missing_ok=True)
    manifest.update(run_id, phases={"destroyed": manifest.now()})


def network(action: str) -> None:
    _cdk(["deploy" if action == "up" else "destroy", "RagNetwork", "--require-approval", "never"]
         + (["--force"] if action == "down" else []))
