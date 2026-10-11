#!/usr/bin/env python3
"""CDK entry point.

  cdk deploy RagCommon
  cdk deploy RagNetwork
  cdk deploy -c config=../configs/smoke.yaml -c overrides='{"ingest.concurrency": 4}' RagA-*

The harness (`rag deploy`) wraps these calls and records the outputs in the manifest.
"""
import json
import subprocess
import sys
from pathlib import Path

import aws_cdk as cdk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs"))

from rag_config import load_config  # noqa: E402
from stacks.common import RagCommonStack  # noqa: E402
from stacks.network import RagNetworkStack  # noqa: E402
from stacks.variant_a import VariantAStack  # noqa: E402


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


app = cdk.App()
env = cdk.Environment(account=app.node.try_get_context("account"), region="us-west-2")

RagCommonStack(app, "RagCommon", env=env)
network = RagNetworkStack(app, "RagNetwork", env=env)

config_path = app.node.try_get_context("config")
if config_path:
    overrides = json.loads(app.node.try_get_context("overrides") or "{}")
    cfg = load_config(config_path, overrides)
    VariantAStack(
        app, f"RagA-{cfg.run_slug()}",
        cfg=cfg, git_sha=app.node.try_get_context("git_sha") or git_sha(),
        vpc=network.vpc if cfg.network.in_vpc else None,
        lambda_sg=network.lambda_sg if cfg.network.in_vpc else None,
        env=env,
    )

app.synth()
