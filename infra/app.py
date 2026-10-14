#!/usr/bin/env python3
"""CDK entry point.

  cdk deploy RagCommon
  cdk deploy RagNetwork
  cdk deploy -c config=../configs/smoke.yaml RagA-*

The harness (`rag deploy`) wraps these calls and records the outputs in the manifest.
"""
import json
import sys
from pathlib import Path

import aws_cdk as cdk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs"))

from rag_config import load_config  # noqa: E402
from stacks.common import RagCommonStack  # noqa: E402
from stacks.network import RagNetworkStack  # noqa: E402
from stacks.variant_a import VariantAStack  # noqa: E402

app = cdk.App()
env = cdk.Environment(account=app.node.try_get_context("account"), region="us-west-2")

RagCommonStack(app, "RagCommon", env=env)
network = RagNetworkStack(app, "RagNetwork", env=env)

config_path = app.node.try_get_context("config")
if config_path:
    overrides = json.loads(app.node.try_get_context("overrides") or "{}")
    cfg = load_config(config_path, overrides)
    # One stack per infrastructure spec; runs that differ only in per-run settings share it.
    VariantAStack(
        app, cfg.infra_stack_name(),
        cfg=cfg,
        vpc=network.vpc if cfg.network.in_vpc else None,
        lambda_sg=network.lambda_sg if cfg.network.in_vpc else None,
        env=env,
    )

app.synth()
