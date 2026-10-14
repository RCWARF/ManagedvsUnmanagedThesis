"""`rag` command line.

  rag prep-beir scifact                 download, checksum, upload to the datasets bucket
  rag network up|down                   VPC + endpoints for a run session
  rag deploy CONFIG [--set k=v ...]     deploy/reuse the stack, fresh index, apply run settings
  rag ingest RUN_ID                     burst-upload docs, wait for drain, verify completeness
  rag query RUN_ID                      closed-loop query load with BEIR test queries
  rag quality RUN_ID                    nDCG@10 / Recall@k / MRR@10 on the run's index
  rag collect RUN_ID                    raw SAAF + REPORT + CloudWatch metrics -> results/raw
  rag cost RUN_ID                       cost estimate from collected records
  rag destroy RUN_ID                    delete the run's vector index (after collect)
  rag run CONFIG|SWEEP [--set k=v ...]  all of the above for each run, then start the teardown
                                        (--keep leaves the stack up for the next run)
  rag teardown [STACK]                  start deleting RagA-* stacks; returns without waiting
  rag stacks                            Rag* stacks and their status, incl. deletes in the last 24 h
  rag notify EMAIL                      email me when a Rag* stack delete finishes or fails
"""
import argparse
import json
import os
import sys

import yaml

from . import beir, collect, cost, deploy, ingest, manifest, quality, query_load
from .aws import client
from .paths import CONFIGS

sys.path.insert(0, str(CONFIGS))
from rag_config import load_config, load_sweep  # noqa: E402


def _overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def _datasets_bucket() -> str:
    if os.environ.get("DATASETS_BUCKET"):
        return os.environ["DATASETS_BUCKET"]
    from .aws import stack_outputs
    return stack_outputs("RagCommon")["DatasetsBucket"]


def cmd_prep_beir(a):
    d = beir.download(a.name)
    bucket = _datasets_bucket()
    s3 = client("s3")
    for rel in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        if (d / rel).exists():
            s3.upload_file(str(d / rel), bucket, f"beir/{a.name}/{rel}")
    sha = beir.sha256_file(d / "corpus.jsonl")
    n_docs = sum(1 for _ in (d / "corpus.jsonl").open())
    print(json.dumps({"dataset": a.name, "docs": n_docs, "sha256": sha, "bucket": bucket}, indent=2))
    print(f"\nPin it: set dataset.sha256: \"{sha}\" in configs/*/a_{a.name}.yaml")


def _one_run(cfg, config_path, overrides, skip_query=False):
    """One run on the shared stack. The stack is deployed once per CLI session and kept."""
    os.environ.setdefault("DATASETS_BUCKET", _datasets_bucket())
    run_id = deploy.deploy(cfg, config_path, overrides)
    try:
        print("ingest", ingest.run(run_id))
        if not skip_query:
            print("query", query_load.run(run_id))
            print("quality", quality.run(run_id))
    finally:
        print("collect", collect.run(run_id))
        print("cost", cost.estimate(run_id))
        deploy.destroy(run_id)
    return run_id


def cmd_run(a):
    os.environ.setdefault("DATASETS_BUCKET", _datasets_bucket())
    doc = yaml.safe_load(open(a.config))
    stacks = []
    try:
        if "base" in doc:  # sweep file
            from pathlib import Path
            base = str(Path(a.config).parent / doc["base"])
            for cfg in load_sweep(a.config):
                stacks.append(cfg.infra_stack_name())
                _one_run(cfg, base, _flatten_diff(cfg, base), a.skip_query)
        else:
            ov = _overrides(a.set)
            cfg = load_config(a.config, ov)
            stacks.append(cfg.infra_stack_name())
            _one_run(cfg, a.config, ov, a.skip_query)
    finally:
        if not a.keep:
            for stack in dict.fromkeys(stacks):
                deploy.teardown(stack)


def cmd_stacks(a):
    rows = deploy.stacks_status()
    if not rows:
        print("no Rag* stacks")
    for r in rows:
        print(f"{r['changed_utc']}  {r['status']:<20} {r['stack']}  {r['reason']}".rstrip())


def _flatten_diff(cfg, base_path) -> dict:
    """Dotted overrides that turn the base config into `cfg` (so CDK rebuilds the same config)."""
    base = load_config(base_path).model_dump(mode="json")
    target = cfg.model_dump(mode="json")
    out = {}

    def walk(b, t, prefix=""):
        for k, v in t.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict) and isinstance(b.get(k), dict):
                walk(b[k], v, key + ".")
            elif b.get(k) != v:
                out[key] = v
    walk(base, target)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep-beir"); s.add_argument("name"); s.set_defaults(f=cmd_prep_beir)
    s = sub.add_parser("network"); s.add_argument("action", choices=["up", "down"])
    s.set_defaults(f=lambda a: deploy.network(a.action))
    s = sub.add_parser("deploy"); s.add_argument("config"); s.add_argument("--set", nargs="*")
    s.set_defaults(f=lambda a: print(deploy.deploy(load_config(a.config, _overrides(a.set)), a.config, _overrides(a.set))))
    for name, fn in (("ingest", ingest.run), ("query", query_load.run), ("quality", quality.run),
                     ("collect", collect.run), ("cost", cost.estimate)):
        s = sub.add_parser(name); s.add_argument("run_id")
        s.set_defaults(f=lambda a, fn=fn: print(json.dumps(fn(a.run_id), indent=2, default=str)))
    s = sub.add_parser("destroy"); s.add_argument("run_id"); s.add_argument("--force", action="store_true")
    s.set_defaults(f=lambda a: deploy.destroy(a.run_id, a.force))
    s = sub.add_parser("run"); s.add_argument("config"); s.add_argument("--set", nargs="*")
    s.add_argument("--skip-query", action="store_true")
    s.add_argument("--keep", action="store_true", help="leave the stack up for the next rag run")
    s.set_defaults(f=cmd_run)
    s = sub.add_parser("teardown"); s.add_argument("stack", nargs="?")
    s.set_defaults(f=lambda a: deploy.teardown(a.stack))
    s = sub.add_parser("stacks"); s.set_defaults(f=cmd_stacks)
    s = sub.add_parser("notify"); s.add_argument("email"); s.set_defaults(f=lambda a: deploy.notify(a.email))
    s = sub.add_parser("show"); s.add_argument("run_id")
    s.set_defaults(f=lambda a: print(json.dumps(manifest.load(a.run_id), indent=2)))
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
