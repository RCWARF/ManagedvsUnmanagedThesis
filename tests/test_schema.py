from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_config import load_config, load_sweep

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_example_configs_validate():
    for path in (CONFIGS / "e1").glob("a_*.yaml"):
        cfg = load_config(path)
        assert cfg.region == "us-west-2"
        assert cfg.dataset.corpus_uri.startswith("s3://test-bucket/")


def test_cross_region_model_rejected():
    with pytest.raises(ValidationError, match="cross-region"):
        load_config(CONFIGS / "smoke.yaml", {"generation.model_id": "us.amazon.nova-lite-v1:0"})


def test_cross_region_allowed_only_with_note():
    with pytest.raises(ValidationError):
        load_config(CONFIGS / "smoke.yaml", {"generation.model_id": "us.amazon.nova-lite-v1:0",
                                             "generation.allow_cross_region": True})
    cfg = load_config(CONFIGS / "smoke.yaml", {"generation.model_id": "us.amazon.nova-lite-v1:0",
                                               "generation.allow_cross_region": True,
                                               "notes": "cross-region profile, labelled confound"})
    assert cfg.generation.allow_cross_region


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        load_config(CONFIGS / "smoke.yaml", {"embedding.dimension": 512})  # typo of `dimensions`


def test_hash_changes_with_any_field_and_is_stable():
    a = load_config(CONFIGS / "smoke.yaml")
    b = load_config(CONFIGS / "smoke.yaml")
    c = load_config(CONFIGS / "smoke.yaml", {"ingest.concurrency": 2})
    assert a.config_hash() == b.config_hash() != c.config_hash()
    assert a.run_slug() != c.run_slug()


def test_put_batch_limit():
    with pytest.raises(ValidationError):
        load_config(CONFIGS / "smoke.yaml", {"vector_store.put_batch_size": 501})


def test_sweep_expands_grid_times_reps():
    runs = load_sweep(CONFIGS / "e1" / "sweep_ingest_scifact.yaml")
    assert len(runs) == 6 * 3
    assert len({r.run_slug() for r in runs}) == 18
    threads = load_sweep(CONFIGS / "e1" / "sweep_threads_scifact.yaml")
    assert {r.embedding.workers_per_invocation for r in threads} == {1, 2, 4, 8}


def test_sweeps_share_one_stack_but_get_their_own_index():
    import re
    for sweep in ("sweep_ingest_scifact.yaml", "sweep_threads_scifact.yaml", "sweep_query_scifact.yaml"):
        runs = load_sweep(CONFIGS / "e1" / sweep)
        assert len({r.infra_stack_name() for r in runs}) == 1, sweep
        names = {r.index_name() for r in runs}
        assert len(names) == len(runs)
        assert all(re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", n) for n in names)
    base = load_config(CONFIGS / "smoke.yaml")
    assert base.infra_hash() != load_config(CONFIGS / "smoke.yaml", {"lambdas.embed": {"memory_mb": 3008, "timeout_s": 900}}).infra_hash()
    assert base.infra_hash() == load_config(CONFIGS / "smoke.yaml", {"embedding.dimensions": 256}).infra_hash()


def test_run_env_carries_every_per_run_setting():
    cfg = load_config(CONFIGS / "smoke.yaml", {"embedding.workers_per_invocation": 4, "embedding.dimensions": 512})
    env = cfg.run_env()
    assert env["RUN_ID"] == cfg.run_slug() and env["CONFIG_HASH"] == cfg.config_hash()
    assert env["EMBED_WORKERS"] == "4" and env["EMBED_DIMS"] == "512"
    assert all(isinstance(v, str) for v in env.values())
