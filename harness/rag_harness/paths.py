from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "infra"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
MANIFESTS = RESULTS / "manifests"
RAW = RESULTS / "raw"
DATA_CACHE = ROOT / ".data"
REGION = "us-west-2"
