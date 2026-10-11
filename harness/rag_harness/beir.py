"""BEIR download, local cache, checksum and upload to the datasets bucket."""
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from .paths import DATA_CACHE

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str) -> Path:
    """Fetch and unzip one BEIR dataset into .data/beir/<name>/ (skipped if present)."""
    target = DATA_CACHE / "beir" / name
    if (target / "corpus.jsonl").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BEIR_URL.format(name=name)) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            rel = member.split("/", 1)[1] if "/" in member else ""
            if not rel or member.endswith("/"):
                continue
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(member))
    return target


def iter_corpus(path: Path, limit: int | None = None):
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)


def load_queries(path: Path) -> dict[str, str]:
    return {q["_id"]: q["text"] for q in map(json.loads, path.open())}


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def local_dir(name: str) -> Path:
    return DATA_CACHE / "beir" / name
