from __future__ import annotations
from pathlib import Path
import json


def load_index_manifest(index_dir: str) -> dict:
    """
    load_index_manifest(index_dir: str) -> dict
    Reads index_manifest.json produced during indexing.
    """
    p = Path(index_dir) / "index_manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"index_manifest.json not found in {index_dir}")
    return json.loads(p.read_text(encoding="utf-8"))
