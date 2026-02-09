from __future__ import annotations
from typing import List
import json
from pathlib import Path
from corep_assistant.retrieval.models import ChunkRecord


def read_chunks_jsonl(chunks_path: str) -> List[ChunkRecord]:
    """
    Raw format:
      read_chunks_jsonl(chunks_path: str) -> List[ChunkRecord]
    """
    p = Path(chunks_path)
    if not p.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    records: List[ChunkRecord] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(ChunkRecord(**json.loads(line)))
    return records
