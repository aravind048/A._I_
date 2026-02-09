from __future__ import annotations
from pathlib import Path
from typing import Iterable
# import json
from corep_assistant.ingestion.models import TextChunk


def write_chunks_jsonl(chunks: Iterable[TextChunk], output_path: str) -> int:
    """
    Raw format:
      write_chunks_jsonl(chunks, output_path) -> int

    Writes one JSON per line (JSONL).

    Returns:
      number of chunks written
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
             # ✅ CHANGE: Pydantic handles datetime serialization
            f.write(c.model_dump_json() + "\n")
            count += 1

    return count
