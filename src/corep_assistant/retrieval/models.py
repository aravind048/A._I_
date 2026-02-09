from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, Dict, Any


class ChunkRecord(BaseModel):
    """
    Represents one chunk loaded from JSONL.

    Input JSON (from ingestion output):
      - source_id, chunk_id, text, start_char, end_char, section_hint, created_at
    """
    source_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    section_hint: Optional[str] = None
    created_at: Optional[str] = None  # stored as ISO string in JSONL (OK for prototype)


class RetrievedChunk(BaseModel):
    """
    What retrieval returns per match.

    score: similarity score (higher means more similar)
    """
    score: float
    source_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    section_hint: Optional[str] = None
    extra: Dict[str, Any] = {}
