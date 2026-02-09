from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
import hashlib


class RawDocument(BaseModel):
    """
    Represents a loaded document.

    Input:
      - source_id: stable identifier (e.g., filename without extension)
      - text: full document content
    Output:
      - validated object
    """
    source_id: str
    text: str


class TextChunk(BaseModel):
    """
    A chunk of text with traceability + stable id.

    chunk_id is stable given (source_id + start/end + text hash).
    """
    source_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    section_hint: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda:datetime.now(timezone.utc))

    @staticmethod
    def make_chunk_id(source_id: str, start_char: int, end_char: int, text: str) -> str:
        """
        Raw format:
          make_chunk_id(source_id, start_char, end_char, text) -> str

        Logic:
          We hash content + span so ids stay stable across runs as long as text is unchanged.
        """
        h = hashlib.sha256()
        h.update(source_id.encode("utf-8"))
        h.update(f"{start_char}:{end_char}".encode("utf-8"))
        h.update(text.encode("utf-8"))
        return h.hexdigest()[:16]
