from __future__ import annotations
from typing import List, Optional
from corep_assistant.ingestion.models import RawDocument, TextChunk


def chunk_text(
    doc: RawDocument,
    chunk_size: int,
    chunk_overlap: int,
    section_hint: Optional[str] = None
) -> List[TextChunk]:
    """
    Raw format:
      chunk_text(doc, chunk_size, chunk_overlap, section_hint=None) -> List[TextChunk]

    Inputs:
      - doc.text: full text
      - chunk_size: max characters per chunk (e.g., 900)
      - chunk_overlap: overlapping characters between chunks (e.g., 150)

    Output:
      - list of TextChunk objects

    Logic:
      Sliding window over characters:
        start = 0
        end = start + chunk_size
        next start = end - chunk_overlap
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    text = doc.text
    n = len(text)
    chunks: List[TextChunk] = []

    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_str = text[start:end].strip()

        if chunk_str:
            chunk_id = TextChunk.make_chunk_id(doc.source_id, start, end, chunk_str)
            chunks.append(
                TextChunk(
                    source_id=doc.source_id,
                    chunk_id=chunk_id,
                    text=chunk_str,
                    start_char=start,
                    end_char=end,
                    section_hint=section_hint
                )
            )

        if end == n:
            break
        start = end - chunk_overlap

    return chunks


def chunk_documents(
    docs: List[RawDocument],
    chunk_size: int,
    chunk_overlap: int
) -> List[TextChunk]:
    """
    Raw format:
      chunk_documents(docs, chunk_size, chunk_overlap) -> List[TextChunk]

    Output:
      flat list of chunks for all docs
    """
    all_chunks: List[TextChunk] = []
    for d in docs:
        all_chunks.extend(chunk_text(d, chunk_size, chunk_overlap))
    return all_chunks
