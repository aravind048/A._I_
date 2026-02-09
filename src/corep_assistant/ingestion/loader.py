from __future__ import annotations
from pathlib import Path
from typing import List
from corep_assistant.ingestion.models import RawDocument


def load_txt_documents(input_dir: str) -> List[RawDocument]:
    """
    Raw format:
      load_txt_documents(input_dir: str) -> List[RawDocument]

    Input:
      - input_dir: folder containing .txt files
    Output:
      - list of RawDocument objects
    """
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    docs: List[RawDocument] = []
    for file in sorted(p.glob("*.txt")):
        text = file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        source_id = file.stem  # filename without extension
        docs.append(RawDocument(source_id=source_id, text=text))

    return docs
