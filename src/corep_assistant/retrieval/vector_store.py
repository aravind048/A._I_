from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np
import faiss
from datetime import datetime, timezone

from corep_assistant.retrieval.models import ChunkRecord
import hashlib # for generating stable IDs from content

class FaissVectorStore:
    """
    FAISS vector store with sidecar metadata.

    Files in index_dir:
      - faiss.index
      - meta.jsonl
      - index_manifest.json

    Similarity:
      - using inner product on normalized embeddings == cosine similarity
    """

    INDEX_FILE = "faiss.index"
    META_FILE = "meta.jsonl"
    MANIFEST_FILE = "index_manifest.json"

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.index_dir / self.INDEX_FILE

    @property
    def meta_path(self) -> Path:
        return self.index_dir / self.META_FILE

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / self.MANIFEST_FILE

    def build_and_save(
        self,
        embeddings: np.ndarray,
        records: List[ChunkRecord],
        embed_model_name: str
    ) -> None:
        """
        Raw format:
          build_and_save(embeddings: np.ndarray, records: List[ChunkRecord], embed_model_name: str) -> None

        Preconditions:
          - embeddings shape == (n, d)
          - len(records) == n

        Writes:
          - FAISS index file
          - meta.jsonl aligned with vector row ids
          - manifest with metadata
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D array (n, d)")
        n, d = embeddings.shape
        if len(records) != n:
            raise ValueError(f"records length ({len(records)}) must match embeddings rows ({n})")

        # FAISS IndexFlatIP for inner-product (cosine on normalized vectors)
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))

        # Write metadata aligned with FAISS row order
        with self.meta_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
        
        # ✅ NEW: stable fingerprint from (source_id, chunk_id) list
        h = hashlib.sha256()
        for r in records:
            h.update(r.source_id.encode("utf-8"))
            h.update(b"::")
            h.update(r.chunk_id.encode("utf-8"))
            h.update(b"\n")
        meta_fingerprint = h.hexdigest()[:16]

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embed_model_name": embed_model_name,
            "num_vectors": n,
            "dimension": d,
            "index_type": "IndexFlatIP",
            "normalized_embeddings": True,

            # ✅ NEW: artifact pointers
            "files": {
                "index": self.index_path.name,
                "metadata": self.meta_path.name
        },
        
            # ✅ NEW: reproducibility/debug helpers
            "index_dir": str(self.index_dir),
            "meta_record_count": len(records),
            "meta_fingerprint": meta_fingerprint
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load(self) -> Tuple[faiss.Index, List[ChunkRecord], dict]:
        """
        Raw format:
          load() -> (faiss_index, records, manifest_dict)
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.meta_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        index = faiss.read_index(str(self.index_path))

        records: List[ChunkRecord] = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(ChunkRecord(**json.loads(line)))

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return index, records, manifest

    def search(
        self,
        index: faiss.Index,
        query_vec: np.ndarray,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Raw format:
          search(index, query_vec, top_k) -> (scores, indices)

        query_vec:
          shape (1, d), float32, normalized

        Output:
          scores shape (1, top_k)
          indices shape (1, top_k), -1 if not found
        """
        if query_vec.ndim != 2 or query_vec.shape[0] != 1:
            raise ValueError("query_vec must have shape (1, d)")
        scores, idx = index.search(query_vec, top_k)
        return scores, idx
