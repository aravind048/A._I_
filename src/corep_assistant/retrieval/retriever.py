from __future__ import annotations
from typing import List
# import numpy as np

from corep_assistant.retrieval.embedder import Embedder
from corep_assistant.retrieval.vector_store import FaissVectorStore
from corep_assistant.retrieval.models import RetrievedChunk


class Retriever:
    """
    High-level API:
      query -> top-k RetrievedChunk

    Time complexity:
      - embed query: O(model_cost)
      - FAISS flat search: O(N * d) (for IndexFlatIP)
        For prototype sizes, this is fine.
    """

    def __init__(self, index_dir: str, embedder: Embedder):
        self.store = FaissVectorStore(index_dir=index_dir)
        self.embedder = embedder
        self._index = None
        self._records = None
        self._manifest = None

    def load(self) -> None:
        self._index, self._records, self._manifest = self.store.load()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if self._index is None:
            self.load()

        qvec = self.embedder.embed_query(query)  # (1, d)
        scores, idx = self.store.search(self._index, qvec, top_k)

        results: List[RetrievedChunk] = []
        for rank in range(top_k):
            i = int(idx[0, rank])
            if i < 0:
                continue
            score = float(scores[0, rank])
            rec = self._records[i]

            results.append(
                RetrievedChunk(
                    score=score,
                    source_id=rec.source_id,
                    chunk_id=rec.chunk_id,
                    text=rec.text,
                    start_char=rec.start_char,
                    end_char=rec.end_char,
                    section_hint=rec.section_hint,
                    extra={"rank": rank, "index_row": i}
                )
            )
        return results
