from __future__ import annotations
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class Embedder:
    """
    Thin wrapper to standardize embedding output and settings.

    Time complexity:
      - embedding M texts: O(M * model_cost)

    Space:
      - embeddings: O(M * d)
    """

    def __init__(self, model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Raw format:
          embed_texts(texts: List[str], batch_size=32) -> np.ndarray shape (n, d)

        Output:
          float32 normalized embeddings (for cosine similarity via inner product)
        """
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return np.asarray(emb, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Raw format:
          embed_query(query: str) -> np.ndarray shape (1, d)
        """
        emb = self.model.encode([query], normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)
