from pathlib import Path

from chunk_utils import load_and_split
from vector_store import VectorStoreService


class KnowledgeBase:
    """Coordinate document ingestion and persistent vector indexing."""

    def __init__(self):
        self.vector_store = VectorStoreService()

    def ingest(self, file_path: str):
        """Process a document and add its chunks to the knowledge base."""
        documents = load_and_split(file_path)
        existing_store = self.vector_store.load()

        if existing_store is None:
            self.vector_store.build(documents)
        else:
            self.vector_store.add_documents(existing_store, documents)

        return {
            "file": Path(file_path).name,
            "chunks_added": len(documents),
        }

    def load(self):
        return self.vector_store.load()
