from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = Path("data/vectorstore")


class VectorStoreService:
    """Manage the local FAISS knowledge base."""

    def __init__(self, index_path: str | Path = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def build(self, documents):
        """Create a vector store from document chunks and persist it."""
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        self._save(vectorstore)
        return vectorstore

    def add_documents(self, vectorstore, documents):
        """Add new document chunks to an existing knowledge base."""
        vectorstore.add_documents(documents)
        self._save(vectorstore)
        return vectorstore

    def load(self):
        """Load the persisted knowledge base, or return None if absent."""
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "index.pkl"

        if not index_file.exists() or not metadata_file.exists():
            return None

        return FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _save(self, vectorstore):
        self.index_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.index_path))
