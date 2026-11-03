import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from chunk_utils import load_and_split
from vector_store import VectorStoreService


class KnowledgeBase:
    """Coordinate document ingestion and persistent knowledge-base metadata."""

    def __init__(self):
        self.vector_store = VectorStoreService()
        self.data_dir = Path("data")
        self.upload_dir = self.data_dir / "uploads"
        self.manifest_path = self.data_dir / "knowledge_base" / "documents.json"

    def _read_manifest(self):
        if not self.manifest_path.exists():
            return []
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _write_manifest(self, documents):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(documents, indent=2), encoding="utf-8"
        )

    def ingest(self, file_path: str):
        """Split, index, and register one uploaded PDF."""
        path = Path(file_path)
        documents = load_and_split(str(path))
        if not documents:
            raise ValueError("The PDF contains no extractable text.")

        existing_store = self.vector_store.load()
        if existing_store is None:
            self.vector_store.build(documents)
        else:
            self.vector_store.add_documents(existing_store, documents)

        registry = self._read_manifest()
        record = {
            "document_id": str(uuid4()),
            "file": path.name,
            "chunks": len(documents),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        registry.append(record)
        self._write_manifest(registry)
        return record

    def list_documents(self):
        """Return registered documents and detect an unregistered legacy index."""
        return {
            "documents": self._read_manifest(),
            "legacy_index_present": (
                self.vector_store.load() is not None
                and not self._read_manifest()
            ),
        }

    def clear(self):
        """Remove the local vector index, uploaded PDFs, and document registry."""
        self.vector_store.clear()
        if self.upload_dir.exists():
            for path in self.upload_dir.iterdir():
                if path.is_file():
                    path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        return {"status": "Knowledge base cleared"}
    
    def load(self):
        return self.vector_store.load()
