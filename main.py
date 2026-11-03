import logging
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from knowledge_base import KnowledgeBase
from rag_pipeline import research_with_sources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description="Evidence-grounded research and analysis across enterprise documents.",
    version="0.2.0",
)
knowledge_base = KnowledgeBase()


class ResearchRequest(BaseModel):
    question: str


@app.get("/health/")
async def health():
    return {"status": "healthy", "service": "enterprise-knowledge-intelligence"}


@app.post("/documents/")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a PDF in the persistent knowledge base."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    file_path = upload_dir / f"{uuid4().hex}_{safe_name}"

    try:
        with file_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        result = knowledge_base.ingest(str(file_path), display_name=safe_name)
        return {"status": "Document indexed", **result}
    except ValueError as exc:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document ingestion failed")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from exc
    finally:
        await file.close()


@app.get("/documents/")
async def list_documents():
    """List documents registered in the knowledge base."""
    try:
        return knowledge_base.list_documents()
    except Exception as exc:
        logger.exception("Unable to list knowledge-base documents")
        raise HTTPException(status_code=500, detail="Unable to list documents.") from exc


@app.delete("/knowledge-base/")
async def clear_knowledge_base():
    """Clear all indexed documents and locally stored uploaded PDFs."""
    try:
        return knowledge_base.clear()
    except Exception as exc:
        logger.exception("Unable to clear knowledge base")
        raise HTTPException(status_code=500, detail="Unable to clear knowledge base.") from exc


@app.post("/research/")
async def research(request: ResearchRequest):
    """Answer a question and return the evidence sources used."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        vectorstore = knowledge_base.load()
        if vectorstore is None:
            raise HTTPException(
                status_code=404,
                detail="No documents have been indexed. Upload documents first.",
            )

        result = research_with_sources(vectorstore, question)
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Research request failed")
        raise HTTPException(status_code=500, detail="Research request failed.") from exc
