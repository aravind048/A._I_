import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from knowledge_base import KnowledgeBase
from rag_pipeline import build_research_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise AI Research Agent")
knowledge_base = KnowledgeBase()


class ResearchRequest(BaseModel):
    question: str


@app.post("/documents/")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF and add it to the persistent knowledge base."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / Path(file.filename).name

    try:
        with file_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        result = knowledge_base.ingest(str(file_path))
        return {"status": "Document indexed", **result}
    except Exception as exc:
        logger.exception("Document ingestion failed")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from exc


@app.post("/research/")
async def research(request: ResearchRequest):
    """Answer a research question from the persistent knowledge base."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    vectorstore = knowledge_base.load()
    if vectorstore is None:
        raise HTTPException(status_code=404, detail="No documents have been indexed yet.")

    try:
        chain = build_research_chain(vectorstore)
        answer = chain.invoke(question)
        return {"question": question, "answer": answer}
    except Exception as exc:
        logger.exception("Research request failed")
        raise HTTPException(status_code=500, detail="Research request failed.") from exc
