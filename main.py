from fastapi import FastAPI, UploadFile
from rag_pipeline import build_rag_chain
from chunk_utils import load_and_split
from dotenv import load_dotenv
import traceback
import logging

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
rag_chain = None

@app.post("/upload/")
async def upload(file: UploadFile):
    try:
        contents = await file.read()
        with open("data/temp.pdf", "wb") as f:
            f.write(contents)

        docs = load_and_split("data/temp.pdf")
        global rag_chain
        rag_chain = build_rag_chain(docs)
        return {"status": "Document loaded and indexed"}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        return {"error": str(e)}

@app.get("/ask/")
async def ask(question: str):
    try:
        if not rag_chain:
            return {"error": "Upload a document first"}
        logger.info(f"Processing question: {question}")
        answer = rag_chain.invoke(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        logger.error(f"Ask error: {str(e)}\n{traceback.format_exc()}")
        return {"error": f"Internal server error: {str(e)}"}
