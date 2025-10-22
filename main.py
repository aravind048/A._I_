from fastapi import FastAPI, UploadFile
from rag_pipeline import build_rag_chain
from chunk_utils import load_and_split
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
rag_chain = None

@app.post("/upload/")
async def upload(file: UploadFile):
    contents = await file.read()
    with open("data/temp.pdf", "wb") as f:
        f.write(contents)

    docs = load_and_split("data/temp.pdf")
    global rag_chain
    rag_chain = build_rag_chain(docs)
    return {"status": "Document loaded and indexed"}

@app.get("/ask/")
async def ask(question: str):
    if not rag_chain:
        return {"error": "Upload a document first"}
    answer = rag_chain.invoke(question)
    return {"question": question, "answer": answer}
