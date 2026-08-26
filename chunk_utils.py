from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def load_and_split(file_path: str):
    """Load a PDF and split it into searchable chunks with source metadata.

    Input:
        file_path: Path to a PDF document.

    Output:
        List of LangChain Document objects containing chunk text and metadata.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["source_file"] = file_path

    return chunks
