from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config import TOP_K
from hf_llm import hf_llm


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore(documents):
    """Create a FAISS vector store from document chunks.

    Input:
        documents: List of LangChain Document objects.

    Output:
        FAISS vector store preserving each document's metadata.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(documents, embeddings)


def _format_context(documents) -> str:
    """Format retrieved documents with source information for the LLM."""
    formatted = []

    for document in documents:
        source = document.metadata.get("source_file", "Unknown source")
        page = document.metadata.get("page")
        chunk_id = document.metadata.get("chunk_id", "Unknown")

        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        formatted.append(
            f"[Source: {source}{page_text}, chunk {chunk_id}]\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


def _retrieve(vectorstore, question: str):
    return vectorstore.similarity_search(question, k=TOP_K)


def build_rag_chain(documents):
    """Build a grounded RAG chain.

    Input:
        documents: List of LangChain Document objects.

    Output:
        Runnable chain accepting a question and returning a grounded answer.
    """
    vectorstore = build_vectorstore(documents)
    retriever = RunnableLambda(lambda question: _retrieve(vectorstore, question))

    prompt = ChatPromptTemplate.from_template(
        """You are an enterprise research assistant.

Answer the user's question using ONLY the supplied evidence.

Rules:
- Do not invent facts.
- Do not use knowledge that is not present in the evidence.
- If the evidence is insufficient, say: "I could not find enough information in the provided sources to answer this confidently."
- Give a concise, useful research answer.
- When making a recommendation, explain the key evidence supporting it.
- Preserve the source labels included in the evidence.

Evidence:
{context}

Question:
{question}
"""
    )

    def prepare_input(question: str):
        documents = retriever.invoke(question)
        return {
            "context": _format_context(documents),
            "question": question,
        }

    return (
        RunnableLambda(prepare_input)
        | prompt
        | hf_llm
        | StrOutputParser()
    )
