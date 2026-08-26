from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from config import TOP_K
from hf_llm import hf_llm


def _prepare_documents(vectorstore, question: str):
    """Retrieve evidence and return both context text and source metadata."""
    documents = vectorstore.similarity_search(question, k=TOP_K)
    context_parts = []
    sources = []

    for document in documents:
        source = document.metadata.get("source_file", "Unknown source")
        page = document.metadata.get("page")
        chunk_id = document.metadata.get("chunk_id", "Unknown")

        source_item = {
            "file": source,
            "page": page + 1 if isinstance(page, int) else None,
            "chunk_id": chunk_id,
        }
        if source_item not in sources:
            sources.append(source_item)

        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        context_parts.append(
            f"[Source: {source}{page_text}, chunk {chunk_id}]\n"
            f"{document.page_content}"
        )

    return {
        "context": "\n\n---\n\n".join(context_parts),
        "sources": sources,
        "question": question,
    }


def build_research_chain(vectorstore):
    """Build a research chain that returns an answer and its evidence sources."""
    prompt = ChatPromptTemplate.from_template(
        """You are an enterprise research assistant.

Answer the user's question using ONLY the supplied evidence.

Rules:
- Do not invent facts.
- Do not use knowledge that is not present in the evidence.
- If the evidence is insufficient, say: "I could not find enough information in the provided sources to answer this confidently."
- Give a concise, useful research answer.
- When making a recommendation, explain the key evidence supporting it.
- Do not cite sources that are not present in the supplied evidence.

Evidence:
{context}

Question:
{question}
"""
    )

    return (
        RunnableLambda(lambda question: _prepare_documents(vectorstore, question))
        | prompt
        | hf_llm
        | StrOutputParser()
    )


def research_with_sources(vectorstore, question: str):
    """Run retrieval + generation and return answer with retrieved sources.

    Input:
        vectorstore: Loaded FAISS vector store.
        question: User's research question.

    Output:
        Dictionary containing the generated answer and source metadata.
    """
    prepared = _prepare_documents(vectorstore, question)
    answer = (prompt_chain := (
        prompt
        | hf_llm
        | StrOutputParser()
    )).invoke({
        "context": prepared["context"],
        "question": question,
    })

    return {
        "answer": answer,
        "sources": prepared["sources"],
    }
