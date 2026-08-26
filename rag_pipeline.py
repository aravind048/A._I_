from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config import TOP_K
from hf_llm import hf_llm


RESEARCH_PROMPT = ChatPromptTemplate.from_template(
    """You are an enterprise research assistant.

Answer the user's question using ONLY the supplied evidence.

Rules:
- Do not invent facts.
- Do not use knowledge that is not present in the evidence.
- Preserve the terminology and meaning used in the evidence.
- Do not infer a category that the evidence does not explicitly support.
- For classification questions, distinguish between programming languages, frameworks, libraries, APIs, platforms, databases, tools, and models instead of treating them as the same type of technology.
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


def _prepare_documents(vectorstore, question: str):
    """Retrieve evidence and return context text plus source metadata."""
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
    """Build a reusable retrieval + generation chain."""
    def prepare_input(question: str):
        return _prepare_documents(vectorstore, question)

    return (
        RunnableLambda(prepare_input)
        | RESEARCH_PROMPT
        | hf_llm
        | StrOutputParser()
    )


def research_with_sources(vectorstore, question: str):
    """Run retrieval + generation and return answer with source metadata."""
    prepared = _prepare_documents(vectorstore, question)

    answer = (
        RESEARCH_PROMPT
        | hf_llm
        | StrOutputParser()
    ).invoke({
        "context": prepared["context"],
        "question": prepared["question"],
    })

    return {
        "answer": answer,
        "sources": prepared["sources"],
    }
