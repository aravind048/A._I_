from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from config import TOP_K
from hf_llm import hf_llm


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


def build_research_chain(vectorstore):
    """Build a research chain using an existing persistent vector store.

    Input:
        vectorstore: Loaded FAISS vector store.

    Output:
        Runnable chain accepting a research question and returning an answer.
    """
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
        documents = vectorstore.similarity_search(question, k=TOP_K)
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
