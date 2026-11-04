from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import rag_pipeline


class FakeVectorStore:
    def __init__(self, matches):
        self.matches = matches

    def similarity_search_with_score(self, question, k):
        return self.matches[:k]


def test_empty_retrieval_returns_insufficient_evidence():
    store = FakeVectorStore([])

    result = rag_pipeline.research_with_sources(store, "What is the policy?")

    assert result["answer"] == rag_pipeline.INSUFFICIENT_EVIDENCE
    assert result["sources"] == []


def test_retrieved_evidence_is_returned_with_source_metadata():
    document = SimpleNamespace(
        page_content="The company policy requires annual security training.",
        metadata={
            "source_file": "Security Policy.pdf",
            "page": 1,
            "chunk_id": 3,
        },
    )
    store = FakeVectorStore([(document, 0.42)])

    fake_model = RunnableLambda(
        lambda prompt: "The policy requires annual security training.")
    with patch.object(rag_pipeline, "hf_llm", fake_model):
        result = rag_pipeline.research_with_sources(
            store, "What training is required?"
        )

    assert "annual security training" in result["answer"]
    assert result["sources"] == [
        {
            "evidence_id": "E1",
            "file": "Security Policy.pdf",
            "page": 2,
            "chunk_id": 3,
            "retrieval_distance": 0.42,
        }
    ]


def test_valid_evidence_citation_is_formatted():
    sources = [
        {
            "evidence_id": "E1",
            "file": "Security Policy.pdf",
            "page": 2,
            "chunk_id": 3,
            "retrieval_distance": 0.42,
        }
    ]

    answer = "Annual training is mandatory [E1]."

    result = rag_pipeline._format_citations(answer, sources)

    assert result == (
        "Annual training is mandatory "
        "[Security Policy.pdf, page 2, chunk 3]."
    )


def test_unknown_evidence_citation_is_rejected():
    sources = [
        {
            "evidence_id": "E1",
            "file": "Security Policy.pdf",
            "page": 2,
            "chunk_id": 3,
            "retrieval_distance": 0.42,
        }
    ]

    answer = "The policy requires training [E99]."

    result = rag_pipeline._format_citations(answer, sources)

    assert "[unverified evidence reference]" in result
    assert "[E99]" not in result
