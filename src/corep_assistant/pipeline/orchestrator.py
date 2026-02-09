from __future__ import annotations
from typing import Dict, Any

from corep_assistant.retrieval.embedder import Embedder
from corep_assistant.retrieval.retriever import Retriever

# ✅ CHANGE (use real client with MOCK fallback)
from corep_assistant.llm.client import LLMClient
from corep_assistant.reporting.validations import validate_ca1
from corep_assistant.reporting.renderer import render_template_extract
from corep_assistant.reporting.audit import build_audit_log
from corep_assistant.reporting.citation_checks import check_citations, citation_coverage
from corep_assistant.config import settings
from corep_assistant.retrieval.manifest import load_index_manifest


def run_ca1_pipeline(
    question: str,
    scenario: Dict[str, Any],
    index_dir: str,
    embed_model_name: str,
    top_k: int
) -> Dict[str, Any]:
    """
    run_ca1_pipeline(question, scenario, index_dir, embed_model_name, top_k) -> Dict[str, Any]

    Flow:
      1) retrieve relevant chunks
      2) structured generation via LLMClient (MOCK or HF_ROUTER)
      3) validate computations/signs
      4) validate citations + compute citation coverage
      5) render template extract
      6) build audit log
    """

    # 1) retrieval
    embedder = Embedder(model_name=embed_model_name)
    retriever = Retriever(index_dir=index_dir, embedder=embedder)
    retrieved = retriever.retrieve(question, top_k=top_k)

    # 2) LLM structured generation (MOCK or HF_ROUTER)
    llm = LLMClient()
    resp = llm.generate_ca1(
        question=question, scenario=scenario, retrieved=retrieved)

    # 3) numeric validations (guardrails)
    resp.validation = validate_ca1(resp)

    # 4) citation validations (defense in depth, even if LLM enforces it)
    retrieved_ids = [{"source_id": r.source_id, "chunk_id": r.chunk_id}
                     for r in retrieved]  # ✅ CHANGE (leaner)
    cite_issues = check_citations(
        resp=resp,
        retrieved=retrieved_ids,
        require_per_field=settings.require_per_field_citations
    )
    resp.validation.extend(cite_issues)
    resp.debug["citation_coverage"] = citation_coverage(resp)
    resp.debug["llm_mode"] = settings.llm_mode  # ✅ NEW (useful for demo)

    # Optional: hard gate (prototype policy)
    # If you want the pipeline to FAIL instead of returning weak audit trail:
    # if resp.debug["citation_coverage"] < settings.min_citation_coverage:
    #     raise ValueError("Citation coverage below minimum threshold.")

    # 5) render + audit
    template_rows = render_template_extract(resp)
    audit_log = build_audit_log(resp)
    manifest = load_index_manifest(index_dir)

    return {
        "llm_mode": settings.llm_mode,  # ✅ NEW
        "retrieved": [r.model_dump() for r in retrieved],
        "structured": resp.model_dump(),
        "template_extract": template_rows,
        "validation": [v.model_dump() for v in resp.validation],
        "audit_log": audit_log,
        "manifest_snapshot": manifest,

    }
