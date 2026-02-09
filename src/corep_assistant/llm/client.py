from __future__ import annotations
from typing import Dict, Any, List
import requests

from corep_assistant.config import settings
from corep_assistant.llm.parser import extract_json_object
from corep_assistant.llm.schema import CorepResponse
from corep_assistant.retrieval.models import RetrievedChunk
from corep_assistant.llm.prompt import build_ca1_prompt
from corep_assistant.llm.mock_llm import MockLLM
from corep_assistant.reporting.citation_checks import check_citations, citation_coverage
from corep_assistant.config import settings
from corep_assistant.reporting.citation_checks import check_citations, citation_coverage


class LLMClient:
    """
    LLM client supporting:
      - MOCK
      - HF_ROUTER (HuggingFace router endpoint)

    Raw format:
      generate_ca1(question: str, scenario: dict, retrieved: List[RetrievedChunk]) -> CorepResponse
    """

    def __init__(self):
        self.mode = settings.llm_mode.upper()

    def generate_ca1(self, question: str, scenario: Dict[str, Any], retrieved: List[RetrievedChunk]) -> CorepResponse:
        if self.mode == "MOCK":
            # Build base response and use deterministic generator
            base = CorepResponse(
                template_id="COREP_OWN_FUNDS_CA1",
                reporting_date=scenario.get("reporting_date", "YYYY-MM-DD"),
                currency=scenario.get("currency", "GBP"),
                entity=scenario["entity"],
                inputs=scenario["inputs"],
                computed={},  # filled
            )
            return MockLLM().generate_ca1(question=question, base_response=base, retrieved=retrieved)

        if self.mode == "HF_ROUTER":
            return self._generate_ca1_hf_router(question, scenario, retrieved)

        raise ValueError(f"Unsupported LLM_MODE: {settings.llm_mode}")

    def _generate_ca1_hf_router(self, question: str, scenario: Dict[str, Any], retrieved: List[RetrievedChunk]) -> CorepResponse:
        if not settings.hf_api_key:
            raise ValueError("HF_API_KEY is missing. Set it in .env")

        prompt = build_ca1_prompt(
            question=question, scenario=scenario, retrieved=retrieved)

        payload = {
            "model": settings.hf_model,
            "messages": [
                {"role": "system", "content": "You output ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {settings.hf_api_key}",
            "Content-Type": "application/json",
        }

        # ✅ NEW: Allowed citations derived from retrieved evidence (stable contract)
        retrieved_ids = [{"source_id": r.source_id,
                        "chunk_id": r.chunk_id} for r in retrieved]

        last_err = None
        for attempt in range(settings.llm_retries + 1):
            try:
                resp_http = requests.post(
                    settings.hf_router_url, json=payload, headers=headers, timeout=60)
                resp_http.raise_for_status()
                data = resp_http.json()

                content = data["choices"][0]["message"]["content"]
                obj = extract_json_object(content)

                # 1) Schema enforcement
                resp_model = CorepResponse.model_validate(obj)

                # ✅ NEW: Citation enforcement
                cite_issues = check_citations(
                    resp=resp_model,
                    retrieved=retrieved_ids,
                    require_per_field=settings.require_per_field_citations
                )
                cov = citation_coverage(resp_model)

                if cite_issues or cov < settings.min_citation_coverage:
                    # ✅ NEW: Citation repair message (numbers must not change)
                    payload["messages"].append({
                        "role": "user",
                        "content": (
                            "CITATION REPAIR REQUIRED.\n"
                            f"- citation_coverage={cov:.2f}, min_required={settings.min_citation_coverage:.2f}\n"
                            f"- issues={[i.model_dump() for i in cite_issues]}\n\n"
                            "Fix ONLY the `justification` arrays inside `corep_mapping` so that:\n"
                            "1) Every corep_mapping item has at least one citation (if required).\n"
                            "2) Citations MUST ONLY use these allowed (source_id, chunk_id) pairs:\n"
                            f"{retrieved_ids}\n"
                            "3) Do NOT change any numeric values (inputs/computed/value).\n"
                            "Return JSON ONLY."
                        )
                    })
                    raise ValueError(
                        "Citations invalid or coverage below threshold.")

                # ✅ Success
                return resp_model

            except Exception as e:
                last_err = e
                # Repair attempt: nudge model to output valid JSON only
                payload["messages"].append({
                    "role": "user",
                    "content": (
                        "Your previous output failed validation.\n"
                        "Return ONLY valid JSON matching the schema.\n"
                        f"Error: {e}"
                    )
                })

        raise RuntimeError(f"LLM failed after retries. Last error: {last_err}")
