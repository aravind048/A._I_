from __future__ import annotations
from typing import List
from corep_assistant.llm.schema import (
    CorepResponse, CorepFieldMapping, Citation, ComputedCA1
)
from corep_assistant.retrieval.models import RetrievedChunk


class MockLLM:
    """
    Deterministic output for prototype pipeline validation.

    Input:
      - question
      - scenario (already validated structure)
      - retrieved_chunks (used for citations)

    Output:
      - CorepResponse (schema compliant)
    """

    def generate_ca1(
        self,
        question: str,
        base_response: CorepResponse,
        retrieved: List[RetrievedChunk]
    ) -> CorepResponse:
        """
        Raw format:
          generate_ca1(question: str, base_response: CorepResponse, retrieved: List[RetrievedChunk]) -> CorepResponse
        """
        i = base_response.inputs

        # compute
        cet1_after = i.cet1_before_deductions - (i.intangibles_deduction + i.dta_future_profit_deduction)
        tier1 = cet1_after + i.at1
        total = tier1 + i.t2

        base_response.computed = ComputedCA1(
            cet1_after_deductions=cet1_after,
            tier1=tier1,
            total_own_funds=total
        )

        # citations: just pick top 2 retrieved chunks for all fields (prototype)
        cites = [Citation(source_id=r.source_id, chunk_id=r.chunk_id) for r in retrieved[:2]]

        base_response.corep_mapping = [
            CorepFieldMapping(field_id="CA1.r010.c010", label="CET1 before deductions", value=i.cet1_before_deductions, justification=cites),
            CorepFieldMapping(field_id="CA1.r020.c010", label="(-) Intangible assets", value=i.intangibles_deduction, justification=cites),
            CorepFieldMapping(field_id="CA1.r030.c010", label="(-) DTA reliant on future profitability", value=i.dta_future_profit_deduction, justification=cites),
            CorepFieldMapping(field_id="CA1.r040.c010", label="CET1 after deductions", value=cet1_after, justification=cites),
            CorepFieldMapping(field_id="CA1.r050.c010", label="AT1", value=i.at1, justification=cites),
            CorepFieldMapping(field_id="CA1.r060.c010", label="Tier 1", value=tier1, justification=cites),
            CorepFieldMapping(field_id="CA1.r070.c010", label="Tier 2", value=i.t2, justification=cites),
            CorepFieldMapping(field_id="CA1.r080.c010", label="Total own funds", value=total, justification=cites),
        ]

        base_response.assumptions.append("Prototype mock logic used for computation; replace with real LLM mapping later.")
        base_response.debug["question"] = question
        base_response.debug["retrieved_count"] = len(retrieved)

        return base_response
