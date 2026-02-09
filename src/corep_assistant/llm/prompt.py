from __future__ import annotations
from typing import Dict, Any, List
from corep_assistant.retrieval.models import RetrievedChunk


def build_ca1_prompt(
    question: str,
    scenario: Dict[str, Any],
    retrieved: List[RetrievedChunk]
) -> str:
    """
    Raw format:
      build_ca1_prompt(question: str, scenario: dict, retrieved: List[RetrievedChunk]) -> str

    Logic:
      - Give the model: question + scenario + retrieved evidence with stable IDs
      - Force JSON-only output that matches CorepResponse schema
      - Require citations to use ONLY provided (source_id, chunk_id)
    """
    evidence_lines = []
    for r in retrieved:
        text = r.text.replace("\n", " ").strip()
        evidence_lines.append(
            f"- [source_id={r.source_id} chunk_id={r.chunk_id} score={r.score:.4f}] {text}"
        )

    evidence_block = "\n".join(
        evidence_lines) if evidence_lines else "(no evidence retrieved)"

    # Keep schema hint short to avoid token bloat (prototype)
    schema_hint = """
Return JSON ONLY (no markdown, no commentary) matching this structure exactly.

MANDATORY RULES (do not violate):
1) corep_mapping MUST contain exactly these 8 field_ids:
   - CA1.r010.c010, CA1.r020.c010, CA1.r030.c010, CA1.r040.c010,
     CA1.r050.c010, CA1.r060.c010, CA1.r070.c010, CA1.r080.c010

2) For EACH corep_mapping item:
   - justification must include at least 1 citation.
   - Each citation must be one of the Evidence IDs provided below (source_id + chunk_id).
   - Do NOT invent citations.

3) Computations must be consistent:
   cet1_after_deductions = cet1_before_deductions - (intangibles_deduction + dta_future_profit_deduction)
   tier1 = cet1_after_deductions + at1
   total_own_funds = tier1 + t2

4) If evidence is insufficient for a field, still populate the numeric value from scenario,
   but add an assumption explaining the limitation AND cite the closest evidence chunk.

OUTPUT MUST BE VALID JSON.
""".strip()

    prompt = f"""
You are a regulatory reporting assistant for PRA COREP.

User question:
{question}

Scenario (facts):
{scenario}

Evidence (retrieved regulatory text and instructions):
{evidence_block}

{schema_hint}
""".strip()

    return prompt
