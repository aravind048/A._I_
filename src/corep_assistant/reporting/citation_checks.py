from __future__ import annotations
from typing import List, Set, Tuple, Dict, Any
from corep_assistant.llm.schema import CorepResponse, ValidationIssue


def _retrieved_id_set(retrieved: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    """
    _retrieved_id_set(retrieved) -> set of (source_id, chunk_id)
    """
    out: Set[Tuple[str, str]] = set()
    for r in retrieved:
        out.add((r["source_id"], r["chunk_id"]))
    return out


def check_citations(
    resp: CorepResponse,
    retrieved: List[Dict[str, Any]],
    require_per_field: bool = True
) -> List[ValidationIssue]:
    """
    check_citations(resp, retrieved, require_per_field=True) -> List[ValidationIssue]

    Validates:
      - Each mapping has >=1 citation (if require_per_field)
      - Every citation appears in retrieved set
    """
    issues: List[ValidationIssue] = []
    allowed = _retrieved_id_set(retrieved)

    for m in resp.corep_mapping:
        if require_per_field and (not m.justification or len(m.justification) == 0):
            issues.append(ValidationIssue(
                rule_id="CITE_001",
                severity="ERROR",
                message=f"Missing citations for field {m.field_id}. Each field must cite at least one retrieved chunk.",
                affected_fields=[m.field_id]
            ))
            continue

        for c in m.justification:
            key = (c.source_id, c.chunk_id)
            if key not in allowed:
                issues.append(ValidationIssue(
                    rule_id="CITE_002",
                    severity="ERROR",
                    message=f"Invalid citation for field {m.field_id}: ({c.source_id}, {c.chunk_id}) not in retrieved evidence.",
                    affected_fields=[m.field_id]
                ))

    return issues


def citation_coverage(resp: CorepResponse) -> float:
    """
    citation_coverage(resp) -> float in [0, 1]

    Coverage = (# fields with >=1 citation) / (total fields)
    """
    if not resp.corep_mapping:
        return 0.0
    cited = sum(1 for m in resp.corep_mapping if m.justification and len(m.justification) > 0)
    return cited / len(resp.corep_mapping)
