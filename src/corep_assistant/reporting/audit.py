from __future__ import annotations
from typing import List, Dict, Any
from corep_assistant.llm.schema import CorepResponse


def build_audit_log(resp: CorepResponse) -> List[Dict[str, Any]]:
    """
    Raw format:
      build_audit_log(resp: CorepResponse) -> List[Dict[str, Any]]

    Output:
      One item per populated field with citations.
    """
    log = []
    for m in resp.corep_mapping:
        log.append({
            "field_id": m.field_id,
            "label": m.label,
            "value": m.value,
            "citations": [c.model_dump() for c in m.justification]
        })
    return log
