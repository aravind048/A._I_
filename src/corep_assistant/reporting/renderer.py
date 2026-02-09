from __future__ import annotations
from typing import List, Dict
from corep_assistant.llm.schema import CorepResponse


def render_template_extract(response: CorepResponse) -> List[Dict[str, str]]:
    """
    Raw format:
      render_template_extract(response: CorepResponse) -> List[Dict[str, str]]

    Output rows suitable for printing or exporting later.
    """
    rows = []
    for m in response.corep_mapping:
        rows.append({
            "field_id": m.field_id,
            "label": m.label,
            "value": f"{m.value}",
            "unit": m.unit
        })
    return rows
