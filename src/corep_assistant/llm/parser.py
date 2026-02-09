from __future__ import annotations
import json
from typing import Any, Dict


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Raw format:
      extract_json_object(text: str) -> dict

    Logic:
      - find first '{' and last '}' and attempt json.loads
      - if fails, try progressively trimming
    """
    s = text.strip()

    # Fast path: direct JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Remove code fences if present
    s = s.replace("```json", "").replace("```", "").strip()

    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found in LLM output.")

    candidate = s[first:last+1].strip()

    # Try full candidate
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        # Progressive trim (cheap heuristic)
        for cut in range(len(candidate), max(len(candidate) - 5000, 0), -200):
            try:
                obj = json.loads(candidate[:cut])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    raise ValueError("Failed to parse JSON from LLM output.")
