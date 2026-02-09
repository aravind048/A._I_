from __future__ import annotations
from pydantic import BaseModel
from corep_assistant.llm.schema import EntityInfo, InputsCA1


class ScenarioCA1(BaseModel):
    """
    ScenarioCA1 validates incoming scenario JSON BEFORE pipeline runs.

    Input example:
      {
        "reporting_date": "2026-01-31",
        "currency": "GBP",
        "entity": {"legal_entity_id":"BANK1","consolidation_level":"SOLO"},
        "inputs": {"cet1_before_deductions":120,"intangibles_deduction":8,"dta_future_profit_deduction":5,"at1":10,"t2":15}
      }
    """
    reporting_date: str
    currency: str
    entity: EntityInfo
    inputs: InputsCA1
