from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime


class Citation(BaseModel):
    """
    citation(source_id, chunk_id) -> identifies retrieved evidence
    """
    source_id: str
    chunk_id: str


class CorepFieldMapping(BaseModel):
    """
    One field populated for the COREP template.
    """
    field_id: str
    label: str
    value: float
    unit: str = "GBP"
    justification: List[Citation] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    rule_id: str
    severity: Literal["ERROR", "WARN"]
    message: str
    affected_fields: List[str] = Field(default_factory=list)


class EntityInfo(BaseModel):
    legal_entity_id: str
    consolidation_level: Literal["SOLO", "CONSOLIDATED"]


class InputsCA1(BaseModel):
    cet1_before_deductions: float = 0.0
    intangibles_deduction: float = 0.0
    dta_future_profit_deduction: float = 0.0
    at1: float = 0.0
    t2: float = 0.0


class ComputedCA1(BaseModel):
    cet1_after_deductions: float = 0.0
    tier1: float = 0.0
    total_own_funds: float = 0.0


class CorepResponse(BaseModel):
    """
    Final structured output for the assistant.

    This is what the LLM (mock/real) must produce.
    """
    template_id: str
    reporting_date: str
    currency: str
    entity: EntityInfo

    inputs: InputsCA1
    computed: ComputedCA1

    corep_mapping: List[CorepFieldMapping] = Field(default_factory=list)
    validation: List[ValidationIssue] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)

    # optional internal debug info
    debug: Dict[str, Any] = Field(default_factory=dict)
