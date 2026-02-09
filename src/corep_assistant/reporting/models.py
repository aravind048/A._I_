from __future__ import annotations
from pydantic import BaseModel
from typing import List


class TemplateField(BaseModel):
    field_id: str
    label: str


class TemplateSpec(BaseModel):
    template_id: str
    fields: List[TemplateField]
