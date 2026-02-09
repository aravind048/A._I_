from __future__ import annotations
from corep_assistant.reporting.models import TemplateSpec, TemplateField


CA1_SPEC = TemplateSpec(
    template_id="COREP_OWN_FUNDS_CA1",
    fields=[
        TemplateField(field_id="CA1.r010.c010", label="CET1 before deductions"),
        TemplateField(field_id="CA1.r020.c010", label="(-) Intangible assets"),
        TemplateField(field_id="CA1.r030.c010", label="(-) DTA reliant on future profitability"),
        TemplateField(field_id="CA1.r040.c010", label="CET1 after deductions"),
        TemplateField(field_id="CA1.r050.c010", label="AT1"),
        TemplateField(field_id="CA1.r060.c010", label="Tier 1"),
        TemplateField(field_id="CA1.r070.c010", label="Tier 2"),
        TemplateField(field_id="CA1.r080.c010", label="Total own funds"),
    ]
)
