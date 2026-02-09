from __future__ import annotations
from typing import List
from corep_assistant.llm.schema import CorepResponse, ValidationIssue


def validate_ca1(resp: CorepResponse) -> List[ValidationIssue]:
    """
    Raw format:
      validate_ca1(resp: CorepResponse) -> List[ValidationIssue]

    Rules:
      VAL_001: CET1 after deductions consistency
      VAL_002: Tier1 consistency
      VAL_003: Total own funds consistency
      VAL_004: Negative deductions warn
    """
    issues: List[ValidationIssue] = []

    i = resp.inputs
    c = resp.computed

    expected_cet1_after = i.cet1_before_deductions - (i.intangibles_deduction + i.dta_future_profit_deduction)
    if abs(c.cet1_after_deductions - expected_cet1_after) > 1e-6:
        issues.append(ValidationIssue(
            rule_id="VAL_001",
            severity="ERROR",
            message=f"CET1 after deductions mismatch. Expected {expected_cet1_after}, got {c.cet1_after_deductions}.",
            affected_fields=["CA1.r040.c010"]
        ))

    expected_tier1 = c.cet1_after_deductions + i.at1
    if abs(c.tier1 - expected_tier1) > 1e-6:
        issues.append(ValidationIssue(
            rule_id="VAL_002",
            severity="ERROR",
            message=f"Tier 1 mismatch. Expected {expected_tier1}, got {c.tier1}.",
            affected_fields=["CA1.r060.c010"]
        ))

    expected_total = c.tier1 + i.t2
    if abs(c.total_own_funds - expected_total) > 1e-6:
        issues.append(ValidationIssue(
            rule_id="VAL_003",
            severity="ERROR",
            message=f"Total own funds mismatch. Expected {expected_total}, got {c.total_own_funds}.",
            affected_fields=["CA1.r080.c010"]
        ))

    if i.intangibles_deduction < 0 or i.dta_future_profit_deduction < 0:
        issues.append(ValidationIssue(
            rule_id="VAL_004",
            severity="WARN",
            message="Deductions are negative; check sign conventions.",
            affected_fields=["CA1.r020.c010", "CA1.r030.c010"]
        ))

    return issues
