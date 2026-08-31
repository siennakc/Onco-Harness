"""Structured output schemas with first-class abstention (2G, axiom A13).

Every model-facing decision is extracted into one of these shapes; free-form
text lives in a scratchpad turn, never in the record. ``abstain`` /
``not_assessable`` are enum values everywhere, and every quantitative field
must cite the ledger entry of the tool call that produced it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Assessment(str, Enum):
    present = "present"
    absent = "absent"
    uncertain = "uncertain"
    abstain = "abstain"
    not_assessable = "not_assessable"


class QCVerdict(str, Enum):
    adequate = "adequate"
    degraded = "degraded"
    inadequate_defer = "inadequate_defer"


class Finding(BaseModel):
    """One candidate finding, always localized (no finding without coordinates)."""

    finding_id: str
    box: tuple[int, int, int, int] = Field(description="x0,y0,x1,y1 in source pixels")
    size_mm: float | None = Field(default=None, description="computed by measure tool, never by the LLM")
    detector_score: float
    assessment: Assessment
    evidence_refs: list[str] = Field(
        default_factory=list, description="ledger entry ids supporting this finding"
    )
    notes: str = ""


class CaseDecision(str, Enum):
    recall = "recall"                  # suspicious — escalate / recall
    no_recall = "no_recall"
    defer_to_human = "defer_to_human"


class CaseReport(BaseModel):
    case_id: str
    qc: QCVerdict
    findings: list[Finding] = Field(default_factory=list)
    decision: CaseDecision
    score: float = Field(ge=0.0, le=1.0, description="calibrated case-level suspicion")
    disagreement_rate: float = Field(
        ge=0.0, le=1.0, description="fraction of self-consistency reads disagreeing"
    )
    deferral_reason: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class AdjudicationRequest(BaseModel):
    """What the LLM decision node receives: facts and handles, never pixels."""

    case_id: str
    qc: QCVerdict
    candidates: list[dict]
    consistency: dict
    atlas_neighbors: list[dict]
    guideline_notes: list[str]


class Adjudication(BaseModel):
    """What the LLM decision node must return."""

    per_candidate: dict[str, Assessment]
    decision: CaseDecision
    rationale: str
    cited_evidence: list[str]
