"""Adjudication trajectory log (U6: E9; axioms A3, A8).

Every LLM adjudication — fallbacks included — leaves one typed, append-only
JSONL row: the deterministic features the decision node saw, the tools it
called, what it spent (turns, tokens, dollars, wall ms), what it decided, and
whether the rule-based adjudicator agreed. The rows are the raw material for
distilling a cheaper student policy and for offline prompt optimization, and
they are the standing record that budget fallbacks actually happened — the
LLM node degrades loudly, never silently.

Discipline: traces are typed and imperative-free (S6) — enums, numbers, and
tool names, no free text a later reader could mistake for instructions
(``fallback_reason`` carries code-authored diagnostics only, never model
output). The writer is code-side, deliberately not a registered tool (S8):
the LLM cannot write to, or truncate, its own trajectory record.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .schemas import CaseDecision

DEFAULT_TRACE_PATH = Path("runs/traces/adjudications.jsonl")


def request_features(request: dict) -> dict[str, float]:
    """Deterministic feature vector of an adjudication request (E9).

    Numbers only, computed by code from the structured request — the exact
    vector a distilled student policy will consume, so traces and student can
    never drift apart on feature semantics.
    """
    candidates = request.get("candidates", [])
    consistency = request.get("consistency", {})
    reproduced = [float(c.get("reproduced_fraction", 0.0)) for c in candidates]
    return {
        "case_score": round(float(consistency.get("case_score", 0.0)), 4),
        "disagreement_rate": round(float(consistency.get("disagreement_rate", 0.0)), 4),
        "n_candidates": float(len(candidates)),
        "n_kept": float(sum(1 for c in candidates if c.get("kept"))),
        "n_vetoed": float(sum(1 for c in candidates if c.get("veto_reason"))),
        "max_candidate_score": round(
            max((float(c.get("score", 0.0)) for c in candidates), default=0.0), 4
        ),
        "mean_reproduced_fraction": round(
            sum(reproduced) / len(reproduced), 4
        ) if reproduced else 0.0,
        "n_blindspot": float(len(request.get("blindspot_candidates", []))),
        "qc_adequate": 1.0 if request.get("qc") == "adequate" else 0.0,
    }


class AdjudicationTrace(BaseModel):
    """One LLM-node adjudication, fully accounted for."""

    case_id: str
    policy_id: str = "unregistered"
    lane: str = "default"
    features: dict[str, float] = Field(
        default_factory=dict,
        description="deterministic request features; the vector a distilled student consumes",
    )
    tool_sequence: list[str] = Field(default_factory=list)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    usd: float = 0.0
    wall_ms: float = 0.0
    decision: CaseDecision
    rule_decision: CaseDecision
    agreed_with_rule: bool
    fallback_used: bool = False
    fallback_reason: str = Field(
        default="", description="code-authored diagnostic; empty when the LLM verdict stood"
    )


class TraceWriter:
    """Append-only JSONL sink under ``runs/traces/`` (code-side, never a tool)."""

    def __init__(self, path: str | Path = DEFAULT_TRACE_PATH) -> None:
        self.path = Path(path)

    def write(self, trace: AdjudicationTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(trace.model_dump(mode="json"), sort_keys=True) + "\n")

    def read_all(self) -> list[AdjudicationTrace]:
        """Schema-validated read-back (distillation / prompt-optimization input)."""
        if not self.path.exists():
            return []
        return [
            AdjudicationTrace.model_validate_json(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]
