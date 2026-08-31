"""Difficulty-gated lane router + early exit (U2: E3/E4/E5; axioms A5, A8, A13).

At screening prevalence most cases are easy negatives, yet the pipeline spends
its full verification stack — TTA reads, zoom re-detections, segmentation,
blindspot re-search — re-verifying the obvious, and (when an LLM adjudicator
is attached) an Opus opinion on every case. Triage trials show the rule-out
arm is the single largest efficiency lever available; LLM-side routing shows
the same shape (BEST-Route: −60% cost at <1% quality drop). This module is
that lever, built to the tasksheet's discipline:

- **Lanes are a pure, deterministic function of cheap features.** The lane is
  decided by code from the primary detector's proposals and one blindspot
  probe — never by the model, never by anything stochastic. Same inputs, same
  lane, forever (the determinism gate still holds).
- **The safety net stays.** A fast-negative still passes through the rule
  adjudicator and the conformal policy layer (A13): a non-singleton
  prediction set defers no matter which lane the case took, and an
  unexplained blindspot hit disqualifies the fast lane outright (A5).
- **The LLM becomes a deferral-rescuer, not a default reader.** Escalation
  fires only on a rule-based deferral, conformal ambiguity, or an
  unexplained blindspot hit; ``llm_max_fraction`` is the circuit breaker
  that stops a degenerate policy from routing everything to Opus.
- **Routing is a policy change.** Every knob here lives in
  ``PolicyConfig.router`` (U4), so moving a lane threshold mints a new
  policy id and must pass the full promotion gate — sens non-inferiority,
  deferral non-regression, efficiency floors — before it runs a case.

Cascade order (U7 hook): rules -> distilled student -> LLM. When a promoted
:class:`~oncoharness.distill.StudentPolicy` is attached, an escalated case
goes to the student first; only below the student's promoted confidence
threshold does the case reach the LLM. The student is identified by content
hash (``student_id``) — a router refuses a student whose hash does not match
the registered policy, so a student can never be silently swapped (S8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum
from typing import Iterable, Protocol

from .schemas import Adjudication, CaseDecision
from .traces import request_features


class Lane(str, Enum):
    fast_negative = "fast_negative"  # detect + blindspot probe only
    routine = "routine"              # full deterministic stack, rule-based adjudication
    hard = "hard"                    # full stack + LLM adjudication


class Student(Protocol):
    """What the router needs from a distilled student (U7); duck-typed so
    ``router`` never imports ``distill`` (the student is data, not a dependency)."""

    def student_id(self) -> str: ...
    def predict(self, features: dict) -> tuple[CaseDecision, float]: ...


@dataclass(frozen=True)
class RouterConfig:
    """The whole reachable routing family; every field is a gated policy knob (U4/S9)."""

    fast_negative_max_score: float = 0.20   # top primary candidate below this
    fast_negative_requires_blindspot_clean: bool = True
    llm_on_rule_deferral: bool = True       # LLM invoked only to rescue deferrals
    llm_on_conformal_ambiguous: bool = True
    llm_max_fraction: float = 0.25          # circuit breaker: cap LLM lane share
    student_min_confidence: float | None = None  # tau (U7); None = no student band
    student_id: str = ""                    # content hash of the promoted student (U7)

    def __post_init__(self) -> None:
        if not 0.0 <= self.fast_negative_max_score <= 1.0:
            raise ValueError("fast_negative_max_score must be in [0, 1]")
        if not 0.0 <= self.llm_max_fraction <= 1.0:
            raise ValueError("llm_max_fraction must be in [0, 1]")
        if self.student_min_confidence is not None and not (
            0.0 <= self.student_min_confidence <= 1.0
        ):
            raise ValueError("student_min_confidence must be in [0, 1] or None")


ROUTER_KNOBS = frozenset(f.name for f in fields(RouterConfig))


def route_after_detect(
    candidates: list[dict], blindspot: list[dict], cfg: RouterConfig
) -> Lane:
    """Pure lane decision from cheap deterministic features (E3).

    Fast-negative iff no primary candidate reaches
    ``fast_negative_max_score`` and (when required) the blindspot probe is
    clean. Everything else takes the routine lane; ``hard`` is never
    assigned here — a case earns it only when the LLM is actually invoked.
    """
    top = max((float(c.get("score", 0.0)) for c in candidates), default=0.0)
    if top >= cfg.fast_negative_max_score:
        return Lane.routine
    if cfg.fast_negative_requires_blindspot_clean and blindspot:
        return Lane.routine
    return Lane.fast_negative


def escalate_to_llm(
    rule_adjudication: Adjudication,
    conformal_ambiguous: bool,
    blindspot: list[dict],
    cfg: RouterConfig,
) -> bool:
    """Pure escalation decision (E4): the LLM rescues deferrals, nothing else.

    True on a rule-based deferral, conformal ambiguity, or an unexplained
    blindspot hit — the three signals that a cheap verdict is not safe to
    ship. A confident rule verdict on a clean case never spends a token.
    """
    if cfg.llm_on_rule_deferral and rule_adjudication.decision == CaseDecision.defer_to_human:
        return True
    if cfg.llm_on_conformal_ambiguous and conformal_ambiguous:
        return True
    return bool(blindspot)


@dataclass
class RouteRecord:
    """One case's routing outcome, as typed telemetry (S6: flags and enums, no prose)."""

    case_id: str
    lane: str
    escalation_wanted: bool = False
    student_absorbed: bool = False
    llm_invoked: bool = False
    llm_denied_by_cap: bool = False

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "lane": self.lane,
            "escalation_wanted": self.escalation_wanted,
            "student_absorbed": self.student_absorbed,
            "llm_invoked": self.llm_invoked,
            "llm_denied_by_cap": self.llm_denied_by_cap,
        }


class DifficultyRouter:
    """Stateful wrapper over the pure decisions: counters + the LLM circuit breaker.

    The lane and escalation calls delegate to the module-level pure
    functions; the only state here is bookkeeping the circuit breaker needs
    (cases seen, LLM invocations granted) plus per-lane counts for
    telemetry. A router is batch-scoped, like a CostMeter.

    ``student`` is the U7 cascade tier. When the config names a
    ``student_id``, the attached student's content hash must match — a
    mismatched (or missing) student is refused at construction, so the
    routing policy that passed the gate is the one that runs (S8: no
    silent replacement).
    """

    def __init__(self, cfg: RouterConfig | None = None, student: Student | None = None) -> None:
        self.cfg = cfg or RouterConfig()
        if self.cfg.student_id:
            if student is None:
                raise ValueError(
                    f"router config names student {self.cfg.student_id!r} but no "
                    "student was attached (the promoted policy must run whole)"
                )
            if student.student_id() != self.cfg.student_id:
                raise ValueError(
                    f"attached student {student.student_id()!r} does not match the "
                    f"registered student_id {self.cfg.student_id!r}; refusing the "
                    "silent replacement (S8)"
                )
        elif student is not None:
            raise ValueError(
                "a student may only ride a policy that registers its student_id "
                "(adopting a student is a gated policy change, U7)"
            )
        self.student = student
        self.cases_seen = 0
        self.llm_invocations = 0
        self.lane_counts: dict[str, int] = {lane.value: 0 for lane in Lane}

    # -- decisions --------------------------------------------------------
    def route_after_detect(self, candidates: list[dict], blindspot: list[dict]) -> Lane:
        lane = route_after_detect(candidates, blindspot, self.cfg)
        self.cases_seen += 1
        self.lane_counts[lane.value] += 1
        return lane

    def escalate_to_llm(
        self,
        rule_adjudication: Adjudication,
        conformal_ambiguous: bool,
        blindspot: list[dict],
    ) -> bool:
        return escalate_to_llm(rule_adjudication, conformal_ambiguous, blindspot, self.cfg)

    def student_verdict(self, request: dict) -> tuple[CaseDecision, float] | None:
        """The distilled student's one shot at an escalated case (U7).

        Returns (decision, confidence) only when a student is attached AND
        its confidence clears the promoted threshold tau; anything less
        falls through to the LLM. The feature vector is the same
        :func:`~oncoharness.traces.request_features` the traces recorded,
        so student and training data can never drift apart.
        """
        if self.student is None or self.cfg.student_min_confidence is None:
            return None
        decision, confidence = self.student.predict(request_features(request))
        if confidence >= self.cfg.student_min_confidence:
            return decision, confidence
        return None

    def admit_llm(self) -> bool:
        """Circuit breaker (E4): grant the LLM only while its lane share stays capped.

        The grant is ceiling-based — ``invocations < ceil(fraction * seen)``
        — so the first escalation of a batch is admissible at any nonzero
        fraction, and the share over any prefix never exceeds the cap
        rounded up to a whole case. Denied cases keep their rule verdict.
        """
        if self.cfg.llm_max_fraction <= 0.0:
            return False
        allowed = math.ceil(self.cfg.llm_max_fraction * max(self.cases_seen, 1))
        if self.llm_invocations < allowed:
            self.llm_invocations += 1
            self.lane_counts[Lane.routine.value] -= 1
            self.lane_counts[Lane.hard.value] += 1
            return True
        return False


def lane_cost_summary(reports: Iterable) -> dict[str, dict]:
    """Per-lane cost roll-up from stamped CaseReports (E1/E2: routed vs not, in numbers).

    Groups reports by their ``lane`` stamp (un-routed reports land under
    ``"unrouted"``) and averages each lane's cost block, so an ablation or a
    nightly report can state what each lane costs — not just that lanes
    exist. Reports without a cost block contribute to ``n`` only.
    """
    grouped: dict[str, list] = {}
    for report in reports:
        lane = getattr(report, "lane", "") or "unrouted"
        grouped.setdefault(lane, []).append(report)
    summary: dict[str, dict] = {}
    for lane, group in sorted(grouped.items()):
        costs = [r.cost for r in group if getattr(r, "cost", None)]
        n = len(costs)
        summary[lane] = {
            "n": len(group),
            "mean_tool_calls": round(
                sum(c.get("total_tool_calls", 0) for c in costs) / n, 2
            ) if n else 0.0,
            "mean_wall_ms": round(sum(c.get("wall_ms", 0.0) for c in costs) / n, 2)
            if n
            else 0.0,
            "mean_llm_usd": round(sum(c.get("llm_usd", 0.0) for c in costs) / n, 6)
            if n
            else 0.0,
            "llm_invocations": sum(1 for c in costs if c.get("llm_turns", 0) > 0),
        }
    return summary
