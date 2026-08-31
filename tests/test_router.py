"""U2 difficulty-gated lane router: lanes, early exit, escalation, circuit breaker.

The router's claims are gated claims: the fast lane must actually cost less
(CostMeter numbers, not vibes), the LLM must actually stay out of easy cases
(spy adjudicator), and the routed policy must survive the same promotion
gate as any other change (paired run vs the always-full arm).
"""

import numpy as np
import pytest

from oncoharness.gate import load_rules, run_gate
from oncoharness.ledger import EvidenceLedger
from oncoharness.phantom import generate_dataset
from oncoharness.router import (
    DifficultyRouter,
    Lane,
    RouterConfig,
    escalate_to_llm,
    lane_cost_summary,
    route_after_detect,
)
from oncoharness.schemas import Adjudication, CaseDecision
from oncoharness.state_machine import HarnessPipeline, RuleBasedAdjudicator
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt


class SpyLLM:
    """Stands in for the LLM adjudicator: counts invocations, answers confidently."""

    def __init__(self):
        self.calls = 0

    def adjudicate(self, request: dict) -> Adjudication:
        self.calls += 1
        score = float(request["consistency"]["case_score"])
        return Adjudication(
            per_candidate={},
            decision=CaseDecision.recall if score >= 0.5 else CaseDecision.no_recall,
            rationale="llm rescue",
            cited_evidence=[],
        )


def _pipeline(tmp_path, router=None, escalation=None, adjudicator=None, **kwargs):
    tb = Toolbelt(
        ArtifactStore(tmp_path / "artifacts"), EvidenceLedger(tmp_path / "ledger.jsonl")
    )
    return HarnessPipeline(
        tb,
        adjudicator=adjudicator,
        consistency_reads=3,
        min_reproduced=2,
        router=router,
        escalation_adjudicator=escalation,
        **kwargs,
    )


def _ramp_negative(size: int = 96) -> np.ndarray:
    """A featureless gradient: passes QC, gives the detector nothing to propose."""
    yy = np.linspace(0.3, 0.5, size, dtype=np.float32)
    return np.tile(yy[:, None], (1, size))


# -- pure decisions -------------------------------------------------------

def test_route_after_detect_is_pure_and_deterministic():
    cfg = RouterConfig()
    weak = [{"score": 0.1}, {"score": 0.19}]
    strong = [{"score": 0.1}, {"score": 0.21}]
    assert route_after_detect(weak, [], cfg) == Lane.fast_negative
    assert route_after_detect(weak, [], cfg) == Lane.fast_negative  # same in, same out
    assert route_after_detect(strong, [], cfg) == Lane.routine
    assert route_after_detect([], [], cfg) == Lane.fast_negative
    # an unexplained blindspot hit disqualifies the fast lane (A5)
    assert route_after_detect(weak, [{"score": 0.6}], cfg) == Lane.routine
    relaxed = RouterConfig(fast_negative_requires_blindspot_clean=False)
    assert route_after_detect(weak, [{"score": 0.6}], relaxed) == Lane.fast_negative


def test_escalate_to_llm_covers_the_three_deferral_signals():
    cfg = RouterConfig()
    defer = Adjudication(
        per_candidate={}, decision=CaseDecision.defer_to_human, rationale="", cited_evidence=[]
    )
    confident = Adjudication(
        per_candidate={}, decision=CaseDecision.no_recall, rationale="", cited_evidence=[]
    )
    assert escalate_to_llm(defer, False, [], cfg)
    assert escalate_to_llm(confident, True, [], cfg)
    assert escalate_to_llm(confident, False, [{"score": 0.6}], cfg)
    assert not escalate_to_llm(confident, False, [], cfg)
    # both rescue channels can be switched off by policy
    off = RouterConfig(llm_on_rule_deferral=False, llm_on_conformal_ambiguous=False)
    assert not escalate_to_llm(defer, True, [], off)


# -- lanes in the pipeline ------------------------------------------------

def test_fast_negative_lane_costs_at_most_three_tool_calls(tmp_path):
    spy = SpyLLM()
    pipeline = _pipeline(tmp_path, router=DifficultyRouter(), escalation=spy)
    report = pipeline.run_case("neg", _ramp_negative())
    assert report.lane == Lane.fast_negative.value
    assert report.decision == CaseDecision.no_recall
    assert report.cost["total_tool_calls"] <= 3
    assert spy.calls == 0
    routes = [e for e in pipeline.tools.ledger.entries() if e["kind"] == "route"]
    assert routes[-1]["payload"]["lane"] == "fast_negative"
    assert pipeline.tools.ledger.verify_chain()


def test_obvious_lesion_takes_routine_lane_and_never_wakes_the_llm(tmp_path):
    case = next(
        c for c in generate_dataset(n_patients=6, images_per_patient=1, prevalence=1.0, seed=5)
        if c.label == 1
    )
    spy = SpyLLM()
    pipeline = _pipeline(tmp_path, router=DifficultyRouter(), escalation=spy)
    report = pipeline.run_case(case.case_id, case.pixels)
    assert report.lane == Lane.routine.value
    assert report.decision == CaseDecision.recall
    assert spy.calls == 0


def test_mid_band_case_escalates_to_hard_lane_exactly_once(tmp_path):
    case = next(
        c for c in generate_dataset(n_patients=6, images_per_patient=1, prevalence=1.0, seed=5)
        if c.label == 1
    )
    spy = SpyLLM()
    # A deferral band that swallows everything below 0.95 forces the rule
    # adjudicator to defer this lesion — the exact case the LLM exists for.
    pipeline = _pipeline(
        tmp_path,
        router=DifficultyRouter(),
        escalation=spy,
        adjudicator=RuleBasedAdjudicator(recall_threshold=0.95, deferral_band=(0.0, 0.95)),
    )
    report = pipeline.run_case(case.case_id, case.pixels)
    assert report.lane == Lane.hard.value
    assert spy.calls == 1
    assert report.decision == CaseDecision.recall  # deferral rescued, not parroted


def test_llm_fraction_breach_keeps_rule_verdict(tmp_path):
    cases = [
        c for c in generate_dataset(n_patients=8, images_per_patient=1, prevalence=1.0, seed=5)
        if c.label == 1
    ][:4]
    spy = SpyLLM()
    pipeline = _pipeline(
        tmp_path,
        router=DifficultyRouter(RouterConfig(llm_max_fraction=0.25)),
        escalation=spy,
        adjudicator=RuleBasedAdjudicator(recall_threshold=0.95, deferral_band=(0.0, 0.95)),
    )
    reports = [pipeline.run_case(c.case_id, c.pixels) for c in cases]
    assert spy.calls == 1  # ceil(0.25 * 4): one grant, three denials
    assert [r.lane for r in reports] == ["hard", "routine", "routine", "routine"]
    # excess cases keep the rule verdict (deferral), never a truncated answer
    assert all(r.decision == CaseDecision.defer_to_human for r in reports[1:])
    denied = [
        e["payload"]["llm_denied_by_cap"]
        for e in pipeline.tools.ledger.entries()
        if e["kind"] == "route"
    ]
    assert denied == [False, True, True, True]


def test_early_exit_skips_tta_when_detect_returns_nothing(tmp_path):
    img = _ramp_negative()
    unrouted = _pipeline(tmp_path / "a")
    routed = _pipeline(
        tmp_path / "b",
        router=DifficultyRouter(RouterConfig(fast_negative_max_score=0.0)),  # fast lane off
    )
    r_full = unrouted.run_case("neg", img)
    r_routed = routed.run_case("neg", img)
    assert r_routed.lane == Lane.routine.value
    assert r_routed.decision == r_full.decision
    assert r_routed.score == r_full.score
    # TTA is vacuous with zero proposals: 3 reads skipped, blindspot still runs
    full_detects = r_full.cost["tool_calls"]["run_detector"]
    routed_detects = r_routed.cost["tool_calls"]["run_detector"]
    assert routed_detects == full_detects - 3 == 2


def test_pipeline_without_router_is_unchanged_and_unlaned(tmp_path):
    case = generate_dataset(n_patients=2, images_per_patient=1, prevalence=0.5, seed=3)[0]
    report = _pipeline(tmp_path).run_case(case.case_id, case.pixels)
    assert report.lane == ""
    assert not [e for e in _pipeline(tmp_path / "x").tools.ledger.entries() if e["kind"] == "route"]


def test_escalation_adjudicator_without_router_is_refused(tmp_path):
    with pytest.raises(ValueError, match="requires a router"):
        _pipeline(tmp_path, router=None, escalation=SpyLLM())


# -- the routed arm vs the always-full arm --------------------------------

def test_routed_arm_passes_paired_gate_and_cuts_tool_calls(tmp_path):
    cases = generate_dataset(n_patients=24, images_per_patient=1, prevalence=0.25, seed=11)
    y = np.array([c.label for c in cases])
    pids = [c.patient_id for c in cases]

    full = _pipeline(tmp_path / "full")
    # The lane threshold is a policy knob calibrated to the detector's score
    # scale; the reference DoG detector runs hot on textured phantom
    # backgrounds (negatives top out near 0.5), so the phantom-calibrated
    # fast lane sits at 0.5 where real calibrated scores would use ~0.2.
    routed = _pipeline(
        tmp_path / "routed",
        router=DifficultyRouter(RouterConfig(fast_negative_max_score=0.5)),
    )
    full_reports = [full.run_case(c.case_id, c.pixels) for c in cases]
    routed_reports = [routed.run_case(c.case_id, c.pixels) for c in cases]

    rules = load_rules("gates/gate_rules.yaml")
    result = run_gate(
        rules,
        y,
        np.array([r.score for r in routed_reports]),
        np.array([r.score for r in full_reports]),
        pids,
    )
    by_name = {c.name: c for c in result.checks}
    assert by_name["primary_metric_non_inferiority"].passed
    assert by_name["negative_flip_rate"].passed

    # deferral non-regression, against the human-owned gate ceiling
    defer = lambda reports: np.mean(  # noqa: E731
        [r.decision == CaseDecision.defer_to_human for r in reports]
    )
    assert defer(routed_reports) <= defer(full_reports) + rules["deferral"]["max_rate_increase"]

    # the efficiency claim, in the meter's numbers (E1): >= 40% fewer calls/case
    mean_calls = lambda reports: np.mean(  # noqa: E731
        [r.cost["total_tool_calls"] for r in reports]
    )
    assert mean_calls(routed_reports) <= 0.6 * mean_calls(full_reports)

    # per-lane costs land in telemetry: the fast lane must be the cheap one
    summary = lane_cost_summary(routed_reports)
    assert summary["fast_negative"]["n"] >= 1
    assert summary["fast_negative"]["mean_tool_calls"] < summary["routine"]["mean_tool_calls"]


def test_ablation_gains_a_routed_arm_with_cost_columns():
    from oncoharness.ablation import run_ablation

    cases = generate_dataset(n_patients=10, images_per_patient=1, prevalence=0.3, seed=11)
    by_arm = {r.arm: r for r in run_ablation(cases)}
    assert "harness_routed" in by_arm
    assert by_arm["harness_routed"].mean_tool_calls < by_arm["harness"].mean_tool_calls
    assert by_arm["harness_routed"].auroc >= by_arm["harness"].auroc - 0.1


# -- routing as a gated policy change -------------------------------------

def test_policy_carries_router_knobs_and_refuses_unknown_ones(tmp_path):
    from oncoharness.policy import PolicyConfig, pipeline_from_policy

    tb = Toolbelt(
        ArtifactStore(tmp_path / "artifacts"), EvidenceLedger(tmp_path / "ledger.jsonl")
    )
    cfg = PolicyConfig(router={"fast_negative_max_score": 0.15, "llm_max_fraction": 0.2})
    pipeline = pipeline_from_policy(cfg, tb)
    assert pipeline.router is not None
    assert pipeline.router.cfg.fast_negative_max_score == 0.15

    rogue = PolicyConfig(router={"route_everything_to_opus": True})
    with pytest.raises(ValueError, match="unknown router knob"):
        pipeline_from_policy(rogue, tb)
    # two routers that differ in one threshold are two different policies (U4)
    assert cfg.policy_id() != PolicyConfig(router={"fast_negative_max_score": 0.2}).policy_id()
