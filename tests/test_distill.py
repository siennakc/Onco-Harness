"""U7 adjudication distillation: agreement band, cascade, gated promotion.

The claims under test: a deterministic teacher distills at >=99% agreement
with a usable tau; noisy regions fall out of the band and stay with the LLM;
the end-to-end cascade strictly reduces LLM invocations at equal decisions;
the student is promoted ONLY through run_gate + PolicyRegistry.promote (a
perfect imitator of a wrong teacher is refused on ground truth); and a
student can never be silently swapped past its registered content hash.
"""

import numpy as np
import pytest
import yaml

from fake_sdk import make_request

from oncoharness.distill import (
    CLASSES,
    FEATURES,
    StudentPolicy,
    agreement_report,
    confidence_threshold,
    fit_student,
    load_traces,
    load_student,
    mine_disagreements,
    save_student,
    train_holdout_split,
)
from oncoharness.failure_bank import FailureBank, FailureRecord, FailureSource, FailureStatus
from oncoharness.gate import run_gate
from oncoharness.ledger import EvidenceLedger
from oncoharness.phantom import generate_dataset
from oncoharness.policy import PolicyConfig, PolicyRegistry, pipeline_from_policy
from oncoharness.router import DifficultyRouter, RouterConfig
from oncoharness.schemas import Adjudication, CaseDecision
from oncoharness.sequential import CaseCost
from oncoharness.state_machine import HarnessPipeline, RuleBasedAdjudicator
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt
from oncoharness.traces import AdjudicationTrace, TraceWriter, request_features

RULES = yaml.safe_load(open("gates/gate_rules.yaml"))


def _features(case_score, disagreement=0.05):
    return request_features(make_request(case_score=round(case_score, 4), disagreement=disagreement))


def _grid_rows(teacher, n=201, seed=None, noisy_band=None):
    """(features, decision) rows over a case-score grid; optionally noisy inside a band."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in np.linspace(0.0, 1.0, n):
        feats = _features(s)
        if noisy_band and noisy_band[0] < s < noisy_band[1]:
            decision = CaseDecision(
                [CaseDecision.recall.value, CaseDecision.no_recall.value][int(rng.integers(2))]
            )
        else:
            decision = teacher(feats)
        rows.append((feats, decision))
    return rows


def _matrix(rows):
    X = np.array([[f.get(name, 0.0) for name in FEATURES] for f, _ in rows])
    y = np.array([CLASSES.index(d.value) for _, d in rows])
    return X, y


class TracingSpyLLM:
    """Mimics the LLM adjudicator's observable economics: counts invocations,
    meters spend into the case's CostMeter, writes AdjudicationTrace rows."""

    def __init__(self, meter=None, trace_writer=None, rule=None, usd_per_call=0.02):
        self.calls = 0
        self.meter = meter
        self.trace_writer = trace_writer
        self.rule = rule or RuleBasedAdjudicator()
        self.usd_per_call = usd_per_call

    def adjudicate(self, request: dict) -> Adjudication:
        self.calls += 1
        feats = request_features(request)
        decision = (
            CaseDecision.recall if feats["case_score"] >= 0.5 else CaseDecision.no_recall
        )
        if self.meter is not None:
            self.meter.record_llm(
                {"input_tokens": 1200, "output_tokens": 80, "cache_read_input_tokens": 900},
                usd=self.usd_per_call,
                turns=2,
            )
        if self.trace_writer is not None:
            rule_decision = self.rule.adjudicate(request).decision
            self.trace_writer.write(
                AdjudicationTrace(
                    case_id=str(request.get("case_id", "")),
                    lane=str(request.get("lane", "default")),
                    features=feats,
                    turns=2,
                    usd=self.usd_per_call,
                    decision=decision,
                    rule_decision=rule_decision,
                    agreed_with_rule=decision == rule_decision,
                )
            )
        return Adjudication(
            per_candidate={}, decision=decision, rationale="llm rescue", cited_evidence=[]
        )


def _routed_pipeline(tmp_path, spy, student=None, tau=None, sid=""):
    tb = Toolbelt(
        ArtifactStore(tmp_path / "artifacts"), EvidenceLedger(tmp_path / "ledger.jsonl")
    )
    spy.meter = tb.meter
    cfg = RouterConfig(
        fast_negative_max_score=0.5,
        llm_max_fraction=1.0,
        student_min_confidence=tau,
        student_id=sid,
    )
    return HarnessPipeline(
        tb,
        adjudicator=RuleBasedAdjudicator(recall_threshold=0.9, deferral_band=(0.05, 0.9)),
        consistency_reads=3,
        min_reproduced=2,
        router=DifficultyRouter(cfg, student=student),
        escalation_adjudicator=spy,
    )


def _cohort(n=1200, seed=0):
    """Calibrated-by-construction score set: the sealed-eval stand-in (test_gate recipe)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    scores = 1.0 / (1.0 + np.exp(-4.0 * x))
    y = (rng.random(n) < scores).astype(int)
    pids = [f"P{i//2}" for i in range(n)]
    return y, scores, pids


# -- features and fitting --------------------------------------------------

def test_features_match_the_trace_schema_exactly():
    """Drift guard: the student consumes precisely what traces record (E9)."""
    assert FEATURES == tuple(request_features(make_request()).keys())


def test_deterministic_teacher_distills_above_99_percent_with_tau(tmp_path):
    def banded(feats):
        s = feats["case_score"]
        if s > 0.6:
            return CaseDecision.recall
        return CaseDecision.defer_to_human if s > 0.4 else CaseDecision.no_recall

    writer = TraceWriter(tmp_path / "traces.jsonl")
    for i, (feats, decision) in enumerate(_grid_rows(banded)):
        writer.write(
            AdjudicationTrace(
                case_id=f"t{i}", features=feats, decision=decision,
                rule_decision=CaseDecision.defer_to_human, agreed_with_rule=False,
            )
        )
    # fallback rows are training poison and must be excluded by load_traces
    writer.write(
        AdjudicationTrace(
            case_id="fb", features=_features(0.99), decision=CaseDecision.no_recall,
            rule_decision=CaseDecision.no_recall, agreed_with_rule=True,
            fallback_used=True, fallback_reason="usd budget breached",
        )
    )
    X, y, names = load_traces(tmp_path / "traces.jsonl", min_rows=200)
    assert names == list(FEATURES) and len(X) == 201  # the fallback row is gone

    # the tree carves the banded 3-class teacher exactly (axis-aligned)
    train, hold = train_holdout_split(len(X))
    student = fit_student(X[train], y[train], kind="tree", max_depth=2, min_leaf=5)
    report = agreement_report(student, X[hold], y[hold])
    assert report["overall_agreement"] >= 0.99, report
    tau = confidence_threshold(report, min_agreement=0.98)
    assert tau is not None and 0.0 <= tau <= 1.0

    # the logistic student nails a linearly separable teacher
    Xl, yl = _matrix(
        _grid_rows(
            lambda f: CaseDecision.recall if f["case_score"] >= 0.5 else CaseDecision.no_recall
        )
    )
    lstudent = fit_student(Xl[train], yl[train], kind="logistic")
    lreport = agreement_report(lstudent, Xl[hold], yl[hold])
    assert lreport["overall_agreement"] >= 0.99, lreport
    assert confidence_threshold(lreport, min_agreement=0.98) is not None

    with pytest.raises(ValueError, match="need >= 500"):
        load_traces(tmp_path / "traces.jsonl", min_rows=500)


def test_noisy_band_falls_below_threshold_and_stays_with_the_llm():
    def clean(feats):
        return (
            CaseDecision.recall if feats["case_score"] >= 0.55 else CaseDecision.no_recall
        )

    rows = _grid_rows(clean, seed=0, noisy_band=(0.45, 0.55))
    X, y = _matrix(rows)
    train, hold = train_holdout_split(len(X))
    student = fit_student(X[train], y[train], kind="tree", max_depth=2, min_leaf=5)
    report = agreement_report(student, X[hold], y[hold])
    tau = confidence_threshold(report, min_agreement=0.98)
    assert tau is not None and tau > 0.55  # the noisy leaf's confidence is out of band
    assert any(
        r["n"] and r["agreement"] < 0.98 for r in report["bins"] if r["lo"] < tau
    )  # and it is out of band FOR CAUSE, measured on held-out traces

    sid = student.student_id()
    router = DifficultyRouter(
        RouterConfig(student_min_confidence=tau, student_id=sid), student=student
    )
    absorbed, to_llm = 0, 0
    for s in (0.9, 0.8, 0.5, 0.52, 0.1):
        verdict = router.student_verdict(make_request(case_score=s))
        if verdict is None:
            to_llm += 1  # below tau: the case the LLM still exists for
        else:
            absorbed += 1
    assert (absorbed, to_llm) == (3, 2)


def test_student_json_roundtrip_and_tamper_detection(tmp_path):
    rows = _grid_rows(
        lambda f: CaseDecision.recall if f["case_score"] >= 0.5 else CaseDecision.no_recall
    )
    X, y = _matrix(rows)
    for kind in ("logistic", "tree"):
        student = fit_student(X, y, kind=kind, max_depth=2)
        clone = StudentPolicy.from_json(student.to_json())
        assert clone.student_id() == student.student_id()
        for s in (0.05, 0.45, 0.55, 0.95):
            assert clone.predict(_features(s)) == student.predict(_features(s))

    sid = save_student(student, tmp_path / "students")
    assert save_student(student, tmp_path / "students") == sid  # idempotent
    loaded = load_student(sid, tmp_path / "students")
    assert loaded.student_id() == sid
    path = tmp_path / "students" / f"{sid}.json"
    path.write_text(path.read_text().replace("0.5", "0.9", 1))
    with pytest.raises(RuntimeError, match="edited on disk"):
        load_student(sid, tmp_path / "students")


def test_high_confidence_disagreements_become_probation_records(tmp_path):
    rows = _grid_rows(
        lambda f: CaseDecision.recall if f["case_score"] >= 0.5 else CaseDecision.no_recall
    )
    X, y = _matrix(rows)
    student = fit_student(X, y, kind="tree", max_depth=2)
    # three traces where the "LLM" contradicted what it usually does
    traces = [
        AdjudicationTrace(
            case_id=f"dev{i}", features=_features(s), decision=decision,
            rule_decision=CaseDecision.defer_to_human, agreed_with_rule=False,
        )
        for i, (s, decision) in enumerate(
            [(0.9, CaseDecision.no_recall), (0.1, CaseDecision.recall), (0.95, CaseDecision.recall)]
        )
    ]
    bank = FailureBank(root=tmp_path / "bank")
    added = mine_disagreements(
        student, traces, tau=0.9, bank=bank, labels={"dev0": 1, "dev1": 0}
    )
    # dev0 and dev1 disagree at high confidence and carry labels; dev2 agrees
    assert len(added) == 2
    probation = bank.records("probation")
    assert {r.case_ref for r in probation} == {"dev0", "dev1"}
    assert all(r.source == FailureSource.eval_miss for r in probation)
    assert bank.records("confirmed") == []  # mined records never gate as-is (A7)
    assert bank.verify_chain()


# -- the cascade, end to end ----------------------------------------------

def test_cascade_strictly_cuts_llm_invocations_at_equal_decisions(tmp_path):
    cases = generate_dataset(n_patients=16, images_per_patient=1, prevalence=0.5, seed=11)
    y = np.array([c.label for c in cases])
    pids = [c.patient_id for c in cases]

    teacher_spy = TracingSpyLLM(trace_writer=TraceWriter(tmp_path / "traces.jsonl"))
    teacher_spy.rule = RuleBasedAdjudicator(recall_threshold=0.9, deferral_band=(0.05, 0.9))
    teacher = _routed_pipeline(tmp_path / "teacher", teacher_spy)
    teacher_reports = [teacher.run_case(c.case_id, c.pixels) for c in cases]
    assert teacher_spy.calls >= 4  # the deferral band actually escalated cases

    X, yy, _ = load_traces(tmp_path / "traces.jsonl", min_rows=4)
    train, hold = train_holdout_split(len(X))
    student = fit_student(X[train], yy[train], kind="tree", max_depth=2, min_leaf=2)
    tau = confidence_threshold(agreement_report(student, X[hold], yy[hold]))
    assert tau is not None
    sid = save_student(student, tmp_path / "students")

    student_spy = TracingSpyLLM()
    cascade = _routed_pipeline(
        tmp_path / "cascade", student_spy,
        student=load_student(sid, tmp_path / "students"), tau=tau, sid=sid,
    )
    cascade_reports = [cascade.run_case(c.case_id, c.pixels) for c in cases]

    assert student_spy.calls < teacher_spy.calls  # spec: strictly fewer
    assert [r.decision for r in cascade_reports] == [r.decision for r in teacher_reports]
    assert [r.score for r in cascade_reports] == [r.score for r in teacher_reports]
    absorbed = [
        e["payload"]["student_absorbed"]
        for e in cascade.tools.ledger.entries()
        if e["kind"] == "route"
    ]
    assert sum(absorbed) == teacher_spy.calls - student_spy.calls

    # gate view of the same two arms: non-inferior, and cheaper in dollars
    def costs(reports):
        return [
            CaseCost(
                tool_calls=r.cost["total_tool_calls"], usd=r.cost["llm_usd"],
                wall_ms=r.cost["wall_ms"], tokens_in=r.cost["llm_input_tokens"],
                tokens_out=r.cost["llm_output_tokens"],
            )
            for r in reports
        ]

    result = run_gate(
        RULES, y,
        np.array([r.score for r in cascade_reports]),
        np.array([r.score for r in teacher_reports]),
        pids,
        cand_costs=costs(cascade_reports), champ_costs=costs(teacher_reports),
    )
    by_name = {c.name: c for c in result.checks}
    assert by_name["primary_metric_non_inferiority"].passed
    assert by_name["efficiency_floor"].passed
    mean_usd = lambda reports: np.mean([r.cost["llm_usd"] for r in reports])  # noqa: E731
    assert mean_usd(cascade_reports) < mean_usd(teacher_reports)


# -- promotion: through the gate or not at all -----------------------------

def test_student_promotion_goes_through_gate_and_registry(tmp_path):
    y, scores, pids = _cohort()
    registry = PolicyRegistry(root=tmp_path / "registry")

    champion_cfg = PolicyConfig(
        router={"fast_negative_max_score": 0.5, "llm_max_fraction": 1.0}
    )
    champ_id = registry.register(champion_cfg, note="U2 routed policy")
    first = run_gate(RULES, y, scores, None, pids)
    registry.promote(champ_id, first.summary(), first.passed)
    assert registry.champion() == champ_id

    rows = _grid_rows(
        lambda f: CaseDecision.recall if f["case_score"] >= 0.5 else CaseDecision.no_recall
    )
    X, yy = _matrix(rows)
    student = fit_student(X, yy, kind="tree", max_depth=2)
    sid = save_student(student, tmp_path / "registry" / "students")
    candidate_cfg = PolicyConfig(
        router={
            "fast_negative_max_score": 0.5,
            "llm_max_fraction": 1.0,
            "student_min_confidence": 0.98,
            "student_id": sid,
        }
    )
    cand_id = registry.register(candidate_cfg, parent_id=champ_id, note="U7 student band")

    # scores from the sealed stand-in; costs from the two measured arms:
    # the student arm answers the same cases with less LLM spend.
    champ_costs = [CaseCost(tool_calls=20, usd=0.02) for _ in range(40)]
    cand_costs = [CaseCost(tool_calls=20, usd=0.004) for _ in range(40)]
    result = run_gate(
        RULES, y, scores, scores.copy(), pids,
        candidate_scores_rerun=scores.copy(),
        cand_costs=cand_costs, champ_costs=champ_costs,
    )
    assert result.passed, result.summary()
    registry.promote(cand_id, result.summary(), result.passed)
    assert registry.champion() == cand_id
    assert registry.verify_chain()

    # the registered policy is runnable exactly as promoted
    tb = Toolbelt(ArtifactStore(tmp_path / "art"), EvidenceLedger(tmp_path / "led.jsonl"))
    pipeline = pipeline_from_policy(
        registry.get(cand_id), tb, student=load_student(sid, tmp_path / "registry" / "students")
    )
    assert pipeline.router is not None and pipeline.router.student is not None

    # and a cost-inflating student generation is refused BY THE GATE (S4)
    bloated = run_gate(
        RULES, y, scores, scores.copy(), pids,
        cand_costs=[CaseCost(tool_calls=20, usd=0.2) for _ in range(40)],
        champ_costs=champ_costs,
    )
    assert not bloated.passed
    with pytest.raises(PermissionError, match="gate did not pass"):
        registry.promote(champ_id, bloated.summary(), bloated.passed)
    assert registry.champion() == cand_id  # the pointer never moved


def test_ground_truth_gate_blocks_a_perfect_imitator(tmp_path):
    """Anti-Goodhart (S7): imitation proposes, ground truth disposes."""
    rows = _grid_rows(lambda f: CaseDecision.no_recall)  # a teacher that misses cancers
    X, yy = _matrix(rows)
    bad_student = fit_student(X, yy, kind="tree", max_depth=2)
    report = agreement_report(bad_student, X, yy)
    assert report["overall_agreement"] == 1.0  # the imitation itself is flawless
    tau = confidence_threshold(report)
    sid = save_student(bad_student, tmp_path / "students")

    # a confirmed failure record from the A7 truth channel: this cancer case
    # must be recalled; the champion (LLM arm) got it right.
    positive = next(
        c for c in generate_dataset(n_patients=6, images_per_patient=1, prevalence=1.0, seed=5)
        if c.label == 1
    )
    bank = FailureBank(root=tmp_path / "bank")
    rid = bank.add(
        FailureRecord(
            case_ref=positive.case_id, label=1,
            expected=CaseDecision.recall, observed=CaseDecision.defer_to_human,
            source=FailureSource.human_adjudication, status=FailureStatus.confirmed,
        )
    )

    spy = TracingSpyLLM()
    pipeline = _routed_pipeline(
        tmp_path / "cand", spy, student=load_student(sid, tmp_path / "students"),
        tau=tau, sid=sid,
    )
    bank_report = bank.replay(
        lambda ref, payload: pipeline.run_case(ref, payload),
        lambda ref: positive.pixels,
        policy_id="candidate",
    )
    assert bank_report.results[rid] is False  # the imitator no-recalls the cancer
    assert spy.calls == 0                     # ...without ever asking the LLM

    y, scores, pids = _cohort()
    result = run_gate(
        RULES, y, scores, scores.copy(), pids,
        failure_bank_report=bank_report, champion_bank_vector={rid: True},
    )
    by_name = {c.name: c for c in result.checks}
    assert not by_name["failure_bank_regression"].passed
    assert not result.passed
    registry = PolicyRegistry(root=tmp_path / "registry")
    cfg = PolicyConfig(router={"student_id": sid, "student_min_confidence": tau or 0.9})
    pid = registry.register(cfg)
    with pytest.raises(PermissionError, match="gate did not pass"):
        registry.promote(pid, result.summary(), result.passed)
    assert registry.champion() is None  # no silent replacement, ever


def test_student_cannot_be_silently_swapped(tmp_path):
    rows = _grid_rows(
        lambda f: CaseDecision.recall if f["case_score"] >= 0.5 else CaseDecision.no_recall
    )
    X, yy = _matrix(rows)
    student = fit_student(X, yy, kind="tree", max_depth=2)
    impostor = fit_student(X, (yy + 1) % len(CLASSES), kind="tree", max_depth=2)
    sid = student.student_id()

    with pytest.raises(ValueError, match="does not match the registered student_id"):
        DifficultyRouter(
            RouterConfig(student_min_confidence=0.9, student_id=sid), student=impostor
        )
    with pytest.raises(ValueError, match="registers its student_id"):
        DifficultyRouter(RouterConfig(student_min_confidence=0.9), student=student)
    tb = Toolbelt(ArtifactStore(tmp_path / "a"), EvidenceLedger(tmp_path / "l.jsonl"))
    cfg = PolicyConfig(router={"student_min_confidence": 0.9, "student_id": sid})
    with pytest.raises(ValueError, match="must run whole"):
        pipeline_from_policy(cfg, tb)  # a policy naming a student needs the student
    with pytest.raises(ValueError, match="ride a registered routing policy"):
        pipeline_from_policy(PolicyConfig(), tb, student=student)
