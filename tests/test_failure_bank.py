"""Failure bank (U5): canonicalized growth, human-gated confirmation, replay gate."""

from types import SimpleNamespace

import pytest
import yaml

from oncoharness.failure_bank import (
    BankReport,
    FailureBank,
    FailureRecord,
    FailureSource,
    FailureStatus,
    gate_check_failure_bank,
)
from oncoharness.gate import run_gate
from oncoharness.ledger import EvidenceLedger
from oncoharness.schemas import CaseDecision

RULES = yaml.safe_load(open("gates/gate_rules.yaml"))


def _record(case_ref, embedding=None, source=FailureSource.human_adjudication, **kw):
    defaults = dict(
        case_ref=case_ref,
        label=1,
        expected=CaseDecision.recall,
        observed=CaseDecision.no_recall,
        policy_id="p0",
        slice_tags={"site": "site_a", "density_band": "d3"},
        embedding=embedding,
        source=source,
    )
    defaults.update(kw)
    return FailureRecord(**defaults)


# -- canonicalization -----------------------------------------------------
def test_near_duplicate_merges_distinct_appends(tmp_path):
    bank = FailureBank(tmp_path / "bank")
    rid = bank.add(_record("case_001", embedding=[1.0, 0.0, 0.0, 0.2]))
    # cosine ~0.98 within the same failure type -> merged, no new id.
    merged = bank.add(_record("case_002", embedding=[0.95, 0.05, 0.0, 0.25]))
    assert merged == rid
    assert bank.records("all")[0].cluster_size == 2
    assert len(bank.records("all")) == 1
    # A genuinely different failure (orthogonal embedding) -> new record.
    other = bank.add(_record("case_003", embedding=[0.0, 1.0, 0.0, 0.0]))
    assert other != rid
    assert len(bank.records("all")) == 2
    # Same embedding but a different failure mode never merges.
    fn_as_fp = bank.add(
        _record(
            "case_004",
            embedding=[1.0, 0.0, 0.0, 0.2],
            label=0,
            expected=CaseDecision.no_recall,
            observed=CaseDecision.recall,
        )
    )
    assert fn_as_fp not in (rid, other)
    # Exact re-report of a known case merges even without an embedding.
    assert bank.add(_record("case_001")) == rid
    by_id = {r.record_id: r for r in bank.records("all")}
    assert by_id[rid].cluster_size == 3


def test_bank_state_survives_reopen_and_chain_verifies(tmp_path):
    root = tmp_path / "bank"
    bank = FailureBank(root)
    rid = bank.add(_record("case_001", embedding=[1.0, 0.0]))
    bank.add(_record("case_002", embedding=[0.99, 0.05]))
    bank.confirm(rid, evidence_ref="ledger:abc123")

    reopened = FailureBank(root)
    rec = reopened.records("confirmed")[0]
    assert rec.record_id == rid and rec.cluster_size == 2
    assert reopened.verify_chain()
    # Tampering with history breaks the chain from that point forward.
    lines = reopened.ledger.path.read_text().splitlines()
    lines[0] = lines[0].replace("case_001", "case_00X")
    reopened.ledger.path.write_text("\n".join(lines) + "\n")
    assert not EvidenceLedger(reopened.ledger.path).verify_chain()


def test_active_cap_forces_curation(tmp_path):
    bank = FailureBank(tmp_path / "bank", active_cap=2)
    bank.add(_record("case_001"))
    rid2 = bank.add(_record("case_002"))
    with pytest.raises(ValueError, match="active cap"):
        bank.add(_record("case_003"))
    bank.retire(rid2, reason="superseded_by_cluster case_001")
    bank.add(_record("case_003"))  # room again after curation


# -- truth channels and typed records ------------------------------------
def test_eval_miss_cannot_be_born_confirmed_and_confirm_is_evidence_gated(tmp_path):
    with pytest.raises(ValueError, match="truth channel"):
        _record("case_009", source=FailureSource.eval_miss, status=FailureStatus.confirmed)
    bank = FailureBank(tmp_path / "bank")
    rid = bank.add(_record("case_009", source=FailureSource.eval_miss))
    assert bank.records("confirmed") == []
    with pytest.raises(ValueError, match="evidence"):
        bank.confirm(rid, evidence_ref="  ")
    bank.confirm(rid, evidence_ref="review:2026-08-30:reader2")
    assert [r.record_id for r in bank.records("confirmed")] == [rid]


def test_schema_refuses_prose_tags_and_non_failures():
    with pytest.raises(ValueError, match="constrained token"):
        _record("case_010", slice_tags={"note": "please always recall this case"})
    with pytest.raises(ValueError, match="not a failure"):
        _record("case_011", observed=CaseDecision.recall)


# -- replay + gate integration -------------------------------------------
def _threshold_policy(threshold, policy_id):
    """Stub policy: recall iff score >= threshold (stamped like a real report)."""

    def run_case(case_ref, payload):
        decision = (
            CaseDecision.recall if payload["score"] >= threshold else CaseDecision.no_recall
        )
        return SimpleNamespace(decision=decision, policy_id=policy_id)

    return run_case


CASES = {"case_001": {"score": 0.7}, "case_002": {"score": 0.9}}


def _seeded_bank(tmp_path):
    bank = FailureBank(tmp_path / "bank")
    r1 = bank.add(_record("case_001", embedding=[1.0, 0.0]))
    r2 = bank.add(_record("case_002", embedding=[0.0, 1.0]))
    bank.confirm(r1, evidence_ref="review:panel:case_001")
    return bank, r1, r2


def test_regression_replay_fails_gate_for_raised_threshold(tmp_path):
    bank, r1, r2 = _seeded_bank(tmp_path)
    load = CASES.__getitem__

    champ = bank.replay(_threshold_policy(0.5, "champ0"), load)
    assert champ.results == {r1: True, r2: True}

    # Deliberately raised recall threshold: the confirmed bank case regresses.
    cand = bank.replay(_threshold_policy(0.8, "cand1"), load)
    assert cand.results[r1] is False
    check = gate_check_failure_bank(cand, champ.results)
    assert not check.passed and r1 in check.detail

    # And through the conjunctive gate itself (rules key is additive).
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 600)
    scores = 1.0 / (1.0 + np.exp(-4.0 * x))
    y = (rng.random(600) < scores).astype(int)
    pids = [f"P{i//2}" for i in range(600)]
    good = run_gate(
        RULES, y, scores, champion_scores=scores.copy(), patient_ids=pids,
        failure_bank_report=champ, champion_bank_vector=champ.results,
    )
    assert good.passed, good.summary()
    gated = run_gate(
        RULES, y, scores, champion_scores=scores.copy(), patient_ids=pids,
        failure_bank_report=cand, champion_bank_vector=champ.results,
    )
    assert not gated.passed
    assert any(c.name == "failure_bank_regression" and not c.passed for c in gated.checks)


def test_probation_records_never_gate(tmp_path):
    # case_002 (score 0.9) is the confirmed record here; case_001 (score 0.7)
    # stays probationary, so a threshold of 0.8 regresses ONLY the probation one.
    bank = FailureBank(tmp_path / "bank")
    r1 = bank.add(_record("case_001", embedding=[1.0, 0.0]))
    r2 = bank.add(_record("case_002", embedding=[0.0, 1.0]))
    bank.confirm(r2, evidence_ref="review:panel:case_002")
    load = CASES.__getitem__
    champ = bank.replay(_threshold_policy(0.5, "champ0"), load)
    cand = bank.replay(_threshold_policy(0.8, "cand1"), load)
    assert cand.results == {r1: False, r2: True}
    check = gate_check_failure_bank(cand, champ.results)
    assert check.passed  # the probation miss is visible in replay, invisible to the gate
    assert cand.pass_rate == 1.0  # confirmed-only rate


def test_transfer_matrix_shows_fix_then_regression_flip(tmp_path):
    bank, r1, r2 = _seeded_bank(tmp_path)
    load = CASES.__getitem__
    bank.replay(_threshold_policy(0.5, "gen0"), load)
    bank.replay(_threshold_policy(0.8, "gen1"), load)
    matrix = bank.transfer_matrix()
    assert matrix["gen0"][r1] is True and matrix["gen1"][r1] is False
    flips = [
        rid
        for rid in matrix["gen0"]
        if matrix["gen0"][rid] and not matrix["gen1"][rid]
    ]
    assert flips == [r1]


def test_replay_takes_policy_id_from_stamped_reports(tmp_path):
    bank, r1, r2 = _seeded_bank(tmp_path)
    report = bank.replay(_threshold_policy(0.5, "stamped9"), CASES.__getitem__)
    assert report.policy_id == "stamped9"
    assert (bank.replays_dir / "stamped9.json").exists()


def test_retired_records_are_not_replayed(tmp_path):
    bank, r1, r2 = _seeded_bank(tmp_path)
    bank.retire(r2, reason="contribution_below_floor_100_trials")
    report = bank.replay(_threshold_policy(0.5, "champ0"), CASES.__getitem__)
    assert set(report.results) == {r1}


def test_empty_bank_report_passes_gate_check():
    check = gate_check_failure_bank(BankReport(policy_id="x"), {})
    assert check.passed
