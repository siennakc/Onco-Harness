"""Sequential anytime-valid acceptance + efficiency floors (U8).

The headline test is the A/A calibration of the promotion machinery itself
(S14): a candidate that IS the champion plus noise must not be accepted in
more than an alpha fraction of nightly cycles, no matter how many looks
each cycle takes. Vectorized batch updates keep 1,000 simulated cycles
inside normal test runtime, so nothing here needs a slow marker.
"""

import copy
import json
from statistics import NormalDist

import numpy as np
import pytest
import yaml

from oncoharness.gate import run_gate
from oncoharness.ledger import EvidenceLedger
from oncoharness.sequential import (
    CaseCost,
    EProcessNonInferiority,
    ProposalAudit,
    gate_check_efficiency,
    gate_check_sequential,
)
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt

RULES = yaml.safe_load(open("gates/gate_rules.yaml"))


def _run_trial(rng, ep, budget, batch, sens, delta, prefix):
    """Feed one nightly trial's paired positive-case stream, batch by batch."""
    n_done = 0
    while n_done < budget and ep.decided == "continue":
        m = min(batch, budget - n_done)
        champ = (rng.random(m) < sens).astype(float)
        cand = (rng.random(m) < sens + delta).astype(float)
        pids = [f"{prefix}_{n_done + i}" for i in range(m)]
        ep.update(np.ones(m, dtype=int), cand, champ, pids)
        n_done += m
    return ep


# -- the e-process --------------------------------------------------------
def test_aa_false_acceptance_rate_below_alpha():
    """S14: champion-vs-champion cannot be noise-mined into a promotion.

    Promotion-grade acceptance is tested at margin=0 — "the candidate is
    not worse at all" — which puts an A/A candidate (true delta = 0)
    exactly on the null boundary, the sharpest point the Ville bound
    covers: P(wealth ever reaches 1/alpha) <= alpha over unlimited looks.
    (At the yaml's non-inferiority margin of 0.02 an A/A candidate is
    genuinely non-inferior, so acceptance there is correct behavior, not
    a false promotion — that case is exercised separately below.)
    """
    rng = np.random.default_rng(20260831)
    alpha, cycles = 0.05, 1000
    accepted = 0
    for cycle in range(cycles):
        ep = EProcessNonInferiority(margin=0.0, alpha=alpha)
        _run_trial(rng, ep, budget=2000, batch=100, sens=0.8, delta=0.0, prefix=f"aa{cycle}")
        accepted += ep.decided == "accept"
    rate = accepted / cycles
    assert rate <= alpha, f"A/A false-acceptance rate {rate:.3f} > alpha {alpha}"


def test_null_boundary_at_configured_margin():
    """A candidate worse by exactly the margin sits on H0's edge: accept <= alpha."""
    rng = np.random.default_rng(7)
    margin = float(RULES["sequential"]["margin"])
    alpha, cycles = 0.05, 300
    accepted = 0
    for cycle in range(cycles):
        ep = EProcessNonInferiority(margin=margin, alpha=alpha)
        _run_trial(rng, ep, budget=1000, batch=100, sens=0.8, delta=-margin, prefix=f"nb{cycle}")
        accepted += ep.decided == "accept"
    assert accepted / cycles <= alpha


def test_true_improvement_accepts_with_fewer_cases_than_fixed_n():
    """A real +5% sensitivity gain accepts early — the loop's own sample efficiency.

    Comparator: the fixed-n paired z-test powered at 90% for the same
    alternative, Bonferroni-corrected for the ~50 proposals a nightly loop
    makes between refreshes — the correction fixed-n testing must pay for
    repeated looks, and the e-process gets for free.
    """
    rng = np.random.default_rng(11)
    margin, alpha, sens, delta = 0.02, 0.05, 0.75, 0.05
    accepted_at = []
    cycles = 200
    for cycle in range(cycles):
        ep = EProcessNonInferiority(margin=margin, alpha=alpha)
        _run_trial(rng, ep, budget=2000, batch=50, sens=sens, delta=delta, prefix=f"h1{cycle}")
        if ep.decided == "accept":
            accepted_at.append(ep.accepted_at)
    assert len(accepted_at) / cycles >= 0.95

    z = NormalDist().inv_cdf
    var_x = (sens + delta) * (1 - sens) + sens * (1 - sens - delta) - delta**2
    n_fixed = (z(1 - alpha / 50) + z(0.9)) ** 2 * var_x / (delta + margin) ** 2
    median_n = float(np.median(accepted_at))
    assert median_n < 0.75 * n_fixed, (
        f"median cases-to-accept {median_n:.0f} not markedly below "
        f"Bonferroni fixed-n {n_fixed:.0f}"
    )


def test_evalue_growth_sanity_under_h1():
    rng = np.random.default_rng(3)
    ep = EProcessNonInferiority(margin=0.02, alpha=0.05)
    max_es, es = [], []
    for b in range(20):
        champ = (rng.random(100) < 0.7).astype(float)
        cand = (rng.random(100) < 0.8).astype(float)
        es.append(ep.update(np.ones(100, dtype=int), cand, champ,
                            [f"g{b}_{i}" for i in range(100)]))
        max_es.append(ep.max_e_value)
    assert all(b >= a for a, b in zip(max_es, max_es[1:]))  # sticky high-water mark
    assert es[-1] > 1.0 and es[-1] > es[0]
    assert ep.decided == "accept" and ep.accepted_at <= ep.n_clusters
    # Negatives carry no sensitivity information: a pure-negative batch is a no-op.
    before = (ep.e_value, ep.n_clusters)
    ep.update(np.zeros(5, dtype=int), np.ones(5), np.ones(5), [f"neg{i}" for i in range(5)])
    assert (ep.e_value, ep.n_clusters) == before


def test_duplicate_patient_clusters_are_refused():
    ep = EProcessNonInferiority(margin=0.02, alpha=0.05)
    ep.update(np.ones(2, dtype=int), np.ones(2), np.ones(2), ["P1", "P2"])
    with pytest.raises(ValueError, match="already bet on"):
        ep.update(np.ones(1, dtype=int), np.ones(1), np.zeros(1), ["P1"])


def test_eprocess_parameter_validation():
    with pytest.raises(ValueError):
        EProcessNonInferiority(margin=0.6, alpha=0.05)
    with pytest.raises(ValueError):
        EProcessNonInferiority(margin=0.02, alpha=0.0)


# -- efficiency floors ----------------------------------------------------
def _cohort(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    scores = 1.0 / (1.0 + np.exp(-4.0 * x))
    y = (rng.random(n) < scores).astype(int)
    return y, scores, [f"P{i//2}" for i in range(n)]


def test_cost_inflating_candidate_fails_the_conjunctive_gate():
    y, scores, pids = _cohort()
    champ_costs = [CaseCost(tool_calls=18, usd=0.010) for _ in range(200)]
    doubled = [CaseCost(tool_calls=18, usd=0.020) for _ in range(200)]
    result = run_gate(
        RULES, y, scores, champion_scores=scores.copy(), patient_ids=pids,
        cand_costs=doubled, champ_costs=champ_costs,
    )
    assert not result.passed  # equal accuracy, 2x cost: refused
    assert any(c.name == "efficiency_floor" and not c.passed for c in result.checks)

    parity = run_gate(
        RULES, y, scores, champion_scores=scores.copy(), patient_ids=pids,
        cand_costs=[CaseCost(tool_calls=17, usd=0.0102) for _ in range(200)],
        champ_costs=champ_costs,
    )
    assert parity.passed, parity.summary()

    tool_inflated = run_gate(
        RULES, y, scores, champion_scores=scores.copy(), patient_ids=pids,
        cand_costs=[CaseCost(tool_calls=25, usd=0.010) for _ in range(200)],
        champ_costs=champ_costs,
    )
    assert any(c.name == "efficiency_floor" and not c.passed for c in tool_inflated.checks)


def test_efficiency_check_fails_closed_without_data():
    with pytest.raises(ValueError, match="non-empty"):
        gate_check_efficiency([], [CaseCost(tool_calls=1)], RULES)
    # Zero-cost champion (local-only runs): calls still gate, $/case at parity.
    check = gate_check_efficiency(
        [CaseCost(tool_calls=10)], [CaseCost(tool_calls=10)], RULES
    )
    assert check.passed
    check = gate_check_efficiency(
        [CaseCost(tool_calls=10, usd=0.01)], [CaseCost(tool_calls=10)], RULES
    )
    assert not check.passed  # champion spent nothing; any spend is infinite inflation


# -- sequential wiring in the gate ---------------------------------------
def test_gate_sequential_check_wiring():
    rng = np.random.default_rng(5)
    accepted = _run_trial(
        rng, EProcessNonInferiority(margin=0.02, alpha=0.05),
        budget=2000, batch=100, sens=0.7, delta=0.1, prefix="w1",
    )
    assert gate_check_sequential(accepted, RULES).passed

    undecided = EProcessNonInferiority(margin=0.02, alpha=0.05)
    check = gate_check_sequential(undecided, RULES)
    assert not check.passed and "continue" in check.detail

    # The loop does not pick its own significance level: a looser e-process
    # is refused even when its wealth crossed the threshold.
    loose = _run_trial(
        rng, EProcessNonInferiority(margin=0.2, alpha=0.05),
        budget=2000, batch=100, sens=0.7, delta=0.1, prefix="w2",
    )
    refused = gate_check_sequential(loose, RULES)
    assert not refused.passed and "REFUSED" in refused.detail

    exhausted = _run_trial(
        rng, EProcessNonInferiority(margin=0.0, alpha=0.05),
        budget=2000, batch=200, sens=0.8, delta=0.0, prefix="w3",
    )
    if exhausted.decided == "continue":
        assert "budget exhausted" in gate_check_sequential(exhausted, RULES).detail


def test_gate_level_aa_no_false_promotion_across_cycles():
    """The full conjunctive gate, A/A, many nightly cycles: promotions <= alpha.

    Candidate scores ARE the champion scores, so every classic check passes
    by construction and the sequential check is the only thing standing
    between an equal-performance candidate and promotion — exactly the
    noise-mining scenario fixed-n repeated testing loses.
    """
    rules = copy.deepcopy(RULES)
    rules["primary"]["bootstrap_iterations"] = 50
    rules["sequential"]["max_cases_per_trial"] = 1000
    y, scores, pids = _cohort(n=400, seed=2)
    rng = np.random.default_rng(99)
    cycles, promoted = 100, 0
    for cycle in range(cycles):
        ep = EProcessNonInferiority(margin=0.0, alpha=0.05)
        _run_trial(rng, ep, budget=1000, batch=100, sens=0.8, delta=0.0, prefix=f"gl{cycle}")
        result = run_gate(
            rules, y, scores, champion_scores=scores.copy(), patient_ids=pids,
            eprocess=ep,
        )
        promoted += result.passed
    assert promoted / cycles <= 0.05, f"gate promoted A/A in {promoted}/{cycles} cycles"


def test_rules_without_new_keys_and_callers_without_new_args_are_unchanged():
    """Additive keying both ways: old rules ignore new inputs; new rules add no
    checks unless the new inputs are supplied (the 27 pre-existing tests)."""
    y, scores, pids = _cohort(n=200, seed=4)
    old_rules = copy.deepcopy(RULES)
    for key in ("efficiency", "sequential", "failure_bank"):
        old_rules.pop(key)
    with_extras = run_gate(
        old_rules, y, scores, champion_scores=scores.copy(), patient_ids=pids,
        cand_costs=[CaseCost(tool_calls=99, usd=9.9)], champ_costs=[CaseCost(tool_calls=1)],
        eprocess=EProcessNonInferiority(margin=0.02, alpha=0.05),
    )
    baseline = run_gate(RULES, y, scores, champion_scores=scores.copy(), patient_ids=pids)
    assert [c.name for c in with_extras.checks] == [c.name for c in baseline.checks]


# -- proposal audit -------------------------------------------------------
def test_proposal_audit_is_append_only_and_chain_verified(tmp_path):
    audit = ProposalAudit(tmp_path / "gate_audit.jsonl")
    for i, passed in enumerate([False, False, True, False]):
        audit.log(f"pol{i}", gate_passed=passed, e_value=float(i), checks={"primary": passed})
    assert audit.verify_chain()
    assert audit.promotion_rate(window=50) == 0.25
    assert audit.promotion_rate(window=2) == 0.5
    lines = audit.path.read_text().splitlines()
    lines[1] = lines[1].replace('"gate_passed": false', '"gate_passed": true')
    audit.path.write_text("\n".join(lines) + "\n")
    assert not EvidenceLedger(audit.path).verify_chain()


def test_run_eval_gate_counts_proposals_but_cannot_author_the_audit(tmp_path):
    y, scores, pids = _cohort(n=300, seed=6)
    results = {
        "policy_id": "cand_abc123",
        "y_true": y.tolist(),
        "candidate_scores": scores.tolist(),
        "champion_scores": scores.tolist(),
        "patient_ids": pids,
        "cand_costs": [{"tool_calls": 12, "usd": 0.01}] * 10,
        "champ_costs": [{"tool_calls": 12, "usd": 0.01}] * 10,
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))
    tb = Toolbelt(
        ArtifactStore(tmp_path / "artifacts"),
        EvidenceLedger(tmp_path / "ledger.jsonl"),
        gate_audit_path=tmp_path / "gate_audit.jsonl",
    )
    out = tb.call("run_eval_gate", results_path=str(results_path))

    audit = ProposalAudit(tmp_path / "gate_audit.jsonl")
    entries = [e for e in audit.entries() if e["kind"] == "gate_evaluation"]
    assert len(entries) == 1
    payload = entries[0]["payload"]
    # The entry records what the GATE said, not what any caller claimed.
    assert payload["policy_id"] == "cand_abc123"
    assert payload["gate_passed"] == out["passed"]
    assert payload["checks"]["efficiency_floor"] is True
    assert audit.verify_chain()

    # No audit-shaped tool exists, and the one gate tool takes no audit args.
    assert not any("audit" in name for name in tb._registry)
    with pytest.raises(TypeError):
        tb.call("run_eval_gate", results_path=str(results_path), gate_passed=True)

    # Without a policy_id there is no proposal to count: no entry is written.
    results.pop("policy_id")
    results_path.write_text(json.dumps(results))
    tb.call("run_eval_gate", results_path=str(results_path))
    assert len([e for e in audit.entries() if e["kind"] == "gate_evaluation"]) == 1
