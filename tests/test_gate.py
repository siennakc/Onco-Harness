"""The promotion gate (T-3.1): PASS/FAIL, non-inferiority, determinism."""

import numpy as np
import yaml

from oncoharness.gate import run_gate

RULES = yaml.safe_load(open("gates/gate_rules.yaml"))


def _cohort(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    # A strong model, calibrated by construction: p = sigmoid(4x), y ~ Bernoulli(p).
    # The gate's ECE check passes for an honest model and fails for a distorted one.
    x = rng.normal(0, 1, n)
    good = 1.0 / (1.0 + np.exp(-4.0 * x))
    y = (rng.random(n) < good).astype(int)
    pids = [f"P{i//2}" for i in range(n)]
    sites = ["site_a" if i % 2 else "site_b" for i in range(n)]
    return y, good, pids, sites


def test_gate_passes_equivalent_candidate():
    y, scores, pids, sites = _cohort()
    result = run_gate(
        RULES, y, scores, champion_scores=scores.copy(), patient_ids=pids,
        subgroups={"site": sites}, candidate_scores_rerun=scores.copy(),
    )
    assert result.passed, result.summary()


def test_gate_fails_degraded_candidate():
    y, good, pids, sites = _cohort()
    rng = np.random.default_rng(1)
    degraded = np.clip(good + rng.normal(0, 0.35, len(good)), 0, 1)  # much noisier
    result = run_gate(RULES, y, degraded, champion_scores=good, patient_ids=pids)
    assert not result.passed


def test_gate_fails_nondeterministic_candidate():
    y, scores, pids, _ = _cohort()
    jittered = scores + 1e-3
    result = run_gate(
        RULES, y, scores, champion_scores=scores.copy(), patient_ids=pids,
        candidate_scores_rerun=jittered,
    )
    assert any(c.name == "determinism_double_run" and not c.passed for c in result.checks)
