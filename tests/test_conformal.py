"""Mondrian conformal prediction: per-class coverage and ambiguity behavior."""

import numpy as np
import pytest

from oncoharness.conformal import MondrianConformal


def _calibrated_world(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    scores = 1.0 / (1.0 + np.exp(-3.0 * x))
    labels = (rng.random(n) < scores).astype(int)
    return scores, labels


def test_per_class_coverage_holds():
    scores, labels = _calibrated_world(seed=1)
    conf = MondrianConformal.fit(scores[:1000], labels[:1000], alpha=0.1)
    test_s, test_y = scores[1000:], labels[1000:]
    covered_pos = np.mean([1 in conf.predict_set(s) for s, y in zip(test_s, test_y) if y == 1])
    covered_neg = np.mean([0 in conf.predict_set(s) for s, y in zip(test_s, test_y) if y == 0])
    # Finite-sample: coverage should be close to (and typically above) 1 - alpha.
    assert covered_pos >= 0.85
    assert covered_neg >= 0.85


def test_extremes_are_singletons_and_middle_is_ambiguous():
    scores, labels = _calibrated_world()
    conf = MondrianConformal.fit(scores, labels, alpha=0.1)
    assert conf.predict_set(0.99) == {1}
    assert conf.predict_set(0.01) == {0}
    assert conf.is_ambiguous(0.5)  # mid scores conform to both classes


def test_single_class_calibration_refused():
    with pytest.raises(ValueError):
        MondrianConformal.fit(np.array([0.1, 0.2]), np.array([0, 0]))
