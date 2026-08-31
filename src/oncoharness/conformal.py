"""Mondrian (class-conditional) split-conformal prediction (2G, axiom A13).

Marginal coverage is vacuous at screening prevalence — a set that covers 95%
of cases can still miss every cancer. Coverage is therefore guaranteed per
class: thresholds are calibrated separately on positive and negative
calibration cases (the disjoint calibration split, never train or test).

The prediction set drives deferral policy:
  {positive}            -> confident recall signal
  {negative}            -> confident no-recall signal
  {negative, positive}  -> ambiguous -> defer to human
  {}                    -> conforms to neither class -> OOD-flavored -> defer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MondrianConformal:
    q_pos: float   # nonconformity threshold calibrated on positives
    q_neg: float   # nonconformity threshold calibrated on negatives
    alpha: float

    @classmethod
    def fit(
        cls, cal_scores: np.ndarray, cal_labels: np.ndarray, alpha: float = 0.1
    ) -> "MondrianConformal":
        """``cal_scores`` are calibrated case-level suspicion scores in [0,1]."""
        cal_scores = np.asarray(cal_scores, dtype=np.float64)
        cal_labels = np.asarray(cal_labels)
        pos = cal_scores[cal_labels == 1]
        neg = cal_scores[cal_labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("conformal calibration needs both classes present")
        # Nonconformity: for the positive class 1 - s, for the negative class s.
        q_pos = _finite_sample_quantile(1.0 - pos, alpha)
        q_neg = _finite_sample_quantile(neg, alpha)
        return cls(q_pos=q_pos, q_neg=q_neg, alpha=alpha)

    def predict_set(self, score: float) -> set[int]:
        out: set[int] = set()
        if (1.0 - score) <= self.q_pos:
            out.add(1)
        if score <= self.q_neg:
            out.add(0)
        return out

    def is_ambiguous(self, score: float) -> bool:
        return len(self.predict_set(score)) != 1


def _finite_sample_quantile(nonconformity: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n empirical quantile with finite-sample correction."""
    n = len(nonconformity)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return float("inf")  # too few calibration cases: everything conforms
    return float(np.sort(nonconformity)[rank - 1])
