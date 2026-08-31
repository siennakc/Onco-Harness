"""Metric implementations (T-1.4, Part 6 doctrine).

AUROC for selection; sensitivity at fixed specificity for release; partial AUC
in the high-specificity region; adaptive-bin ECE + Brier for calibration;
patient-level clustered bootstrap for uncertainty. Validated against known
closed-form cases in tests before any model work (T-1.4).
"""

from __future__ import annotations

import numpy as np


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (equivalent to Mann-Whitney U), ties handled."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # midranks for ties
    combined = np.concatenate([pos, neg])
    sorted_vals = combined[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_pos = ranks[: len(pos)].sum()
    u = rank_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def sensitivity_at_specificity(
    y_true: np.ndarray, scores: np.ndarray, specificity: float = 0.96
) -> float:
    """Sensitivity at the operating point achieving at least the target specificity."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    neg = np.sort(scores[y_true == 0])
    if len(neg) == 0 or (y_true == 1).sum() == 0:
        return float("nan")
    # Threshold = smallest value classifying >= specificity of negatives as negative.
    k = int(np.ceil(specificity * len(neg)))
    k = min(max(k, 1), len(neg))
    threshold = neg[k - 1]
    pos = scores[y_true == 1]
    return float((pos > threshold).mean())


def partial_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    min_specificity: float = 0.90,
    normalize: bool = True,
) -> float:
    """Partial AUC over the high-specificity region [min_specificity, 1.0]."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.unique(scores)[::-1]
    fpr_cap = 1.0 - min_specificity
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    n_pos = max(int((y_true == 1).sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)
    for t in thresholds:
        pred = scores >= t
        fpr = float((pred & (y_true == 0)).sum()) / n_neg
        tpr = float((pred & (y_true == 1)).sum()) / n_pos
        points.append((fpr, tpr))
    points.append((1.0, 1.0))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if x0 >= fpr_cap:
            break
        x1c = min(x1, fpr_cap)
        if x1c <= x0:
            continue
        # linear interpolation of tpr at x1c
        y1c = y0 + (y1 - y0) * ((x1c - x0) / (x1 - x0)) if x1 > x0 else y1
        area += (x1c - x0) * (y0 + y1c) / 2.0
    return float(area / fpr_cap) if normalize else float(area)


def brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs) - np.asarray(y_true)) ** 2))


def ece_adaptive(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Adaptive-bin (equal-mass) expected calibration error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(probs)
    bins = np.array_split(order, n_bins)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        ece += (len(b) / len(probs)) * abs(probs[b].mean() - y_true[b].mean())
    return float(ece)


def paired_bootstrap_delta_ci(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    patient_ids: list[str],
    metric_fn,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired clustered bootstrap of metric(a) - metric(b) on shared resamples.

    The paired form is what non-inferiority needs: an identical candidate has
    delta exactly 0 on every resample, instead of being penalized for the
    metric's own sampling variance.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    point = metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b)
    by_patient: dict[str, list[int]] = {}
    for i, pid in enumerate(patient_ids):
        by_patient.setdefault(pid, []).append(i)
    patients = sorted(by_patient)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([by_patient[p] for p in sampled])
        da = metric_fn(y_true[idx], scores_a[idx])
        db = metric_fn(y_true[idx], scores_b[idx])
        if not (np.isnan(da) or np.isnan(db)):
            deltas.append(da - db)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def clustered_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    patient_ids: list[str],
    metric_fn,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Patient-level clustered bootstrap: resample patients, keep their images together.

    Returns (point_estimate, ci_low, ci_high).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    point = metric_fn(y_true, scores)
    by_patient: dict[str, list[int]] = {}
    for i, pid in enumerate(patient_ids):
        by_patient.setdefault(pid, []).append(i)
    patients = sorted(by_patient)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(iterations):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([by_patient[p] for p in sampled])
        value = metric_fn(y_true[idx], scores[idx])
        if not np.isnan(value):
            stats.append(value)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
