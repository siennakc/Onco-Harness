"""Light classification head + the calibration stack (T-2.1, T-2.2, axiom A10).

Logistic head fit by gradient descent on cached embeddings — seconds of CPU,
which is what makes nightly refits and DFR-style group-balanced refits cheap.
Includes temperature scaling and the analytic prior-correction logit shift
that must follow any rebalanced training (pitfall: rebalancing without prior
correction inflates outputs 50-200x).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class LogisticHead:
    weights: np.ndarray
    bias: float
    temperature: float = 1.0
    prior_logit_shift: float = 0.0

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        l2: float = 1e-3,
        lr: float = 0.5,
        epochs: int = 400,
        seed: int = 0,
        class_balanced: bool = False,
    ) -> "LogisticHead":
        rng = np.random.default_rng(seed)
        n, d = X.shape
        w = rng.normal(0, 0.01, d)
        b = 0.0
        sample_w = np.ones(n)
        if class_balanced:
            pos = max(y.sum(), 1)
            neg = max(n - y.sum(), 1)
            sample_w = np.where(y == 1, n / (2 * pos), n / (2 * neg))
        for _ in range(epochs):
            p = _sigmoid(X @ w + b)
            grad_z = (p - y) * sample_w / n
            w -= lr * (X.T @ grad_z + l2 * w)
            b -= lr * float(grad_z.sum())
        return cls(weights=w, bias=b)

    def logits(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return z / self.temperature + self.prior_logit_shift

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(X))

    def calibrate_temperature(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """Fit temperature on the disjoint calibration split by NLL line search."""
        raw = X_cal @ self.weights + self.bias
        best_t, best_nll = 1.0, np.inf
        for t in np.geomspace(0.1, 10.0, 120):
            p = _sigmoid(raw / t + self.prior_logit_shift)
            eps = 1e-9
            nll = -np.mean(y_cal * np.log(p + eps) + (1 - y_cal) * np.log(1 - p + eps))
            if nll < best_nll:
                best_t, best_nll = float(t), float(nll)
        self.temperature = best_t
        return best_t

    def apply_prior_correction(self, train_prevalence: float, true_prevalence: float) -> float:
        """Analytic logit shift restoring the deployment prior after rebalanced training."""
        eps = 1e-9
        shift = float(
            np.log(true_prevalence / (1 - true_prevalence + eps) + eps)
            - np.log(train_prevalence / (1 - train_prevalence + eps) + eps)
        )
        self.prior_logit_shift = shift
        return shift

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "weights": self.weights.tolist(),
                    "bias": self.bias,
                    "temperature": self.temperature,
                    "prior_logit_shift": self.prior_logit_shift,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticHead":
        raw = json.loads(Path(path).read_text())
        return cls(
            weights=np.array(raw["weights"]),
            bias=raw["bias"],
            temperature=raw["temperature"],
            prior_logit_shift=raw["prior_logit_shift"],
        )


def dfr_refit(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0
) -> LogisticHead:
    """Deep Feature Reweighting: refit the head on a group-balanced sample.

    The default subgroup remediation (axiom A11): seconds of CPU on frozen
    embeddings. ``groups`` is any registered subgroup labeling.
    """
    rng = np.random.default_rng(seed)
    idx_by_group: dict = {}
    for i, g in enumerate(groups):
        idx_by_group.setdefault(g, []).append(i)
    n_min = min(len(v) for v in idx_by_group.values())
    sel = np.concatenate(
        [rng.choice(v, size=n_min, replace=False) for v in idx_by_group.values()]
    )
    return LogisticHead.fit(X[sel], y[sel], class_balanced=True)
