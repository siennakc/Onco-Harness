"""Adjudication distillation: LLM trajectories -> transparent student policy (U7: E4/E9; A4, A8, S7/S9).

This is the mechanism that makes the RSI loop *reduce* cost every cycle
instead of only raising accuracy: the LLM handles the frontier, and what it
decides consistently gets absorbed into a rules-level student the router can
trust — the cascade tier between the rule adjudicator and the LLM. The
student is deliberately the safest possible learner: a numpy logistic model
(one-vs-rest IRLS) or a depth-<=3 gini tree over the same deterministic
:func:`~oncoharness.traces.request_features` vector the traces record, fully
inspectable, serialized to content-addressed JSON under
``runs/registry/students/``, and adopted only as a registered policy change
(``student_id`` + threshold in ``PolicyConfig.router``) that passes the full
promotion gate.

Anti-Goodhart discipline (S7, the RLVR-gaming lesson): the student is
*trained* to imitate the LLM, but it is *promoted* against ground truth
through :func:`~oncoharness.gate.run_gate` — imitation is the proposal
mechanism, never the acceptance criterion. Self-verification is a mirage
(A4), so nothing in this module can move the champion pointer: fitting
returns a candidate, and the only road from candidate to production runs
through the gate's sens non-inferiority, failure-bank replay, and
efficiency-floor checks, then :meth:`~oncoharness.policy.PolicyRegistry.promote`.

The agreement band: :func:`agreement_report` bins held-out traces by the
student's confidence, and :func:`confidence_threshold` finds the lowest tau
whose bins all meet the required agreement with the LLM. The router hands an
escalated case to the student only at confidence >= tau; below it, the LLM
answers as before. High-confidence *disagreements* are mined into probation
:class:`~oncoharness.failure_bank.FailureRecord` rows (A7: probation, never
confirmed — the loop is not a truth channel), so every student generation
also sharpens the bank.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .failure_bank import FailureBank, FailureRecord, FailureSource, FailureStatus
from .schemas import CaseDecision
from .traces import TraceWriter

# The exact keys request_features emits, in emission order — the traces are
# the authoritative feature source (E9), so student and training data can
# never drift apart on feature semantics. (The research spec sketched a
# slightly different list; the built trace schema wins.)
FEATURES: tuple[str, ...] = (
    "case_score",
    "disagreement_rate",
    "n_candidates",
    "n_kept",
    "n_vetoed",
    "max_candidate_score",
    "mean_reproduced_fraction",
    "n_blindspot",
    "qc_adequate",
)

# Fixed class order: an index in y / a leaf count vector always means this.
CLASSES: tuple[str, ...] = (
    CaseDecision.recall.value,
    CaseDecision.no_recall.value,
    CaseDecision.defer_to_human.value,
)

DEFAULT_STUDENTS_ROOT = Path("runs/registry/students")
STUDENT_ID_LEN = 12


def load_traces(
    path: str | Path, min_rows: int = 200
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Feature matrix X and LLM decisions y (3-class) from the trace log (E9).

    Fallback rows are excluded: when the budget or the SDK failed, the
    recorded decision is the rule adjudicator's, and imitating the rule
    through a student would launder the fallback into "what the LLM does".
    Refuses to fit on fewer than ``min_rows`` usable rows — a student
    distilled from a handful of trajectories is noise wearing a threshold.
    """
    traces = [t for t in TraceWriter(path).read_all() if not t.fallback_used]
    if len(traces) < min_rows:
        raise ValueError(
            f"only {len(traces)} usable (non-fallback) traces at {path}; "
            f"need >= {min_rows} to distill"
        )
    X = np.array(
        [[float(t.features.get(f, 0.0)) for f in FEATURES] for t in traces], dtype=np.float64
    )
    y = np.array([CLASSES.index(t.decision.value) for t in traces], dtype=np.int64)
    return X, y, list(FEATURES)


def train_holdout_split(n: int, holdout_every: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic interleaved split: every ``holdout_every``-th row is held out.

    Traces arrive time-ordered; interleaving keeps both splits covering the
    same regime without any RNG in the promotion path.
    """
    idx = np.arange(n)
    hold = idx[idx % holdout_every == holdout_every - 1]
    train = idx[idx % holdout_every != holdout_every - 1]
    return train, hold


@dataclass(frozen=True)
class StudentPolicy:
    """A transparent, content-addressed adjudication student.

    ``kind`` is ``"logistic"`` (standardized one-vs-rest IRLS, softmax-
    normalized) or ``"tree"`` (depth-bounded greedy gini). Everything needed
    to predict lives in ``params`` as plain JSON — no pickles, no code — so
    the reachable student family is capacity-bounded by construction (S9)
    and any edit to the serialized file breaks the content hash.
    """

    kind: str
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    params: dict

    # -- identity ---------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "feature_names": list(self.feature_names),
                "classes": list(self.classes),
                "params": self.params,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str) -> "StudentPolicy":
        data = json.loads(text)
        if data["kind"] not in ("logistic", "tree"):
            raise ValueError(f"unknown student kind {data['kind']!r}")
        return cls(
            kind=data["kind"],
            feature_names=tuple(data["feature_names"]),
            classes=tuple(data["classes"]),
            params=data["params"],
        )

    def student_id(self) -> str:
        """sha256 of the canonical JSON, truncated: the student's one true name."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:STUDENT_ID_LEN]

    # -- inference --------------------------------------------------------
    def _vector(self, features: dict) -> np.ndarray:
        return np.array([float(features.get(f, 0.0)) for f in self.feature_names])

    def predict(self, features: dict) -> tuple[CaseDecision, float]:
        """(decision, confidence) for one deterministic feature dict."""
        idx, confidence = self._predict_row(self._vector(features))
        return CaseDecision(self.classes[idx]), float(confidence)

    def _predict_row(self, x: np.ndarray) -> tuple[int, float]:
        if self.kind == "logistic":
            mean = np.asarray(self.params["mean"])
            std = np.asarray(self.params["std"])
            std = np.where(std <= 0.0, 1.0, std)  # never divide by a stored zero
            xb = np.append((x - mean) / std, 1.0)
            z = np.array([float(xb @ np.asarray(w)) for w in self.params["weights"]])
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            probs = s / s.sum() if s.sum() > 0 else np.full(len(s), 1.0 / len(s))
        else:
            node = self.params["tree"]
            while "leaf" not in node:
                node = (
                    node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
                )
            counts = np.asarray(node["leaf"], dtype=np.float64)
            probs = counts / counts.sum()
        idx = int(np.argmax(probs))  # ties break to the lowest class index, always
        return idx, float(probs[idx])

    def predict_rows(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = [self._predict_row(np.asarray(row, dtype=np.float64)) for row in X]
        return np.array([i for i, _ in out]), np.array([c for _, c in out])


# -- fitting (pure numpy, deterministic) -----------------------------------

def fit_student(
    X: np.ndarray,
    y: np.ndarray,
    kind: str = "logistic",
    max_depth: int = 3,
    min_leaf: int = 5,
    ridge: float = 1e-3,
    iterations: int = 50,
) -> StudentPolicy:
    """Fit a candidate student; the caller decides its fate at the gate.

    Kind is a modeling choice, not a safety one — both are bounded and
    inspectable. Note the capability boundary: a one-vs-rest *linear*
    student cannot carve a middle band (e.g. "defer between 0.4 and 0.6")
    out of monotone features; banded teachers need the tree. The agreement
    report will say so honestly either way — a student that cannot learn
    the teacher simply earns no confidence band.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if kind == "logistic":
        params = _fit_logistic(X, y, len(CLASSES), ridge=ridge, iterations=iterations)
    elif kind == "tree":
        params = {"tree": _grow_tree(X, y, len(CLASSES), 0, max_depth, min_leaf)}
    else:
        raise ValueError(f"unknown student kind {kind!r}")
    return StudentPolicy(
        kind=kind, feature_names=FEATURES, classes=CLASSES, params=params
    )


def _fit_logistic(
    X: np.ndarray, y: np.ndarray, n_classes: int, ridge: float, iterations: int
) -> dict:
    """One-vs-rest binary logistic via IRLS, ridge-stabilized, standardized inputs."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0  # near-constant features scale to unit, not to noise
    Xb = np.hstack([(X - mean) / std, np.ones((len(X), 1))])
    d = Xb.shape[1]
    weights = []
    for k in range(n_classes):
        t = (y == k).astype(np.float64)
        if t.sum() == 0:  # class never seen: a constant, confident "no"
            w = np.zeros(d)
            w[-1] = -30.0
            weights.append(w.tolist())
            continue
        # Accelerate/BLAS raises spurious FP flags on saturated magnitudes
        # (the same artifact registration.py's matmuls show); correctness is
        # guarded inside the fit via explicit finite checks and best-iterate
        # revert, so the flags are noise here.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            weights.append(_irls_one_vs_rest(Xb, t, ridge, iterations))
    return {
        "mean": [round(float(v), 12) for v in mean],
        "std": [round(float(v), 12) for v in std],
        "weights": weights,
    }


def _irls_one_vs_rest(Xb: np.ndarray, t: np.ndarray, ridge: float, iterations: int) -> list[float]:
    """One ridge-IRLS binary fit; returns the best-deviance iterate's weights."""
    d = Xb.shape[1]
    w = np.zeros(d)
    best_w, best_loss = w.copy(), np.inf
    for _ in range(iterations):
        z = np.clip(Xb @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        loss = float(
            -(t * np.log(p + 1e-12) + (1.0 - t) * np.log(1.0 - p + 1e-12)).sum()
            + 0.5 * ridge * (w @ w)
        )
        # IRLS oscillates once the classes separate (the Hessian degenerates
        # as p saturates); keep the best iterate and stop the moment the
        # penalized deviance stops improving.
        if not np.isfinite(loss) or loss > best_loss - 1e-10:
            w = best_w
            break
        best_w, best_loss = w.copy(), loss
        if float(np.max(np.abs(t - p))) < 1e-6:
            break  # separated and saturated: more margin buys nothing
        r = p * (1.0 - p) + 1e-9
        H = (Xb * r[:, None]).T @ Xb + ridge * np.eye(d)
        g = Xb.T @ (t - p) - ridge * w
        step = np.linalg.solve(H, g)
        if not np.all(np.isfinite(step)):
            w = best_w
            break
        w = w + np.clip(step, -10.0, 10.0)
    return [round(float(v), 12) for v in w]


def _gini(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def _grow_tree(
    X: np.ndarray, y: np.ndarray, n_classes: int, depth: int, max_depth: int, min_leaf: int
) -> dict:
    """Greedy gini splits, axis-aligned, depth-bounded — inspectable by reading."""
    counts = np.bincount(y, minlength=n_classes)
    if depth >= max_depth or len(y) < 2 * min_leaf or counts.max() == len(y):
        return {"leaf": counts.tolist()}
    parent_gini = _gini(counts.astype(np.float64))
    best: tuple[float, int, float] | None = None
    for j in range(X.shape[1]):
        values = np.unique(X[:, j])
        if len(values) < 2:
            continue
        if len(values) > 33:  # bounded, deterministic threshold grid
            values = np.quantile(values, np.linspace(0.0, 1.0, 33))
        thresholds = (values[:-1] + values[1:]) / 2.0
        for thr in thresholds:
            left = X[:, j] <= thr
            nl = int(left.sum())
            nr = len(y) - nl
            if nl < min_leaf or nr < min_leaf:
                continue
            g = (
                nl * _gini(np.bincount(y[left], minlength=n_classes).astype(np.float64))
                + nr * _gini(np.bincount(y[~left], minlength=n_classes).astype(np.float64))
            ) / len(y)
            if g < parent_gini - 1e-12 and (best is None or g < best[0] - 1e-12):
                best = (g, j, float(thr))
    if best is None:
        return {"leaf": counts.tolist()}
    _, j, thr = best
    left = X[:, j] <= thr
    return {
        "feature": int(j),
        "threshold": round(float(thr), 12),
        "left": _grow_tree(X[left], y[left], n_classes, depth + 1, max_depth, min_leaf),
        "right": _grow_tree(X[~left], y[~left], n_classes, depth + 1, max_depth, min_leaf),
    }


# -- the agreement band ----------------------------------------------------

def agreement_report(
    student: StudentPolicy, X: np.ndarray, y: np.ndarray, bins: int = 10
) -> dict:
    """Per-confidence-bin agreement with the LLM on a held-out trace split.

    The bins are the evidence behind tau: agreement is measured where the
    student *claims* certainty, not on average, because the router will act
    on exactly those claims.
    """
    pred, conf = student.predict_rows(np.asarray(X, dtype=np.float64))
    y = np.asarray(y, dtype=np.int64)
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        mask = (conf >= lo) & ((conf < hi) if b < bins - 1 else (conf <= hi))
        n = int(mask.sum())
        rows.append(
            {
                "lo": round(lo, 6),
                "hi": round(hi, 6),
                "n": n,
                "agreement": round(float(np.mean(pred[mask] == y[mask])), 6) if n else None,
            }
        )
    overall = float(np.mean(pred == y)) if len(y) else 0.0
    return {"bins": rows, "n": int(len(y)), "overall_agreement": round(overall, 6)}


def confidence_threshold(report: dict, min_agreement: float = 0.98) -> float | None:
    """Lowest tau such that every populated bin at/above tau meets ``min_agreement``.

    Returns None when no band qualifies (or nothing was held out above any
    qualifying edge) — and None means the student is NOT used: absence of
    evidence is a routing decision, never a default-open door.
    """
    bins = report["bins"]
    for b in bins:
        tau = b["lo"]
        above = [r for r in bins if r["lo"] >= tau]
        populated = [r for r in above if r["n"]]
        if not populated:
            continue
        if all(r["agreement"] >= min_agreement for r in populated):
            return float(tau)
    return None


# -- registry-side persistence (content-addressed, like policies) ----------

def save_student(student: StudentPolicy, root: str | Path = DEFAULT_STUDENTS_ROOT) -> str:
    """Persist under the content hash; returns the id for ``PolicyConfig.router``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sid = student.student_id()
    path = root / f"{sid}.json"
    payload = student.to_json()
    if path.exists():
        if path.read_text() != payload:
            raise RuntimeError(f"student registry corruption: {path} does not match its id")
        return sid
    path.write_text(payload)
    return sid


def load_student(student_id: str, root: str | Path = DEFAULT_STUDENTS_ROOT) -> StudentPolicy:
    """Load and re-verify the content hash (tamper detection, as for policies)."""
    path = Path(root) / f"{student_id}.json"
    if not path.exists():
        raise KeyError(f"unknown student_id {student_id!r}")
    student = StudentPolicy.from_json(path.read_text())
    if student.student_id() != student_id:
        raise RuntimeError(
            f"student {student_id} was edited on disk; content now hashes to "
            f"{student.student_id()}"
        )
    return student


# -- disagreement mining (U5 coupling) -------------------------------------

def mine_disagreements(
    student: StudentPolicy,
    traces,
    tau: float,
    bank: FailureBank,
    labels: dict[str, int],
) -> list[str]:
    """High-confidence student-vs-LLM disagreements become probation bank records.

    A case where the student is confident AND contradicts the LLM is either
    a student blind spot or an LLM inconsistency — both worth locking. The
    record enters as ``eval_miss`` on probation (A7: the loop is not a truth
    channel; only :meth:`FailureBank.confirm` with human evidence promotes
    it), with the LLM's decision as ``expected`` and the student's as
    ``observed``. Cases without a ground-truth label are skipped — a bank
    record without a label would be a guess wearing a schema.
    """
    added: list[str] = []
    for trace in traces:
        if getattr(trace, "fallback_used", False):
            continue
        if trace.case_id not in labels:
            continue
        decision, confidence = student.predict(trace.features)
        if confidence < tau or decision == trace.decision:
            continue
        rid = bank.add(
            FailureRecord(
                case_ref=trace.case_id,
                label=int(labels[trace.case_id]),
                expected=trace.decision,
                observed=decision,
                policy_id=trace.policy_id,
                slice_tags={"origin": "distill_disagreement"},
                status=FailureStatus.probation,
                source=FailureSource.eval_miss,
            )
        )
        added.append(rid)
    return added
