"""Anytime-valid acceptance + efficiency floors for the gate (U8; A6, A8, S3/S4/S14, T-5.3).

A nightly loop proposing many candidates against a fixed-n bootstrap CI
will eventually promote noise: every re-look spends Type-I error the
one-shot test never budgeted. The fix is testing by betting — an
e-process. We bet against H0 ("the candidate's sensitivity is worse than
the champion's by more than ``margin``") with a capital process over the
paired per-patient outcome stream; wealth is a nonnegative supermartingale
under H0, so by Ville's inequality the chance it EVER reaches ``1/alpha``
is at most ``alpha`` — at any stopping time, over any number of looks,
across an unbounded stream of batches (Waudby-Smith & Ramdas-style
betting martingale; the machinery PACE built for self-evolving agents).
Good candidates accept early; a trial that never accepts simply ends when
its case budget runs out — there is no "reject forever", only "not
promoted tonight".

The bet sizes (``lambda``) are frozen at the start of each ``update``
batch from strictly-prior data (an aGRAPA/Kelly plug-in), which keeps the
process predictable, deterministic, and pure numpy.

Efficiency floors close the loop the other way (S4): without a cost
ceiling, "self-improvement" converges on "spend more". The efficiency
check fails a candidate whose mean per-case cost or tool-call count
inflates past the human-owned ratios in ``gates/gate_rules.yaml`` — and
it fails CLOSED on missing data, because a safeguard the optimizer can
starve of input is not a safeguard (the DGM lesson).

Proposal accounting lives in :class:`ProposalAudit`, a hash-chained log
written only by gate code: the optimizer can trigger a gate evaluation,
never author what the audit says about it (S8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .gate import GateCheck
from .ledger import EvidenceLedger


@dataclass(frozen=True)
class CaseCost:
    """Per-case cost snapshot (the U1 telemetry shape, minimally).

    ``tool_calls`` is always measurable locally; the money/token fields
    stay zero until an LLM adjudicator is in the loop.
    """

    tool_calls: int
    usd: float = 0.0
    wall_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


class EProcessNonInferiority:
    """Betting-martingale test of H0: candidate sens is worse than champion by > margin.

    Stream semantics: each call to :meth:`update` contributes one paired
    outcome per NEW patient-cluster — ``cand``/``champ`` are per-case
    correctness indicators (1 = handled correctly), only positive cases
    (``y_true == 1``) inform sensitivity, and a patient's cases collapse
    to one bounded payoff (their mean paired difference plus the margin).
    Re-presenting a patient already bet on is refused: paying the same
    evidence twice is how loops certify themselves.

    Accept H0's rejection (i.e. accept the candidate) when wealth ever
    reaches ``1/alpha``; the verdict is sticky. ``decided`` never says
    "reject" — the trial's case budget, owned by the gate rules, is what
    ends a losing night.
    """

    def __init__(self, margin: float, alpha: float) -> None:
        if not 0.0 <= margin < 0.5:
            raise ValueError("margin must be in [0, 0.5)")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.margin = float(margin)
        self.alpha = float(alpha)
        self._lambda_cap = 0.5 / (1.0 - self.margin)
        self._log_wealth = 0.0
        self._max_log_wealth = 0.0
        self._n_clusters = 0
        self._accepted_at: int | None = None
        # Kelly plug-in regularizers: one pseudo-observation at the
        # delta=0 alternative, plus a quarter of prior variance mass.
        self._sum_x = self.margin
        self._sum_x2 = self.margin**2 + 0.25
        self._seen: set[str] = set()

    # -- state ------------------------------------------------------------
    @property
    def e_value(self) -> float:
        return float(math.exp(min(self._log_wealth, 700.0)))

    @property
    def max_e_value(self) -> float:
        return float(math.exp(min(self._max_log_wealth, 700.0)))

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def accepted_at(self) -> int | None:
        """Cluster count at first threshold crossing, or None."""
        return self._accepted_at

    @property
    def decided(self) -> Literal["accept", "continue"]:
        threshold = math.log(1.0 / self.alpha)
        return "accept" if self._max_log_wealth >= threshold else "continue"

    # -- the bet ----------------------------------------------------------
    def update(
        self,
        y_true: np.ndarray,
        cand: np.ndarray,
        champ: np.ndarray,
        patient_ids: list[str],
    ) -> float:
        """Feed one batch of paired outcomes; returns the current e-value."""
        y_true = np.asarray(y_true)
        cand = np.asarray(cand, dtype=np.float64)
        champ = np.asarray(champ, dtype=np.float64)
        if not (len(y_true) == len(cand) == len(champ) == len(patient_ids)):
            raise ValueError("y_true, cand, champ, patient_ids must be equally long")

        pos = np.flatnonzero(np.asarray(y_true) == 1)
        if len(pos) == 0:
            return self.e_value
        d = cand[pos] - champ[pos]
        pids = np.asarray([patient_ids[i] for i in pos])
        uniq, first_idx, inverse = np.unique(pids, return_index=True, return_inverse=True)
        dup = self._seen.intersection(uniq.tolist())
        if dup:
            raise ValueError(
                f"patient cluster(s) already bet on: {sorted(dup)[:3]} — one paired "
                "outcome per cluster, ever (independence is the wager's collateral)"
            )
        self._seen.update(uniq.tolist())
        # One payoff per cluster (mean paired difference), in first-appearance order.
        sums = np.zeros(len(uniq))
        counts = np.zeros(len(uniq))
        np.add.at(sums, inverse, d)
        np.add.at(counts, inverse, 1.0)
        x = sums / counts + self.margin
        x = x[np.argsort(first_idx, kind="stable")]

        # lambda frozen at batch start from strictly-prior data (predictable).
        lam = self._sum_x / self._sum_x2
        lam = float(min(max(lam, 0.0), self._lambda_cap))

        contribs = np.log1p(lam * x)
        path = self._log_wealth + np.cumsum(contribs)
        threshold = math.log(1.0 / self.alpha)
        if self._accepted_at is None:
            crossed = np.flatnonzero(path >= threshold)
            if len(crossed):
                self._accepted_at = self._n_clusters + int(crossed[0]) + 1
        self._max_log_wealth = max(self._max_log_wealth, float(path.max()))
        self._log_wealth = float(path[-1])
        self._n_clusters += len(x)
        self._sum_x += float(x.sum())
        self._sum_x2 += float((x**2).sum())
        return self.e_value


class ProposalAudit:
    """Hash-chained ledger of every gate evaluation the loop ever requested.

    Sequential validity is only as honest as the proposal count, so the
    count lives where the optimizer cannot edit it: an append-only,
    hash-chained file written exclusively by gate code. The LLM-callable
    ``run_eval_gate`` tool can cause an entry to be written about the gate
    result it just received — it cannot author entry contents, and no
    Toolbelt tool exposes this class.
    """

    def __init__(self, path: str | Path = "runs/gate_audit.jsonl") -> None:
        self._ledger = EvidenceLedger(path)

    @property
    def path(self) -> Path:
        return self._ledger.path

    def log(
        self,
        policy_id: str,
        gate_passed: bool,
        e_value: float | None,
        checks: dict,
    ) -> str:
        return self._ledger.append(
            "gate_evaluation",
            {
                "policy_id": policy_id,
                "gate_passed": bool(gate_passed),
                "e_value": e_value,
                "checks": checks,
            },
        )

    def entries(self) -> list[dict]:
        return self._ledger.entries()

    def verify_chain(self) -> bool:
        return self._ledger.verify_chain()

    def promotion_rate(self, window: int = 50) -> float:
        """Fraction of the last ``window`` evaluations that passed the gate.

        Champion-vs-champion cycles (S14) should hold this at or below
        alpha; anything higher is a bug in the gate, not a discovery.
        """
        recent = [e for e in self.entries() if e["kind"] == "gate_evaluation"][-window:]
        if not recent:
            return 0.0
        return float(np.mean([e["payload"]["gate_passed"] for e in recent]))


def gate_check_efficiency(
    cand_costs: list[CaseCost],
    champ_costs: list[CaseCost],
    rules: dict,
) -> GateCheck:
    """FAIL when the candidate's mean $/case or tool-calls/case inflate past the caps.

    Fails closed: refusing to certify efficiency without cost data is the
    point — a check the optimizer can starve of telemetry is no check.
    """
    if not cand_costs or not champ_costs:
        raise ValueError("efficiency check requires non-empty cost lists for both arms")
    eff = rules["efficiency"]
    max_cost_ratio = float(eff["max_cost_ratio_vs_champion"])
    max_calls_ratio = float(eff["max_tool_calls_ratio"])

    def _ratio(cand_mean: float, champ_mean: float) -> float:
        if champ_mean <= 0.0:
            return 1.0 if cand_mean <= 0.0 else float("inf")
        return cand_mean / champ_mean

    cand_usd = float(np.mean([c.usd for c in cand_costs]))
    champ_usd = float(np.mean([c.usd for c in champ_costs]))
    cand_calls = float(np.mean([c.tool_calls for c in cand_costs]))
    champ_calls = float(np.mean([c.tool_calls for c in champ_costs]))
    usd_ratio = _ratio(cand_usd, champ_usd)
    calls_ratio = _ratio(cand_calls, champ_calls)
    passed = usd_ratio <= max_cost_ratio and calls_ratio <= max_calls_ratio
    return GateCheck(
        "efficiency_floor",
        passed=passed,
        detail=(
            f"$/case ratio {usd_ratio:.3f} <= {max_cost_ratio} "
            f"(cand {cand_usd:.4f} vs champ {champ_usd:.4f}); "
            f"tool-calls ratio {calls_ratio:.3f} <= {max_calls_ratio} "
            f"(cand {cand_calls:.1f} vs champ {champ_calls:.1f})"
        ),
    )


def gate_check_sequential(eprocess: EProcessNonInferiority, rules: dict) -> GateCheck:
    """Promotion requires the e-process to have ACCEPTED within its trial budget.

    The human-owned rules stay authoritative: an e-process configured
    looser than ``gates/gate_rules.yaml`` (larger alpha, wider margin)
    fails regardless of its wealth — the loop does not get to pick its
    own significance level.
    """
    seq = rules["sequential"]
    alpha_rule = float(seq["alpha"])
    margin_rule = float(seq["margin"])
    budget = int(seq["max_cases_per_trial"])

    config_ok = eprocess.alpha <= alpha_rule + 1e-12 and eprocess.margin <= margin_rule + 1e-12
    accepted = eprocess.decided == "accept"
    detail = (
        f"e-value {eprocess.e_value:.2f} (max {eprocess.max_e_value:.2f}, "
        f"accept at {1.0 / eprocess.alpha:.0f}), "
        f"{eprocess.n_clusters}/{budget} clusters, decided={eprocess.decided}"
    )
    if not config_ok:
        detail += (
            f"; REFUSED: e-process (alpha={eprocess.alpha}, margin={eprocess.margin}) "
            f"is looser than rules (alpha={alpha_rule}, margin={margin_rule})"
        )
    elif not accepted and eprocess.n_clusters >= budget:
        detail += "; trial budget exhausted without acceptance"
    return GateCheck("sequential_acceptance", passed=accepted and config_ok, detail=detail)
