"""The PASS/FAIL promotion gate (T-3.1, axioms A6 and A8).

A candidate model (or scaffold change) is promoted only if EVERY check passes:
gates are conjunctive, never traded off against each other. Rules load from
``gates/gate_rules.yaml`` — a protected path the harness's own service account
cannot write (T-3.3): the loop is self-improving, never self-certifying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .metrics import (
    clustered_bootstrap_ci,
    ece_adaptive,
    paired_bootstrap_delta_ci,
    sensitivity_at_specificity,
)


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        lines = [f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks]
        lines.append(f"GATE: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def load_rules(path: str | Path = "gates/gate_rules.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def run_gate(
    rules: dict,
    y_true: np.ndarray,
    candidate_scores: np.ndarray,
    champion_scores: np.ndarray | None,
    patient_ids: list[str],
    subgroups: dict[str, list[str]] | None = None,
    candidate_scores_rerun: np.ndarray | None = None,
) -> GateResult:
    """Evaluate the conjunctive gate.

    ``subgroups`` maps subgroup-attribute name -> per-case values (e.g. site).
    ``candidate_scores_rerun`` is a second run on identical inputs for the
    determinism check.
    """
    result = GateResult()
    y_true = np.asarray(y_true)
    candidate_scores = np.asarray(candidate_scores)
    spec = float(rules["primary"]["specificity"])
    margin = float(rules["primary"]["non_inferiority_margin"])
    iters = int(rules["primary"].get("bootstrap_iterations", 1000))
    alpha = float(rules["primary"].get("alpha", 0.05))

    metric = lambda yt, sc: sensitivity_at_specificity(yt, sc, spec)  # noqa: E731

    # 1. Primary metric non-inferiority vs champion (paired clustered bootstrap).
    if champion_scores is None:
        cand_point, cand_lo, _ = clustered_bootstrap_ci(
            y_true, candidate_scores, patient_ids, metric, iterations=iters, alpha=alpha
        )
        result.checks.append(
            GateCheck(
                "primary_metric",
                passed=not np.isnan(cand_point),
                detail=f"no champion; candidate sens@{spec:.2f}spec = {cand_point:.3f} "
                f"(CI low {cand_lo:.3f})",
            )
        )
    else:
        delta, delta_lo, _ = paired_bootstrap_delta_ci(
            y_true, candidate_scores, np.asarray(champion_scores), patient_ids,
            metric, iterations=iters, alpha=alpha,
        )
        passed = delta_lo >= -margin
        result.checks.append(
            GateCheck(
                "primary_metric_non_inferiority",
                passed=passed,
                detail=f"paired delta {delta:+.3f} (CI low {delta_lo:+.3f}) >= -{margin}",
            )
        )

    # 2. Per-subgroup floors with a worst-slice margin.
    if subgroups:
        floor_margin = float(rules["subgroup_floors"]["worst_slice_margin"])
        min_n = int(rules["subgroup_floors"]["min_slice_size"])
        worst = np.inf
        worst_name = "-"
        for attr, values in subgroups.items():
            values = np.asarray(values)
            for level in np.unique(values):
                mask = values == level
                if mask.sum() < min_n or y_true[mask].sum() == 0:
                    continue
                s = metric(y_true[mask], candidate_scores[mask])
                if not np.isnan(s) and s < worst:
                    worst, worst_name = s, f"{attr}={level}"
        overall = metric(y_true, candidate_scores)
        passed = bool(worst == np.inf or worst >= overall - floor_margin)
        result.checks.append(
            GateCheck(
                "subgroup_worst_slice_floor",
                passed=passed,
                detail=f"worst slice {worst_name}: "
                f"{worst if worst != np.inf else float('nan'):.3f} vs overall {overall:.3f}",
            )
        )

    # 3. Calibration.
    max_ece = float(rules["calibration"]["max_ece_adaptive"])
    ece = ece_adaptive(y_true, candidate_scores)
    result.checks.append(
        GateCheck("calibration_ece", passed=ece <= max_ece, detail=f"ECE {ece:.4f} <= {max_ece}")
    )

    # 4. Negative-flip rate vs champion (regression ratchet).
    if champion_scores is not None:
        max_flip = float(rules["regression"]["max_negative_flip_rate"])
        champ_thr = np.quantile(np.asarray(champion_scores)[y_true == 0], spec)
        cand_thr = np.quantile(candidate_scores[y_true == 0], spec)
        champ_correct_pos = (np.asarray(champion_scores) > champ_thr) & (y_true == 1)
        cand_missed_pos = (candidate_scores <= cand_thr) & (y_true == 1)
        flips = champ_correct_pos & cand_missed_pos
        flip_rate = float(flips.sum() / max(champ_correct_pos.sum(), 1))
        result.checks.append(
            GateCheck(
                "negative_flip_rate",
                passed=flip_rate <= max_flip,
                detail=f"{flip_rate:.4f} <= {max_flip}",
            )
        )

    # 5. Determinism double-run agreement.
    if candidate_scores_rerun is not None:
        required = float(rules["determinism"]["required_agreement"])
        agreement = float(
            np.mean(np.isclose(candidate_scores, np.asarray(candidate_scores_rerun)))
        )
        result.checks.append(
            GateCheck(
                "determinism_double_run",
                passed=agreement >= required,
                detail=f"agreement {agreement:.4f} >= {required}",
            )
        )

    return result
