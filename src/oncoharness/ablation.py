"""T-4.5: the ablation that justifies the architecture.

Runs the same cases through (1) detector-alone and (2) the full harness, and
reports the primary metrics side by side — the MedRAX-style comparison. The
VLM-alone and LLM-adjudicated arms join once an adjudicator is attached; the
rule-based harness IS the detector-plus-verification arm, so this measures
what the verification machinery itself buys.

Cost columns are first-class (U1: E2): every arm reports mean tool calls,
wall ms, and LLM dollars per case beside its accuracy, so the ablation can
never trade invisible dollars for visible AUROC — what the harness buys must
be worth what the harness spends.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass

import numpy as np

from .phantom import PhantomCase
from .ledger import EvidenceLedger
from .state_machine import HarnessPipeline
from .store import ArtifactStore
from .telemetry import CostMeter
from .tools import Toolbelt
from .reference.detector import DoGBlobDetector
from .metrics import auroc, sensitivity_at_specificity


@dataclass
class ArmResult:
    arm: str
    auroc: float
    sensitivity_at_96_spec: float
    n_cases: int
    mean_tool_calls: float = 0.0
    mean_wall_ms: float = 0.0
    mean_llm_usd: float = 0.0


def run_arm_detector_alone(cases: list[PhantomCase]) -> tuple[np.ndarray, list[dict]]:
    """Arm 1: the raw specialist — top candidate score, no verification.

    Metered like any other arm: one detector read per case, so its cost row
    is the floor the harness's verification spend is judged against.
    """
    detector = DoGBlobDetector()
    meter = CostMeter()
    scores, costs = [], []
    for case in cases:
        meter.start_case(case.case_id)
        t0 = time.perf_counter()
        proposals = detector.propose(case.pixels)
        meter.record_tool(
            "run_detector", (time.perf_counter() - t0) * 1000.0, pixels=int(case.pixels.size)
        )
        scores.append(max((c.score for c in proposals), default=0.0))
        costs.append(meter.finish_case().as_dict())
    return np.array(scores), costs


def run_arm_harness(
    cases: list[PhantomCase], workdir: str | None = None
) -> tuple[np.ndarray, list[dict]]:
    """Arm 2: the full deterministic harness (TTA + zoom + FP/FN hunters)."""
    root = workdir or tempfile.mkdtemp(prefix="oncoharness_ablation_")
    pipeline = HarnessPipeline(
        Toolbelt(ArtifactStore(f"{root}/artifacts"), EvidenceLedger(f"{root}/ledger.jsonl")),
        consistency_reads=3,
        min_reproduced=2,
    )
    scores, costs = [], []
    for case in cases:
        report = pipeline.run_case(case.case_id, case.pixels)
        scores.append(report.score)
        costs.append(report.cost or {})
    return np.array(scores), costs


def run_arm_harness_routed(
    cases: list[PhantomCase], workdir: str | None = None
) -> tuple[np.ndarray, list[dict]]:
    """Arm 3 (U2): the same harness behind the difficulty router.

    Identical stack and thresholds as :func:`run_arm_harness`, plus a
    :class:`~oncoharness.router.RouterConfig` with its fast lane calibrated
    to the reference detector's score scale on phantoms (the DoG runs hot on
    textured backgrounds: negatives top out near 0.5, where calibrated real
    scores would sit near the 0.2 default) — so the delta between the two
    arms is exactly what difficulty gating buys, accuracy and cost side by
    side (E2/E3).
    """
    from .router import DifficultyRouter, RouterConfig

    root = workdir or tempfile.mkdtemp(prefix="oncoharness_ablation_routed_")
    pipeline = HarnessPipeline(
        Toolbelt(ArtifactStore(f"{root}/artifacts"), EvidenceLedger(f"{root}/ledger.jsonl")),
        consistency_reads=3,
        min_reproduced=2,
        router=DifficultyRouter(RouterConfig(fast_negative_max_score=0.5)),
    )
    scores, costs = [], []
    for case in cases:
        report = pipeline.run_case(case.case_id, case.pixels)
        scores.append(report.score)
        costs.append(report.cost or {})
    return np.array(scores), costs


def detector_alone_scores(cases: list[PhantomCase]) -> np.ndarray:
    """Back-compat wrapper: scores only."""
    return run_arm_detector_alone(cases)[0]


def harness_scores(cases: list[PhantomCase], workdir: str | None = None) -> np.ndarray:
    """Back-compat wrapper: scores only."""
    return run_arm_harness(cases, workdir)[0]


def _mean_cost(costs: list[dict], key: str, ndigits: int = 4) -> float:
    return round(float(np.mean([c.get(key, 0) for c in costs])), ndigits) if costs else 0.0


def run_ablation(cases: list[PhantomCase]) -> list[ArmResult]:
    y = np.array([c.label for c in cases])
    results = []
    for arm, (scores, costs) in (
        ("detector_alone", run_arm_detector_alone(cases)),
        ("harness", run_arm_harness(cases)),
        ("harness_routed", run_arm_harness_routed(cases)),
    ):
        results.append(
            ArmResult(
                arm=arm,
                auroc=round(auroc(y, scores), 4),
                sensitivity_at_96_spec=round(sensitivity_at_specificity(y, scores, 0.96), 4),
                n_cases=len(cases),
                mean_tool_calls=_mean_cost(costs, "total_tool_calls"),
                mean_wall_ms=_mean_cost(costs, "wall_ms"),
                mean_llm_usd=_mean_cost(costs, "llm_usd", ndigits=6),
            )
        )
    return results


def format_table(results: list[ArmResult]) -> str:
    lines = [
        f"{'arm':<16} {'AUROC':>8} {'sens@96%spec':>14} {'n':>5} "
        f"{'tools/case':>10} {'ms/case':>9} {'$/case':>8}"
    ]
    for r in results:
        lines.append(
            f"{r.arm:<16} {r.auroc:>8.4f} {r.sensitivity_at_96_spec:>14.4f} {r.n_cases:>5} "
            f"{r.mean_tool_calls:>10.1f} {r.mean_wall_ms:>9.1f} {r.mean_llm_usd:>8.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from .phantom import generate_dataset

    cases = generate_dataset(n_patients=40, images_per_patient=1, prevalence=0.4, seed=11)
    print(format_table(run_ablation(cases)))
