"""T-4.5: the ablation that justifies the architecture.

Runs the same cases through (1) detector-alone and (2) the full harness, and
reports the primary metrics side by side — the MedRAX-style comparison. The
VLM-alone and LLM-adjudicated arms join once an adjudicator is attached; the
rule-based harness IS the detector-plus-verification arm, so this measures
what the verification machinery itself buys.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass

import numpy as np

from .phantom import PhantomCase
from .ledger import EvidenceLedger
from .state_machine import HarnessPipeline
from .store import ArtifactStore
from .tools import Toolbelt
from .reference.detector import DoGBlobDetector
from .metrics import auroc, sensitivity_at_specificity


@dataclass
class ArmResult:
    arm: str
    auroc: float
    sensitivity_at_96_spec: float
    n_cases: int


def detector_alone_scores(cases: list[PhantomCase]) -> np.ndarray:
    """Arm 1: the raw specialist — top candidate score, no verification."""
    detector = DoGBlobDetector()
    return np.array(
        [max((c.score for c in detector.propose(case.pixels)), default=0.0) for case in cases]
    )


def harness_scores(cases: list[PhantomCase], workdir: str | None = None) -> np.ndarray:
    """Arm 2: the full deterministic harness (TTA + zoom + FP/FN hunters)."""
    root = workdir or tempfile.mkdtemp(prefix="oncoharness_ablation_")
    pipeline = HarnessPipeline(
        Toolbelt(ArtifactStore(f"{root}/artifacts"), EvidenceLedger(f"{root}/ledger.jsonl")),
        consistency_reads=3,
        min_reproduced=2,
    )
    return np.array([pipeline.run_case(c.case_id, c.pixels).score for c in cases])


def run_ablation(cases: list[PhantomCase]) -> list[ArmResult]:
    y = np.array([c.label for c in cases])
    results = []
    for arm, scores in (
        ("detector_alone", detector_alone_scores(cases)),
        ("harness", harness_scores(cases)),
    ):
        results.append(
            ArmResult(
                arm=arm,
                auroc=round(auroc(y, scores), 4),
                sensitivity_at_96_spec=round(sensitivity_at_specificity(y, scores, 0.96), 4),
                n_cases=len(cases),
            )
        )
    return results


def format_table(results: list[ArmResult]) -> str:
    lines = [f"{'arm':<16} {'AUROC':>8} {'sens@96%spec':>14} {'n':>5}"]
    for r in results:
        lines.append(
            f"{r.arm:<16} {r.auroc:>8.4f} {r.sensitivity_at_96_spec:>14.4f} {r.n_cases:>5}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from .phantom import generate_dataset

    cases = generate_dataset(n_patients=40, images_per_patient=1, prevalence=0.4, seed=11)
    print(format_table(run_ablation(cases)))
