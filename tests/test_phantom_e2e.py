"""End-to-end phantom run (T-3.2, T-4.5): the whole apparatus without real data.

Splits -> leakage audit -> detector+head -> harness pipeline -> reports ->
sealed scoring -> gate. Also proves the two standing invariants:
determinism (double-run agreement) and gates/ immutability during a run.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from oncoharness.phantom import generate_dataset
from oncoharness.splits import make_splits
from oncoharness.leakage import audit
from oncoharness.metrics import auroc
from oncoharness.ledger import EvidenceLedger
from oncoharness.schemas import CaseDecision
from oncoharness.state_machine import HarnessPipeline
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt

FRACTIONS = {
    "train": 0.6,
    "calibration": 0.1,
    "threshold": 0.1,
    "slice_discovery": 0.05,
    "test": 0.15,
}


@pytest.fixture(scope="module")
def phantom_world():
    cases = generate_dataset(n_patients=40, images_per_patient=1, prevalence=0.4, seed=11)
    patients = {c.patient_id: c.site for c in cases}
    manifest = make_splits(patients, FRACTIONS)
    report = audit(
        manifest.assignment, [(c.case_id, c.patient_id, c.pixels) for c in cases]
    )
    assert report.clean, f"phantom world leaked: {report}"
    return cases, manifest


def _pipeline(tmp_path) -> HarnessPipeline:
    tb = Toolbelt(ArtifactStore(tmp_path / "artifacts"), EvidenceLedger(tmp_path / "ledger.jsonl"))
    return HarnessPipeline(tb, consistency_reads=3, min_reproduced=2)


def test_pipeline_scores_beat_chance_on_phantoms(phantom_world, tmp_path):
    cases, manifest = phantom_world
    test_cases = [c for c in cases if manifest.split_of(c.patient_id) == "test"]
    # ensure both classes present in this tiny smoke slice
    if not any(c.label for c in test_cases) or all(c.label for c in test_cases):
        test_cases = cases[:10]
    pipeline = _pipeline(tmp_path)
    reports = [c.label for c in test_cases], [
        pipeline.run_case(c.case_id, c.pixels).score for c in test_cases
    ]
    y, scores = np.array(reports[0]), np.array(reports[1])
    assert auroc(y, scores) > 0.7, "harness should separate obvious phantom lesions"


def test_reports_are_structured_and_ledgered(phantom_world, tmp_path):
    cases, _ = phantom_world
    pipeline = _pipeline(tmp_path)
    positive = next(c for c in cases if c.label == 1)
    report = pipeline.run_case(positive.case_id, positive.pixels)
    assert report.decision in CaseDecision
    for finding in report.findings:
        assert finding.size_mm is not None          # measured by tool, never authored
        assert finding.evidence_refs                # every finding cites the ledger
    assert pipeline.tools.ledger.verify_chain()


def test_double_run_determinism(phantom_world, tmp_path):
    cases, _ = phantom_world
    case = cases[0]
    r1 = _pipeline(tmp_path / "run1").run_case(case.case_id, case.pixels)
    r2 = _pipeline(tmp_path / "run2").run_case(case.case_id, case.pixels)
    assert r1.score == r2.score
    assert r1.decision == r2.decision
    assert [f.box for f in r1.findings] == [f.box for f in r2.findings]


def test_blank_image_defers(tmp_path):
    pipeline = _pipeline(tmp_path)
    report = pipeline.run_case("blank", np.full((64, 64), 0.5, dtype=np.float32))
    assert report.decision == CaseDecision.defer_to_human
    assert report.deferral_reason


def test_gate_rules_are_never_written_by_a_run(phantom_world, tmp_path):
    rules_path = Path("gates/gate_rules.yaml")
    before = hashlib.sha256(rules_path.read_bytes()).hexdigest()
    cases, _ = phantom_world
    _pipeline(tmp_path).run_case(cases[0].case_id, cases[0].pixels)
    after = hashlib.sha256(rules_path.read_bytes()).hexdigest()
    assert before == after, "a pipeline run modified the protected gates/ path"
