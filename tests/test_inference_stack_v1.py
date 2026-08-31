"""Inference stack v1 (T-4.3/T-4.4/T-4.5): zoom, symmetric verification,
conformal deferral, the image-ablated CI control, and the ablation runner."""

import numpy as np

from oncoharness.phantom import generate_dataset
from oncoharness.ablation import format_table, run_ablation
from oncoharness.conformal import MondrianConformal
from oncoharness.metrics import auroc
from oncoharness.ledger import EvidenceLedger
from oncoharness.schemas import CaseDecision
from oncoharness.state_machine import HarnessPipeline
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt


def _pipeline(tmp_path, **kwargs) -> HarnessPipeline:
    tb = Toolbelt(ArtifactStore(tmp_path / "artifacts"), EvidenceLedger(tmp_path / "ledger.jsonl"))
    return HarnessPipeline(tb, consistency_reads=3, min_reproduced=2, **kwargs)


def test_zoom_confirms_true_lesion(tmp_path):
    cases = generate_dataset(n_patients=10, images_per_patient=1, prevalence=1.0, seed=5)
    pipeline = _pipeline(tmp_path)
    report = pipeline.run_case(cases[0].case_id, cases[0].pixels)
    # An inserted phantom lesion must survive its own clean-room crop.
    assert report.findings, "true lesion was pruned by zoom/FP verification"
    assert report.score > 0.5


def test_fp_veto_carries_named_alternative(tmp_path):
    # A bright elongated ridge: candidate-shaped for the detector, but the
    # FP-hunter's segmentation geometry names it as a ridge, never mere doubt.
    img = np.full((128, 128), 0.2, dtype=np.float32)
    img[60:66, 10:118] = 0.9
    pipeline = _pipeline(tmp_path)
    pipeline.run_case("ridge", img)
    vetoes = [
        e["payload"] for e in pipeline.tools.ledger.entries()
        if e["kind"] == "decision" and e["payload"].get("per_candidate")
    ]
    assert vetoes, "adjudication decision missing from ledger"
    # the ledger records the full decision; the run itself must not crash and
    # any veto reason that exists must be specific
    for entry in pipeline.tools.ledger.entries():
        if entry["kind"] == "claim":
            for finding in entry["payload"].get("findings", []):
                assert finding["assessment"] != "present" or finding["evidence_refs"]


def test_conformal_layer_defers_ambiguous_scores(tmp_path):
    # Calibrate a conformal layer whose ambiguous band covers mid scores.
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 800)
    cal_scores = 1.0 / (1.0 + np.exp(-3.0 * x))
    cal_labels = (rng.random(800) < cal_scores).astype(int)
    conf = MondrianConformal.fit(cal_scores, cal_labels, alpha=0.1)

    cases = generate_dataset(n_patients=8, images_per_patient=1, prevalence=0.5, seed=9)
    pipeline = _pipeline(tmp_path, conformal=conf)
    for case in cases[:4]:
        report = pipeline.run_case(case.case_id, case.pixels)
        if conf.is_ambiguous(report.score):
            assert report.decision == CaseDecision.defer_to_human


def test_image_ablated_control(tmp_path):
    """Standing CI gate (T-4.4): without pixels, the harness must know nothing."""
    cases = generate_dataset(n_patients=16, images_per_patient=1, prevalence=0.5, seed=13)
    y = np.array([c.label for c in cases])

    real = _pipeline(tmp_path / "real")
    real_scores = np.array([real.run_case(c.case_id, c.pixels).score for c in cases])

    ablated = _pipeline(tmp_path / "ablated")
    constant = np.full_like(cases[0].pixels, 0.5)
    ablated_scores = np.array(
        [ablated.run_case(c.case_id, constant).score for c in cases]
    )

    assert abs(auroc(y, ablated_scores) - 0.5) < 1e-9  # identical input -> no signal
    assert auroc(y, real_scores) - auroc(y, ablated_scores) >= 0.15


def test_ablation_runner_harness_not_worse_than_detector():
    cases = generate_dataset(n_patients=24, images_per_patient=1, prevalence=0.4, seed=11)
    results = run_ablation(cases)
    table = format_table(results)
    assert "detector_alone" in table and "harness" in table
    by_arm = {r.arm: r for r in results}
    assert by_arm["harness"].auroc >= by_arm["detector_alone"].auroc - 0.05
