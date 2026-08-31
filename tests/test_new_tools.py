"""segment, compare_prior, run_eval_gate, detector profiles (T-4.1 completion)."""

import json

import numpy as np
import pytest

from oncoharness.ledger import EvidenceLedger
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt


def _toolbelt(tmp_path):
    return Toolbelt(ArtifactStore(tmp_path / "a"), EvidenceLedger(tmp_path / "l.jsonl"))


def _disc_image(size=96, cy=48, cx=48, r=10):
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size), 0.2, dtype=np.float32)
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = 0.9
    return img


def test_segment_disc_geometry(tmp_path):
    tb = _toolbelt(tmp_path)
    info = tb.store.put(_disc_image(), kind="image")
    out = tb.call(
        "segment", image_handle=info.handle, box=[33, 33, 63, 63], pixel_spacing_mm=[0.5, 0.5]
    )
    assert out["found"]
    # disc r=10 px at 0.5 mm/px -> diameter ~10 mm, high circularity
    assert 8.0 <= out["equivalent_diameter_mm"] <= 12.0
    assert out["circularity"] > 0.5


def test_segment_refuses_empty_center(tmp_path):
    tb = _toolbelt(tmp_path)
    img = np.full((64, 64), 0.2, dtype=np.float32)
    img[2:6, 2:6] = 1.0  # bright corner, dark center
    info = tb.store.put(img, kind="image")
    out = tb.call("segment", image_handle=info.handle, box=[20, 20, 50, 50])
    assert not out["found"]


def test_compare_prior_recovers_shift_and_reports_stability(tmp_path):
    tb = _toolbelt(tmp_path)
    current = _disc_image()
    prior = np.roll(np.roll(current, 4, axis=0), -3, axis=1)  # same anatomy, shifted
    cur = tb.store.put(current, kind="image")
    pri = tb.store.put(prior, kind="image")
    out = tb.call(
        "compare_prior", current_handle=cur.handle, prior_handle=pri.handle, box=[33, 33, 63, 63]
    )
    assert out["status"] == "compared"
    assert out["qc"]["passed"]
    # The tool reports the shift APPLIED to the prior to align it: the inverse
    # of the displacement we synthesized.
    assert out["qc"]["shift_px"] == [-4, 3]
    assert out["change"] == "stable_within_measurement_error"


def test_compare_prior_refuses_unregistrable_pair(tmp_path):
    tb = _toolbelt(tmp_path)
    rng = np.random.default_rng(0)
    cur = tb.store.put(rng.random((64, 64)).astype(np.float32), kind="image")
    pri = tb.store.put(rng.random((64, 64)).astype(np.float32), kind="image")
    out = tb.call(
        "compare_prior", current_handle=cur.handle, prior_handle=pri.handle, box=[10, 10, 30, 30]
    )
    assert out["status"] == "no_valid_correspondence"
    assert not out["qc"]["passed"]


def test_run_eval_gate_tool(tmp_path):
    tb = _toolbelt(tmp_path)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1200)
    scores = 1.0 / (1.0 + np.exp(-4.0 * x))
    y = (rng.random(1200) < scores).astype(int)
    results = {
        "y_true": y.tolist(),
        "candidate_scores": scores.tolist(),
        "champion_scores": scores.tolist(),
        "patient_ids": [f"P{i//2}" for i in range(1200)],
        "candidate_scores_rerun": scores.tolist(),
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))
    out = tb.call("run_eval_gate", results_path=str(path))
    assert out["passed"], out["summary"]
    assert "GATE: PASS" in out["summary"]


def test_blindspot_profile_exists_and_unknown_profile_refused(tmp_path):
    tb = _toolbelt(tmp_path)
    info = tb.store.put(_disc_image(), kind="image")
    out = tb.call("run_detector", image_handle=info.handle, profile="blindspot")
    assert out["profile"] == "blindspot"
    with pytest.raises(ValueError):
        tb.call("run_detector", image_handle=info.handle, profile="rogue")
