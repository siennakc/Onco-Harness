"""Evidence ledger integrity, artifact store handles, tool allowlist."""

import json

import numpy as np
import pytest

from oncoharness.ledger import EvidenceLedger
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt


def test_ledger_chain_verifies_and_detects_tampering(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    ledger.append("tool_call", {"tool": "run_detector"})
    ledger.append("tool_result", {"tool": "run_detector", "candidates": 2})
    assert ledger.verify_chain()

    # Tamper with the first entry's payload.
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    entry["payload"]["tool"] = "something_else"
    lines[0] = json.dumps(entry, sort_keys=True)
    (tmp_path / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    assert not EvidenceLedger(tmp_path / "ledger.jsonl").verify_chain()


def test_store_roundtrip_and_describe_has_no_pixels(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    arr = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    info = store.put(arr, kind="image", meta={"case_id": "c1"})
    assert np.array_equal(store.get(info.handle), arr)
    described = json.dumps(store.describe())
    # The description is text-safe: handles and facts only, never pixel values.
    assert info.handle in described
    assert "0.5" not in json.dumps([d.get("meta") for d in store.describe()])


def test_toolbelt_denies_unregistered_tool(tmp_path):
    tb = Toolbelt(ArtifactStore(tmp_path / "a"), EvidenceLedger(tmp_path / "l.jsonl"))
    with pytest.raises(PermissionError):
        tb.call("delete_everything")


def test_tool_calls_are_ledgered(tmp_path):
    tb = Toolbelt(ArtifactStore(tmp_path / "a"), EvidenceLedger(tmp_path / "l.jsonl"))
    img = np.zeros((64, 64), dtype=np.float32)
    img[20:30, 20:30] = 1.0
    info = tb.store.put(img, kind="image")
    result = tb.call("run_detector", image_handle=info.handle)
    assert "evidence_ref" in result
    kinds = [e["kind"] for e in tb.ledger.entries()]
    assert kinds == ["tool_call", "tool_result"]
    assert tb.ledger.verify_chain()


def test_measure_computes_mm_from_spacing(tmp_path):
    tb = Toolbelt(ArtifactStore(tmp_path / "a"), EvidenceLedger(tmp_path / "l.jsonl"))
    info = tb.store.put(np.zeros((100, 100), dtype=np.float32), kind="image")
    out = tb.call(
        "measure", image_handle=info.handle, box=[10, 10, 50, 30], pixel_spacing_mm=[0.5, 0.5]
    )
    assert out["width_mm"] == 20.0
    assert out["height_mm"] == 10.0
    assert out["long_axis_mm"] == 20.0
