"""U3 content-hash tool memoization: soundness scope, invalidation, replay speedup.

The claims under test are the spec's: byte-identical content hits, one flipped
pixel misses, a policy-level parameter bump misses, handle-producing results
never leak across runs, and the determinism double-run keeps agreement 1.0
while the detector body executes ~0 times on the replay — counted, not timed.
"""

import numpy as np
import pytest

from oncoharness.ledger import EvidenceLedger
from oncoharness.memo import ToolMemo, derive_tool_versions
from oncoharness.phantom import generate_dataset
from oncoharness.reference.detector import DoGBlobDetector
from oncoharness.state_machine import HarnessPipeline
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt


class CountingDetector(DoGBlobDetector):
    """Spy detector: counts propose() executions in a class-level tally.

    The tally lives on the class, not the instance, so ``vars(instance)``
    — which feeds the derived tool version — stays identical across runs.
    """

    executed = 0

    def propose(self, pixels):
        CountingDetector.executed += 1
        return super().propose(pixels)

    @classmethod
    def reset(cls):
        cls.executed = 0


def _belt(tmp_path, memo_path=None, detector=None, **detector_profile_kwargs):
    tb = Toolbelt(
        ArtifactStore(tmp_path / "artifacts"),
        EvidenceLedger(tmp_path / "ledger.jsonl"),
        detector=detector or CountingDetector(**detector_profile_kwargs),
    )
    tb.detector_profiles["blindspot"] = CountingDetector(
        sigma_small=3.0, sigma_large=9.0, score_threshold=0.30
    )
    if memo_path is not None:
        tb.memo = ToolMemo(memo_path, derive_tool_versions(tb))
    return tb


@pytest.fixture(autouse=True)
def _reset_spy():
    CountingDetector.reset()
    yield
    CountingDetector.reset()


def _pixels(seed=3):
    return generate_dataset(n_patients=1, images_per_patient=1, prevalence=1.0, seed=seed)[
        0
    ].pixels


def test_identical_content_executes_once_and_cites_cached_evidence(tmp_path):
    tb = _belt(tmp_path, memo_path=tmp_path / "memo.jsonl")
    pixels = _pixels()
    h1 = tb.store.put(pixels, kind="image").handle
    h2 = tb.store.put(pixels.copy(), kind="image").handle  # new handle, same bytes

    r1 = tb.call("run_detector", image_handle=h1)
    r2 = tb.call("run_detector", image_handle=h2)
    assert CountingDetector.executed == 1  # the body ran once, ever
    assert r2["candidates"] == r1["candidates"]
    assert r2["evidence_ref"] != r1["evidence_ref"]  # a hit is its own ledger entry

    cached = [e for e in tb.ledger.entries() if e["kind"] == "tool_result_cached"]
    assert len(cached) == 1
    assert cached[0]["payload"]["original_ref"] == r1["evidence_ref"]
    assert cached[0]["payload"]["tool"] == "run_detector"
    assert tb.ledger.verify_chain()

    # metered as a call AND a cached hit: the U1 invariant survives memoization
    cost = tb.meter.finish_case()
    assert cost.tool_calls["run_detector"] == 2
    assert cost.cached_hits == 1
    ledger_calls = [e for e in tb.ledger.entries() if e["kind"] == "tool_call"]
    assert cost.total_tool_calls() == len(ledger_calls) == 2


def test_one_flipped_pixel_misses(tmp_path):
    tb = _belt(tmp_path, memo_path=tmp_path / "memo.jsonl")
    pixels = _pixels()
    tampered = pixels.copy()
    tampered[7, 7] += 1e-3
    tb.call("run_detector", image_handle=tb.store.put(pixels, kind="image").handle)
    tb.call("run_detector", image_handle=tb.store.put(tampered, kind="image").handle)
    assert CountingDetector.executed == 2


def test_detector_param_bump_via_policy_invalidates(tmp_path):
    pixels = _pixels()
    tb1 = _belt(tmp_path / "a", memo_path=tmp_path / "memo.jsonl")
    tb1.call("run_detector", image_handle=tb1.store.put(pixels, kind="image").handle)
    assert CountingDetector.executed == 1

    # same memo file, same bytes, but the policy re-parameterizes the profile
    tb2 = _belt(tmp_path / "b", memo_path=None, sigma_small=2.5)
    tb2.memo = ToolMemo(tmp_path / "memo.jsonl", derive_tool_versions(tb2))
    tb2.call("run_detector", image_handle=tb2.store.put(pixels, kind="image").handle)
    assert CountingDetector.executed == 2  # clean miss, no stale detection

    # an explicit PolicyConfig tool_versions pin outranks the derived version
    from oncoharness.policy import PolicyConfig

    tb3 = _belt(tmp_path / "c", memo_path=None)
    pins = PolicyConfig(tool_versions={"run_detector": "recalibrated-v2"}).tool_versions
    tb3.memo = ToolMemo(tmp_path / "memo.jsonl", {**derive_tool_versions(tb3), **pins})
    tb3.call("run_detector", image_handle=tb3.store.put(pixels, kind="image").handle)
    assert CountingDetector.executed == 3


def test_handle_producing_tools_never_leak_cross_run(tmp_path):
    pixels = _pixels()
    memo_path = tmp_path / "memo.jsonl"

    tb1 = _belt(tmp_path / "run1", memo_path=memo_path)
    h1 = tb1.store.put(pixels, kind="image").handle
    c1 = tb1.call("crop_region", image_handle=h1, box=[10, 10, 40, 40])
    c1b = tb1.call("crop_region", image_handle=h1, box=[10, 10, 40, 40])
    assert c1b["crop_handle"] == c1["crop_handle"]  # in-run: same store, cached
    assert tb1.store.get(c1b["crop_handle"]).shape  # and the handle is live

    tb2 = _belt(tmp_path / "run2", memo_path=memo_path)  # same memo file, new store
    h2 = tb2.store.put(pixels, kind="image").handle
    c2 = tb2.call("crop_region", image_handle=h2, box=[10, 10, 40, 40])
    # a cross-run hit would have replayed run1's handle — dead in this store
    assert tb2.store.get(c2["crop_handle"]).shape
    cached = [e for e in tb2.ledger.entries() if e["kind"] == "tool_result_cached"]
    assert not cached  # the crop re-executed; nothing was served cross-run
    # and nothing handle-bearing was ever persisted to the shared memo file
    assert not memo_path.exists() or "crop_handle" not in memo_path.read_text()


def test_side_effect_and_gate_tools_are_never_memoized(tmp_path):
    tb = _belt(tmp_path, memo_path=tmp_path / "memo.jsonl")
    assert tb.memo.scope_of("submit_review") is None
    assert tb.memo.scope_of("run_eval_gate") is None
    assert tb.memo.scope_of("describe_store") is None
    assert tb.memo.scope_of("compare_prior") is None  # unlisted = deny-by-default
    tb.call("submit_review", case_id="c", reason="r", ranked_regions=[])
    tb.call("submit_review", case_id="c", reason="r", ranked_regions=[])
    kinds = [e["kind"] for e in tb.ledger.entries()]
    assert kinds.count("tool_result_cached") == 0
    # and the memo itself is not a registered tool the LLM could reach (S8)
    with pytest.raises(PermissionError):
        tb.call("memo")


def test_determinism_double_run_replays_with_zero_detector_executions(tmp_path):
    """Spec test 4: agreement 1.0 while detector executions drop >= 5x."""
    cases = generate_dataset(n_patients=8, images_per_patient=1, prevalence=0.4, seed=11)
    memo_path = tmp_path / "memo.jsonl"

    def run(root):
        tb = _belt(root, memo_path=memo_path)
        pipeline = HarnessPipeline(tb, consistency_reads=3, min_reproduced=2)
        return [pipeline.run_case(c.case_id, c.pixels) for c in cases]

    first = run(tmp_path / "run1")
    exec_first = CountingDetector.executed
    CountingDetector.reset()
    second = run(tmp_path / "run2")
    exec_second = CountingDetector.executed

    assert [r.score for r in first] == [r.score for r in second]        # agreement 1.0
    assert [r.decision for r in first] == [r.decision for r in second]
    assert exec_first >= 5 * max(exec_second, 1)
    assert exec_second == 0  # every detector read on the replay came from the memo
    hits = sum(r.cost["cached_hits"] for r in second)
    assert hits > 0 and all(r.cost["cached_hits"] > 0 for r in second)


def test_lru_cap_evicts_and_compacts_but_survives_reload(tmp_path):
    memo = ToolMemo(tmp_path / "memo.jsonl", {}, max_bytes=600)
    for i in range(8):
        memo.put(f"cross:{i:064d}", {"value": i, "evidence_ref": f"ref{i}"})
    assert (tmp_path / "memo.jsonl").stat().st_size <= 600
    assert memo.get("cross:" + "0" * 63 + "0") is None      # oldest evicted
    fresh = ToolMemo(tmp_path / "memo.jsonl", {})            # reload from disk
    newest = fresh.get(f"cross:{7:064d}")
    assert newest is not None and newest["result"]["value"] == 7
    assert newest["original_ref"] == "ref7"
