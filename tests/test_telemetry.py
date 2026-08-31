"""U1 CostMeter: per-case cost accounting, report cost blocks, ablation columns."""

import json

import numpy as np

from oncoharness.ablation import format_table, run_ablation
from oncoharness.ledger import EvidenceLedger
from oncoharness.metrics import efficiency_summary
from oncoharness.phantom import generate_dataset
from oncoharness.state_machine import HarnessPipeline
from oncoharness.store import ArtifactStore
from oncoharness.telemetry import CostMeter
from oncoharness.tools import Toolbelt


def _pipeline(tmp_path, **toolbelt_kwargs) -> HarnessPipeline:
    tb = Toolbelt(
        ArtifactStore(tmp_path / "artifacts"),
        EvidenceLedger(tmp_path / "ledger.jsonl"),
        **toolbelt_kwargs,
    )
    return HarnessPipeline(tb, consistency_reads=3, min_reproduced=2)


def test_meter_accumulates_and_resets_between_cases():
    meter = CostMeter()
    meter.start_case("c1")
    meter.record_tool("run_detector", 12.5, pixels=64 * 64)
    meter.record_tool("run_detector", 7.5, pixels=64 * 64)
    meter.record_tool("measure", 1.0)
    meter.record_llm(
        {"input_tokens": 900, "output_tokens": 80, "cache_read_input_tokens": 700},
        usd=0.012,
        turns=3,
    )
    cost = meter.finish_case()
    assert cost.tool_calls == {"run_detector": 2, "measure": 1}
    assert cost.total_tool_calls() == 3
    assert cost.detector_pixels == 2 * 64 * 64
    assert cost.tool_wall_ms == 21.0
    assert cost.wall_ms > 0.0
    assert cost.llm_input_tokens == 900
    assert cost.llm_output_tokens == 80
    assert cost.llm_cache_read_tokens == 700
    assert cost.llm_turns == 3
    assert cost.llm_usd == 0.012
    # the next case starts from zero
    meter.start_case("c2")
    assert meter.finish_case().total_tool_calls() == 0


def test_meter_counts_cache_hits_separately():
    meter = CostMeter()
    meter.start_case("c1")
    meter.record_tool("run_detector", 5.0, cached=True)
    cost = meter.finish_case()
    assert cost.cached_hits == 1
    assert cost.tool_calls == {"run_detector": 1}  # a memo hit is still a call


def test_case_cost_matches_ledger_tool_calls(tmp_path):
    """The U1 invariant: cost.tool_calls totals == tool_call ledger entries."""
    case = generate_dataset(n_patients=2, images_per_patient=1, prevalence=1.0, seed=3)[0]
    pipeline = _pipeline(tmp_path)
    report = pipeline.run_case(case.case_id, case.pixels)

    assert report.cost is not None
    ledger_calls = [
        e["payload"]["tool"]
        for e in pipeline.tools.ledger.entries()
        if e["kind"] == "tool_call"
    ]
    by_name: dict[str, int] = {}
    for name in ledger_calls:
        by_name[name] = by_name.get(name, 0) + 1
    assert report.cost["tool_calls"] == by_name
    assert report.cost["total_tool_calls"] == len(ledger_calls)
    assert report.cost["detector_pixels"] > 0
    assert report.cost["wall_ms"] > 0.0
    assert report.cost["llm_usd"] == 0.0  # rule-based arm spends no dollars
    # the cost block reached the ledger's claim entry with the report
    claims = [e for e in pipeline.tools.ledger.entries() if e["kind"] == "claim"]
    assert claims[-1]["payload"]["cost"]["tool_calls"] == by_name


def test_qc_deferred_case_still_reports_cost(tmp_path):
    pipeline = _pipeline(tmp_path)
    report = pipeline.run_case("blank", np.full((64, 64), 0.5, dtype=np.float32))
    assert report.cost is not None
    assert report.cost["tool_calls"] == {"submit_review": 1}


def test_costs_jsonl_written_per_case(tmp_path):
    meter = CostMeter(jsonl_path=tmp_path / "runs" / "r1" / "costs.jsonl")
    pipeline = _pipeline(tmp_path, meter=meter)
    cases = generate_dataset(n_patients=2, images_per_patient=1, prevalence=0.5, seed=7)
    for case in cases:
        pipeline.run_case(case.case_id, case.pixels)
    rows = [
        json.loads(line)
        for line in (tmp_path / "runs" / "r1" / "costs.jsonl").read_text().splitlines()
    ]
    assert [r["case_id"] for r in rows] == [c.case_id for c in cases]
    assert all(r["total_tool_calls"] > 0 for r in rows)


def test_failed_tool_call_is_still_metered(tmp_path):
    """A refused profile leaves a tool_call ledger entry; the meter must match."""
    import pytest

    pipeline = _pipeline(tmp_path)
    tb = pipeline.tools
    info = tb.store.put(np.zeros((32, 32), dtype=np.float32), kind="image")
    with pytest.raises(ValueError):
        tb.call("run_detector", image_handle=info.handle, profile="rogue")
    ledger_calls = [e for e in tb.ledger.entries() if e["kind"] == "tool_call"]
    cost = tb.meter.finish_case()
    assert cost.total_tool_calls() == len(ledger_calls) == 1
    assert cost.detector_pixels == 0  # nothing was actually detected on


def test_ablation_table_includes_cost_columns():
    cases = generate_dataset(n_patients=12, images_per_patient=1, prevalence=0.4, seed=11)
    results = run_ablation(cases)
    table = format_table(results)
    header = table.splitlines()[0]
    assert header.split() == [
        "arm", "AUROC", "sens@96%spec", "n", "tools/case", "ms/case", "$/case",
    ]
    by_arm = {r.arm: r for r in results}
    assert by_arm["detector_alone"].mean_tool_calls == 1.0
    assert by_arm["harness"].mean_tool_calls > by_arm["detector_alone"].mean_tool_calls
    assert by_arm["harness"].mean_wall_ms > 0.0
    assert by_arm["harness"].mean_llm_usd == 0.0  # no LLM in the rule-based arms


def test_efficiency_summary_per_unit_costs():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    costs = [
        {"total_tool_calls": 250, "llm_usd": 0.05, "wall_ms": 30000.0},
        {"total_tool_calls": 250, "llm_usd": 0.05, "wall_ms": 30000.0},
    ]
    out = efficiency_summary(y, scores, costs)
    assert out["sens_at_96_spec"] == 1.0
    assert out["total_tool_calls"] == 500
    assert out["total_llm_usd"] == 0.1
    assert out["sens_per_1k_tool_calls"] == 2.0  # 1.0 sens / 0.5k calls
    assert out["sens_per_usd"] == 10.0           # 1.0 sens / $0.10
    assert out["sens_per_minute"] == 1.0         # 1.0 sens / 1 minute
    # zero denominators are None, never a fake infinity
    free = efficiency_summary(y, scores, [{"tool_calls": {"run_detector": 4}}])
    assert free["total_tool_calls"] == 4         # falls back to summing by name
    assert free["sens_per_usd"] is None
    assert free["sens_per_minute"] is None
