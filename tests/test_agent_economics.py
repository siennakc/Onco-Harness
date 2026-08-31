"""U6 LLM-node economics: compaction, budgets -> fallback, session reuse, traces."""

import asyncio
import json

import numpy as np
import pytest

from fake_sdk import FakeSDK, good_result, make_request

from oncoharness import agent as agent_mod
from oncoharness.agent import LLMAdjudicator, SYSTEM_PROMPT, compact_request
from oncoharness.ledger import EvidenceLedger
from oncoharness.schemas import CaseDecision
from oncoharness.state_machine import RuleBasedAdjudicator
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt
from oncoharness.traces import AdjudicationTrace, TraceWriter


def _toolbelt(tmp_path) -> Toolbelt:
    return Toolbelt(ArtifactStore(tmp_path / "a"), EvidenceLedger(tmp_path / "l.jsonl"))


def _adjudicator(tmp_path, monkeypatch, fake, **kwargs) -> tuple[LLMAdjudicator, TraceWriter]:
    monkeypatch.setattr(agent_mod, "_require_sdk", lambda: fake)
    writer = TraceWriter(tmp_path / "traces" / "adjudications.jsonl")
    tb = kwargs.pop("toolbelt", None) or _toolbelt(tmp_path)
    return LLMAdjudicator(tb, trace_writer=writer, **kwargs), writer


# -- compaction (E7) -----------------------------------------------------

def test_compact_request_golden():
    request = {
        "case_id": "case-7",
        "qc": "adequate",
        "candidates": [
            {
                "candidate_id": f"p{i}",
                "box": [np.int64(10 + i), 20, 40, 50],  # numpy ints, as the pipeline emits
                "score": round(0.91 - 0.09 * i, 6),
                "evidence_ref": "e" * 64,
                "reproduced": 2,
                "reproduced_fraction": 2 / 3,
                "kept": i < 3,
                "zoom_confirmed": i % 2 == 0,
                "veto_reason": None if i < 3 else "elongated background ridge",
                "equivalent_diameter_mm": 7.123456,
            }
            for i in range(4)
        ],
        "consistency": {"case_score": 0.8123456, "disagreement_rate": 0.0},
        "blindspot_candidates": [],
        "atlas_neighbors": [],
        "guideline_notes": [],
    }
    compact = compact_request(request, top_k=2)
    expected = {
        "case_id": "case-7",
        "qc": "adequate",
        "consistency": {"case_score": 0.812, "disagreement_rate": 0.0},
        "candidates": [
            {
                "candidate_id": "p0",
                "box": [10, 20, 40, 50],
                "score": 0.91,
                "reproduced_fraction": 0.667,
                "zoom_confirmed": True,
                "equivalent_diameter_mm": 7.123,
            },
            {
                "candidate_id": "p1",
                "box": [11, 20, 40, 50],
                "score": 0.82,
                "reproduced_fraction": 0.667,
                "zoom_confirmed": False,
                "equivalent_diameter_mm": 7.123,
            },
        ],
        "omitted_candidates": 2,
        "max_omitted_score": 0.73,
    }
    assert compact == expected
    # byte-stable: two invocations serialize identically, and are pure JSON
    dump = lambda: json.dumps(  # noqa: E731
        compact_request(request, top_k=2), sort_keys=True, separators=(",", ":")
    )
    assert dump() == dump()
    assert "evidence_ref" not in dump()
    # empty lists are dropped entirely
    assert "atlas_neighbors" not in compact and "blindspot_candidates" not in compact


def test_compact_request_keeps_top_k_by_score():
    request = make_request(n_candidates=12)
    compact = compact_request(request, top_k=8)
    assert len(compact["candidates"]) == 8
    scores = [c["score"] for c in compact["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert compact["omitted_candidates"] == 4
    assert compact["max_omitted_score"] == round(request["candidates"][8]["score"], 3)


# -- success path: trace + cost (E9, U1 test 2) --------------------------

def test_success_writes_trace_and_meters_usage(tmp_path, monkeypatch):
    fake = FakeSDK(
        script=[[good_result(
            usage={"input_tokens": 1200, "output_tokens": 90, "cache_read_input_tokens": 950},
            usd=0.021,
            turns=4,
        )]]
    )
    adj, writer = _adjudicator(tmp_path, monkeypatch, fake, max_usd_per_case=1.0)
    adj.toolbelt.meter.start_case("case-1")

    out = adj.adjudicate(make_request(case_score=0.9))
    cost = adj.toolbelt.meter.finish_case()

    assert out.decision == CaseDecision.recall  # the LLM verdict stood
    traces = writer.read_all()
    assert len(traces) == 1
    trace = traces[0]
    assert not trace.fallback_used and trace.fallback_reason == ""
    assert trace.input_tokens == 1200
    assert trace.output_tokens == 90
    assert trace.cache_read_tokens == 950
    assert trace.usd == 0.021
    assert trace.turns == 4
    assert trace.policy_id.startswith("pol:")
    assert trace.rule_decision == CaseDecision.recall  # rules agree at 0.9
    assert trace.agreed_with_rule
    assert trace.features["case_score"] == 0.9
    assert trace.wall_ms > 0.0
    # the same spend landed in the case's cost block (U1 integration)
    assert cost.llm_input_tokens == 1200
    assert cost.llm_output_tokens == 90
    assert cost.llm_cache_read_tokens == 950
    assert cost.llm_usd == 0.021
    assert cost.llm_turns == 4


# -- budgets and fallback (E8) -------------------------------------------

def test_usd_budget_breach_falls_back_to_rules(tmp_path, monkeypatch):
    fake = FakeSDK(script=[[good_result(usd=0.75)]])  # LLM said "recall", expensively
    adj, writer = _adjudicator(tmp_path, monkeypatch, fake, max_usd_per_case=0.10)

    out = adj.adjudicate(make_request(case_score=0.5))  # rules: inside deferral band

    assert out.decision == CaseDecision.defer_to_human  # not the LLM's confident recall
    trace = writer.read_all()[-1]
    assert trace.fallback_used
    assert "usd budget breached" in trace.fallback_reason
    assert trace.usd == 0.75  # the spend is still recorded honestly
    assert trace.decision == CaseDecision.defer_to_human
    assert trace.rule_decision == CaseDecision.defer_to_human


def test_output_token_budget_breach_falls_back(tmp_path, monkeypatch):
    fake = FakeSDK(script=[[good_result(usage={"input_tokens": 10, "output_tokens": 999})]])
    adj, writer = _adjudicator(tmp_path, monkeypatch, fake, max_output_tokens=100)
    out = adj.adjudicate(make_request(case_score=0.5))
    assert out.decision == CaseDecision.defer_to_human
    assert "output-token budget breached" in writer.read_all()[-1].fallback_reason


def test_invalid_structured_output_falls_back(tmp_path, monkeypatch):
    fake = FakeSDK(
        script=[[FakeSDK.ResultMessage(structured_output=None, result="not json {")]]
    )
    adj, writer = _adjudicator(tmp_path, monkeypatch, fake)
    out = adj.adjudicate(make_request(case_score=0.9))
    assert out.decision == CaseDecision.recall  # rule verdict for 0.9, not a half-answer
    trace = writer.read_all()[-1]
    assert trace.fallback_used and "JSONDecodeError" in trace.fallback_reason


def test_sdk_error_falls_back_and_custom_fallback_wins(tmp_path, monkeypatch):
    fake = FakeSDK(error=RuntimeError("stream exploded"))
    always_defer = RuleBasedAdjudicator(deferral_band=(0.0, 1.0))
    adj, writer = _adjudicator(tmp_path, monkeypatch, fake, fallback=always_defer)
    out = adj.adjudicate(make_request(case_score=0.2))  # rules alone would say no_recall
    assert out.decision == CaseDecision.defer_to_human  # the custom fallback answered
    trace = writer.read_all()[-1]
    assert trace.fallback_used and "stream exploded" in trace.fallback_reason
    assert trace.rule_decision == CaseDecision.defer_to_human


def test_importable_and_degrades_without_sdk(tmp_path):
    """The module must import — and budget/fallback logic run — with no SDK."""
    try:
        import claude_agent_sdk  # noqa: F401

        pytest.skip("claude-agent-sdk is installed; cannot exercise its absence")
    except ImportError:
        pass
    writer = TraceWriter(tmp_path / "t.jsonl")
    adj = LLMAdjudicator(_toolbelt(tmp_path), trace_writer=writer)
    out = adj.adjudicate(make_request(case_score=0.5))
    assert out.decision == CaseDecision.defer_to_human  # degrade toward deferral
    trace = writer.read_all()[0]
    assert trace.fallback_used
    assert "claude-agent-sdk is not installed" in trace.fallback_reason


# -- session reuse and cache-aligned prefix (E7) -------------------------

def test_consecutive_adjudications_reuse_server_and_stable_prefix(tmp_path, monkeypatch):
    fake = FakeSDK(script=[[good_result()], [good_result()]])
    adj, writer = _adjudicator(tmp_path, monkeypatch, fake)

    adj.adjudicate(make_request(case_id="case-a", n_candidates=12))
    adj.adjudicate(make_request(case_id="case-b"))

    assert fake.server_builds == 1  # one MCP server across the batch
    (prompt_a, options_a), (prompt_b, options_b) = fake.queries
    assert options_a is options_b  # one options object: the cacheable prefix
    assert options_a.system_prompt == SYSTEM_PROMPT  # no per-case content in it
    assert options_a.tools == [] and len(options_a.allowed_tools) == 7
    assert prompt_a != prompt_b  # the compacted request is the only volatile part
    assert '"case_id":"case-a"' in prompt_a and '"case_id":"case-b"' in prompt_b
    assert '"omitted_candidates":4' in prompt_a  # top-K compaction reached the wire
    assert "evidence_ref" not in prompt_a
    assert len(writer.read_all()) == 2
    adj.close()


def test_llm_tool_calls_route_through_toolbelt_and_are_sequenced(tmp_path, monkeypatch):
    fake = FakeSDK(script=[[good_result()]])
    adj, _ = _adjudicator(tmp_path, monkeypatch, fake)
    adj._ensure_session()
    tb = adj.toolbelt
    info = tb.store.put(np.zeros((32, 32), dtype=np.float32), kind="image")

    handlers = {t.name: t.handler for t in adj._server.tools}
    result = asyncio.run(
        handlers["measure"](
            {"image_handle": info.handle, "box": [0, 0, 10, 10], "pixel_spacing_mm": [0.1, 0.1]}
        )
    )
    assert "long_axis_mm" in result["content"][0]["text"]
    assert adj._tool_sequence == ["measure"]
    assert [e["payload"]["tool"] for e in tb.ledger.entries() if e["kind"] == "tool_call"] == [
        "measure"
    ]
    # a bad call is reported as a tool error, never an unhandled exception
    errored = asyncio.run(handlers["measure"]({"image_handle": "art:nope", "box": [0, 0, 1, 1],
                                               "pixel_spacing_mm": [0.1, 0.1]}))
    assert errored.get("is_error")


# -- trace log (E9) ------------------------------------------------------

def test_trace_log_appends_and_validates(tmp_path):
    path = tmp_path / "adjudications.jsonl"
    writer = TraceWriter(path)
    base = AdjudicationTrace(
        case_id="c1",
        decision=CaseDecision.recall,
        rule_decision=CaseDecision.recall,
        agreed_with_rule=True,
    )
    writer.write(base)
    writer.write(base.model_copy(update={"case_id": "c2", "fallback_used": True}))
    assert len(path.read_text().splitlines()) == 2
    # a fresh writer on the same path appends — never truncates
    TraceWriter(path).write(base.model_copy(update={"case_id": "c3"}))
    back = writer.read_all()  # schema-validated read-back
    assert [t.case_id for t in back] == ["c1", "c2", "c3"]
    assert back[1].fallback_used and not back[2].fallback_used
