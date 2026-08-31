"""A minimal stand-in for claude_agent_sdk, so LLM-node economics (budgets,
fallback, traces, session reuse) are testable without any LLM (U6).

Implements exactly the surface LLMAdjudicator touches: ``tool``,
``create_sdk_mcp_server``, ``ClaudeAgentOptions``, ``ResultMessage``, and
``query`` (an async generator). Each adjudication consumes one scripted
list of messages from ``script``; ``error`` makes the query raise instead.
"""

from __future__ import annotations

from types import SimpleNamespace


class FakeSDK:
    class ResultMessage:
        def __init__(
            self,
            structured_output=None,
            result="",
            usage=None,
            total_cost_usd=0.0,
            num_turns=1,
        ):
            self.structured_output = structured_output
            self.result = result
            self.usage = usage or {}
            self.total_cost_usd = total_cost_usd
            self.num_turns = num_turns

    def __init__(self, script=None, error=None):
        self.script = list(script or [])  # one list of messages per adjudication
        self.error = error
        self.server_builds = 0
        self.queries: list[tuple[str, object]] = []  # (prompt, options) per query

    def tool(self, name, description, schema):
        def decorator(handler):
            return SimpleNamespace(
                name=name, description=description, schema=schema, handler=handler
            )

        return decorator

    def create_sdk_mcp_server(self, name, version, tools):
        self.server_builds += 1
        return SimpleNamespace(name=name, version=version, tools=list(tools))

    def ClaudeAgentOptions(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def query(self, prompt, options):
        self.queries.append((prompt, options))
        error = self.error
        messages = self.script.pop(0) if self.script else []

        async def _gen():
            if error is not None:
                raise error
            for message in messages:
                yield message

        return _gen()


GOOD_STRUCTURED = {
    "per_candidate": {"p0": "present"},
    "decision": "recall",
    "rationale": "reproduced, zoom-confirmed finding",
    "cited_evidence": ["e0"],
}


def good_result(usage=None, usd=0.01, turns=2, structured=None):
    return FakeSDK.ResultMessage(
        structured_output=dict(GOOD_STRUCTURED if structured is None else structured),
        usage=usage
        or {"input_tokens": 1000, "output_tokens": 90, "cache_read_input_tokens": 800},
        total_cost_usd=usd,
        num_turns=turns,
    )


def make_request(case_id="case-1", case_score=0.9, disagreement=0.05, n_candidates=1):
    """A structurally faithful adjudication request (state_machine shape)."""
    candidates = [
        {
            "candidate_id": f"p{i}",
            "box": [10 + i, 20 + i, 40 + i, 50 + i],
            "score": round(case_score - 0.05 * i, 4),
            "evidence_ref": "f" * 64,
            "reproduced": 3,
            "reproduced_fraction": 1.0,
            "kept": True,
            "zoom_confirmed": True,
        }
        for i in range(n_candidates)
    ]
    return {
        "case_id": case_id,
        "qc": "adequate",
        "candidates": candidates,
        "consistency": {"case_score": case_score, "disagreement_rate": disagreement},
        "blindspot_candidates": [],
        "atlas_neighbors": [],
        "guideline_notes": [],
    }
