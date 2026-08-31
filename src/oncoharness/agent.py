"""LLM adjudication via the Claude Agent SDK (T-4.1; U6: E7/E8/E9).

The LLM sits at exactly one decision node of the state machine: adjudication.
It receives facts and opaque handles — never pixels — and may call a small
set of read-only MCP tools to gather more evidence before returning a
structured :class:`~oncoharness.schemas.Adjudication`.

Defense in depth (Part 5):
1. Built-in tools are stripped (``tools=[]``) — no Bash, no file access.
2. ``allowed_tools`` names each permitted MCP tool explicitly; no wildcards.
3. The :class:`~oncoharness.tools.Toolbelt` registry re-checks every
   call in *our* code, so the boundary does not depend on SDK behavior.
4. Structured output is schema-enforced; free text never becomes a record.

LLM-node economics (U6):
1. Cache-aligned context (E7): the system prompt and tool schemas are the
   stable, cacheable prefix — per-case content never enters the system
   prompt. The compacted request JSON (:func:`compact_request`: top-K
   candidates, few keys, 3dp floats) is the only volatile suffix.
2. Session reuse (E7): the MCP server, options object, and event loop are
   built once and reused across consecutive adjudications, so prompt caching
   and connection reuse actually engage over a batch.
3. Hard budgets with graceful degradation (E8): on a USD or output-token
   budget breach — or an SDK error, or schema-invalid output — the
   deterministic fallback adjudicator answers instead. The rule-based
   fallback defers when uncertain, so degradation is toward *deferral*,
   never toward a truncated half-answer or a confident guess.
4. Trajectory log (E9): every adjudication, fallback included, writes one
   :class:`~oncoharness.traces.AdjudicationTrace` row — spend, tool
   sequence, and the decision beside the rule baseline.

Install with ``pip install -e '.[agent]'`` and authenticate the Anthropic SDK
environment before use. The module imports — and every budget/fallback path
runs — without the SDK installed: only a real LLM call requires it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import numpy as np

from .schemas import Adjudication
from .state_machine import Adjudicator, RuleBasedAdjudicator
from .telemetry import CostMeter
from .tools import Toolbelt
from .traces import AdjudicationTrace, TraceWriter, request_features

SYSTEM_PROMPT = """You are the adjudication node of Oncoscope, a medical-imaging
analysis harness. You decide, for one case, whether verified detector findings
warrant recall, no recall, or deferral to a human reader.

Hard rules you must never break:
- You never author pixels, coordinates, or numbers. Every quantitative value
  you cite must come from a tool result and be referenced by its evidence id.
- Abstention is a first-class output: when evidence is thin, conflicting, or
  the atlas has no near neighbors, defer to a human with a specific reason.
- To raise confidence in any finding you must cite a NEW tool call.
- An FP-hunter needs a specific alternative explanation (not mere doubt) to
  overturn a reproduced finding; re-search apparent negatives before agreeing.

Return ONLY the structured adjudication object."""

# Read-only evidence-gathering tools only. Deliberately absent: submit_review
# (deferral is the state machine's outcome, not a side effect the adjudicator
# can trigger mid-thought) and run_eval_gate (the gate belongs to the
# improvement loop; the adjudicator must not query its own promotion machinery).
_ALLOWED_TOOLS = (
    "describe_store",
    "crop_region",
    "segment",
    "measure",
    "compare_prior",
    "retrieve_similar",
    "lookup_criteria",
)

# The only candidate fields the LLM needs to adjudicate; everything else
# (evidence refs, bookkeeping flags) stays code-side and is retrievable
# through tools if the LLM decides it must look.
_COMPACT_CANDIDATE_KEYS = (
    "candidate_id",
    "box",
    "score",
    "reproduced_fraction",
    "zoom_confirmed",
    "veto_reason",
    "equivalent_diameter_mm",
)


def _require_sdk():
    try:
        import claude_agent_sdk  # noqa: F401

        return claude_agent_sdk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "claude-agent-sdk is not installed; install with pip install -e '.[agent]' "
            "or use the default RuleBasedAdjudicator"
        ) from exc


def _jsonable(obj: Any, ndigits: int = 3) -> Any:
    """Rounded, JSON-safe copy: floats to ``ndigits``, numpy scalars unboxed."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), ndigits)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, ndigits) for v in obj]
    return obj


def compact_request(request: dict, top_k: int = 8) -> dict:
    """Token-thrifty volatile suffix (E7): what the LLM actually needs, no more.

    Deterministic and byte-stable: candidates sorted score-desc (candidate_id
    breaks ties), only the top ``top_k`` kept, each reduced to
    ``_COMPACT_CANDIDATE_KEYS``, floats rounded to 3dp, empty lists dropped.
    The omitted tail is summarized as counts — compaction never silently
    discards the fact that something was discarded.
    """
    candidates = sorted(
        request.get("candidates", []),
        key=lambda c: (-float(c.get("score", 0.0)), str(c.get("candidate_id", ""))),
    )
    kept, omitted = candidates[:top_k], candidates[top_k:]

    compact: dict = {}
    for key in ("case_id", "qc", "lane"):
        if key in request:
            compact[key] = request[key]
    compact["consistency"] = _jsonable(dict(request.get("consistency", {})))
    kept_compact = [
        _jsonable({k: c[k] for k in _COMPACT_CANDIDATE_KEYS if c.get(k) is not None})
        for c in kept
    ]
    if kept_compact:
        compact["candidates"] = kept_compact
    blindspot = [
        _jsonable({"box": b.get("box"), "score": b.get("score", 0.0)})
        for b in request.get("blindspot_candidates", [])
    ]
    if blindspot:
        compact["blindspot_candidates"] = blindspot
    for key in ("atlas_neighbors", "guideline_notes"):
        if request.get(key):
            compact[key] = _jsonable(request[key])
    if omitted:
        compact["omitted_candidates"] = len(omitted)
        compact["max_omitted_score"] = round(
            max(float(c.get("score", 0.0)) for c in omitted), 3
        )
    return compact


class LLMAdjudicator:
    """Drop-in :class:`Adjudicator` backed by a Claude agent, economically managed.

    Budgets (``max_usd_per_case``, ``max_output_tokens``, ``max_turns``) are
    enforced at run time; on breach — or any SDK/schema failure — the
    ``fallback`` adjudicator (default: :class:`RuleBasedAdjudicator`) answers,
    and the trace records ``fallback_used`` with a code-authored reason. The
    meter defaults to the toolbelt's own, so LLM spend lands in the same
    per-case cost block as tool spend (U1).
    """

    def __init__(
        self,
        toolbelt: Toolbelt,
        model: str = "claude-opus-5",
        max_turns: int = 24,
        max_usd_per_case: float | None = None,
        max_output_tokens: int | None = None,
        top_k_candidates: int = 8,
        meter: CostMeter | None = None,
        trace_writer: TraceWriter | None = None,
        fallback: Adjudicator | None = None,
        policy_id: str | None = None,
    ) -> None:
        self.toolbelt = toolbelt
        self.model = model
        self.max_turns = max_turns
        self.max_usd_per_case = max_usd_per_case
        self.max_output_tokens = max_output_tokens
        self.top_k_candidates = top_k_candidates
        self.meter = meter if meter is not None else getattr(toolbelt, "meter", None)
        self.trace_writer = trace_writer
        self.fallback: Adjudicator = fallback if fallback is not None else RuleBasedAdjudicator()
        self._rule = (
            self.fallback
            if isinstance(self.fallback, RuleBasedAdjudicator)
            else RuleBasedAdjudicator()
        )
        self.policy_id = policy_id or self._default_policy_id()
        # Session state, built once and reused across a batch of cases (E7).
        self._sdk: Any = None
        self._server: Any = None
        self._options: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tool_sequence: list[str] = []

    def _default_policy_id(self) -> str:
        """Content-derived id until the policy registry lands (A8 forward-compat)."""
        body = json.dumps(
            {
                "model": self.model,
                "max_turns": self.max_turns,
                "max_usd_per_case": self.max_usd_per_case,
                "max_output_tokens": self.max_output_tokens,
                "top_k_candidates": self.top_k_candidates,
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            },
            sort_keys=True,
        )
        return "pol:" + hashlib.sha256(body.encode()).hexdigest()[:12]

    # -- MCP server over the toolbelt -----------------------------------
    def _build_server(self, sdk) -> tuple[Any, list[str]]:
        def make_handler(name: str):
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                self._tool_sequence.append(name)
                try:
                    # Toolbelt.call re-checks the allowlist and writes the ledger.
                    result = self.toolbelt.call(name, **args)
                    return {"content": [{"type": "text", "text": json.dumps(result)}]}
                except Exception as exc:
                    return {
                        "content": [{"type": "text", "text": f"tool error: {exc}"}],
                        "is_error": True,
                    }

            return handler

        schemas: dict[str, dict] = {
            "describe_store": {},
            "crop_region": {"image_handle": str, "box": list},
            "segment": {"image_handle": str, "box": list, "pixel_spacing_mm": list},
            "measure": {"image_handle": str, "box": list, "pixel_spacing_mm": list},
            "compare_prior": {"current_handle": str, "prior_handle": str, "box": list},
            "retrieve_similar": {"crop_handle": str, "k": int},
            "lookup_criteria": {"topic": str},
        }
        tools = [
            sdk.tool(name, f"Oncoscope deterministic tool: {name}", schemas[name])(
                make_handler(name)
            )
            for name in _ALLOWED_TOOLS
        ]
        server = sdk.create_sdk_mcp_server(name="oncoharness", version="0.1.0", tools=tools)
        qualified = [f"mcp__oncoharness__{name}" for name in _ALLOWED_TOOLS]
        return server, qualified

    def _ensure_session(self) -> tuple[Any, Any, asyncio.AbstractEventLoop]:
        """Build SDK server + options + event loop once; reuse per batch (E7).

        The options object — system prompt, tool schemas, model — is the
        stable prefix prompt caching keys on; rebuilding it per case would
        silently defeat the cache.
        """
        if self._sdk is None:
            sdk = _require_sdk()
            self._server, qualified = self._build_server(sdk)
            self._options = sdk.ClaudeAgentOptions(
                tools=[],                            # strip ALL built-ins
                mcp_servers={"oncoharness": self._server},
                allowed_tools=qualified,             # explicit, no wildcards
                system_prompt=SYSTEM_PROMPT,         # constant: cacheable prefix
                model=self.model,
                max_turns=self.max_turns,
                output_format={
                    "type": "json_schema",
                    "schema": Adjudication.model_json_schema(),
                },
            )
            self._sdk = sdk
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._sdk, self._options, self._loop

    def close(self) -> None:
        """Release the persistent event loop (the batch is over)."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None

    # -- adjudication ----------------------------------------------------
    def adjudicate(self, request: dict) -> Adjudication:
        t0 = time.perf_counter()
        self._tool_sequence = []
        spend = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "usd": 0.0,
            "turns": 0,
        }
        fallback_reason = ""
        adjudication: Adjudication | None = None

        try:
            sdk, options, loop = self._ensure_session()
            prompt = (
                "Adjudicate this case. Facts (handles reference the artifact store; "
                "you may inspect them only through your tools):\n"
                + json.dumps(
                    compact_request(request, self.top_k_candidates),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )

            async def _run() -> Adjudication:
                structured: dict | None = None
                final_text = ""
                async for message in sdk.query(prompt=prompt, options=options):
                    if isinstance(message, sdk.ResultMessage):
                        usage = getattr(message, "usage", None) or {}
                        spend["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                        spend["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                        spend["cache_read_input_tokens"] += int(
                            usage.get("cache_read_input_tokens", 0) or 0
                        )
                        spend["usd"] += float(getattr(message, "total_cost_usd", 0.0) or 0.0)
                        spend["turns"] += int(getattr(message, "num_turns", 0) or 0)
                        structured = getattr(message, "structured_output", None)
                        final_text = getattr(message, "result", "") or ""
                if structured is None:
                    structured = json.loads(final_text)
                return Adjudication.model_validate(structured)

            adjudication = loop.run_until_complete(_run())

            if self.max_usd_per_case is not None and spend["usd"] > self.max_usd_per_case:
                fallback_reason = (
                    f"usd budget breached: {spend['usd']:.4f} > {self.max_usd_per_case:.4f}"
                )
            elif (
                self.max_output_tokens is not None
                and spend["output_tokens"] > self.max_output_tokens
            ):
                fallback_reason = (
                    f"output-token budget breached: {spend['output_tokens']} > "
                    f"{self.max_output_tokens}"
                )
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"

        fallback_used = bool(fallback_reason) or adjudication is None
        rule_adjudication = self._rule.adjudicate(request)
        if fallback_used:
            adjudication = (
                rule_adjudication
                if self.fallback is self._rule
                else self.fallback.adjudicate(request)
            )

        wall_ms = (time.perf_counter() - t0) * 1000.0
        if self.meter is not None:
            self.meter.record_llm(
                {
                    "input_tokens": spend["input_tokens"],
                    "output_tokens": spend["output_tokens"],
                    "cache_read_input_tokens": spend["cache_read_input_tokens"],
                },
                usd=spend["usd"],
                turns=spend["turns"],
            )
        if self.trace_writer is not None:
            self.trace_writer.write(
                AdjudicationTrace(
                    case_id=str(request.get("case_id", "")),
                    policy_id=self.policy_id,
                    lane=str(request.get("lane", "default")),
                    features=request_features(request),
                    tool_sequence=list(self._tool_sequence),
                    turns=spend["turns"],
                    input_tokens=spend["input_tokens"],
                    output_tokens=spend["output_tokens"],
                    cache_read_tokens=spend["cache_read_input_tokens"],
                    usd=round(spend["usd"], 6),
                    wall_ms=round(wall_ms, 3),
                    decision=adjudication.decision,
                    rule_decision=rule_adjudication.decision,
                    agreed_with_rule=adjudication.decision == rule_adjudication.decision,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                )
            )
        return adjudication
