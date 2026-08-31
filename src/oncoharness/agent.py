"""LLM adjudication via the Claude Agent SDK (T-4.1).

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

Install with ``pip install -e '.[agent]'`` and authenticate the Anthropic SDK
environment before use. Without the extra, the deterministic
``RuleBasedAdjudicator`` runs instead — the pipeline never requires an LLM.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .schemas import Adjudication
from .tools import Toolbelt

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


def _require_sdk():
    try:
        import claude_agent_sdk  # noqa: F401

        return claude_agent_sdk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "claude-agent-sdk is not installed; install with pip install -e '.[agent]' "
            "or use the default RuleBasedAdjudicator"
        ) from exc


class LLMAdjudicator:
    """Drop-in :class:`Adjudicator` backed by a Claude agent."""

    def __init__(
        self,
        toolbelt: Toolbelt,
        model: str = "claude-opus-5",
        max_turns: int = 24,
    ) -> None:
        self.toolbelt = toolbelt
        self.model = model
        self.max_turns = max_turns

    # -- MCP server over the toolbelt -----------------------------------
    def _build_server(self, sdk) -> tuple[Any, list[str]]:
        def make_handler(name: str):
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
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

    # -- adjudication ----------------------------------------------------
    def adjudicate(self, request: dict) -> Adjudication:
        sdk = _require_sdk()
        server, qualified_tools = self._build_server(sdk)

        options = sdk.ClaudeAgentOptions(
            tools=[],                            # strip ALL built-ins
            mcp_servers={"oncoharness": server},
            allowed_tools=qualified_tools,       # explicit, no wildcards
            system_prompt=SYSTEM_PROMPT,
            model=self.model,
            max_turns=self.max_turns,
            output_format={
                "type": "json_schema",
                "schema": Adjudication.model_json_schema(),
            },
        )

        prompt = (
            "Adjudicate this case. Facts (handles reference the artifact store; "
            "you may inspect them only through your tools):\n"
            + json.dumps(request, indent=1)
        )

        async def _run() -> Adjudication:
            structured: dict | None = None
            final_text = ""
            async for message in sdk.query(prompt=prompt, options=options):
                if isinstance(message, sdk.ResultMessage):
                    structured = getattr(message, "structured_output", None)
                    final_text = getattr(message, "result", "") or ""
            if structured is None:
                structured = json.loads(final_text)
            return Adjudication.model_validate(structured)

        return asyncio.run(_run())
