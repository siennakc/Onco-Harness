"""Per-case efficiency telemetry: the CostMeter (U1: E1/E2; axioms A6, A8).

You cannot optimize — or gate — what you do not measure. Until now the repo
could not state what one case costs: the ledger recorded *that* tools ran,
never what they spent. The meter counts it: tool calls by name, pixels pushed
through the detector, wall time, memo hits, LLM turns/tokens/dollars. The
finished :class:`CaseCost` lands on every ``CaseReport`` (and therefore in the
ledger's ``claim`` entry), so "improvement" can never again mean "spent 10x
more" invisibly — cost is a first-class metric the ratchet gates (A8), not an
afterthought.

Write-protection discipline (S8, the DGM lesson): any signal the optimizer can
write to is not a safeguard. The meter is code-side instrumentation only — it
is never registered as a tool, so neither the LLM nor the improvement loop
holds write access to its own measurement; the per-run ``costs.jsonl``
aggregate lives outside the artifact store the agent can touch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CaseCost:
    """What one case actually cost. Zeros mean "none observed", never "unknown"."""

    tool_calls: dict[str, int] = field(default_factory=dict)  # tool name -> count
    cached_hits: int = 0        # memo hits (stays 0 until tool memoization lands)
    detector_pixels: int = 0    # sum(h*w) over detector inputs
    wall_ms: float = 0.0        # whole-case wall time, start_case -> finish_case
    tool_wall_ms: float = 0.0   # time spent inside tool bodies alone
    llm_turns: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cache_read_tokens: int = 0
    llm_usd: float = 0.0

    def total_tool_calls(self) -> int:
        return int(sum(self.tool_calls.values()))

    def as_dict(self) -> dict:
        """JSON-safe cost block (embedded in CaseReport / costs.jsonl rows)."""
        return {
            "tool_calls": dict(sorted(self.tool_calls.items())),
            "total_tool_calls": self.total_tool_calls(),
            "cached_hits": self.cached_hits,
            "detector_pixels": self.detector_pixels,
            "wall_ms": round(self.wall_ms, 3),
            "tool_wall_ms": round(self.tool_wall_ms, 3),
            "llm_turns": self.llm_turns,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_cache_read_tokens": self.llm_cache_read_tokens,
            "llm_usd": round(self.llm_usd, 6),
        }


class CostMeter:
    """Accumulates one :class:`CaseCost` between ``start_case``/``finish_case``.

    A record arriving outside an explicit bracket opens an implicit unnamed
    case, so bare ``Toolbelt`` usage (tests, notebooks) never crashes. When
    ``jsonl_path`` is set, ``finish_case`` also appends one JSON line per case
    — the per-run aggregate (``runs/<run>/costs.jsonl``).
    """

    def __init__(self, jsonl_path: str | Path | None = None) -> None:
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._case_id: str = ""
        self._cost: CaseCost | None = None
        self._t0: float = 0.0

    # -- case lifecycle --------------------------------------------------
    def start_case(self, case_id: str) -> None:
        self._case_id = case_id
        self._cost = CaseCost()
        self._t0 = time.perf_counter()

    def _current(self) -> CaseCost:
        if self._cost is None:
            self.start_case("")
        assert self._cost is not None
        return self._cost

    def finish_case(self) -> CaseCost:
        """Close the case: stamp wall time, optionally persist, reset the meter."""
        cost = self._current()
        cost.wall_ms = (time.perf_counter() - self._t0) * 1000.0
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"case_id": self._case_id, **cost.as_dict()}
            with self.jsonl_path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        self._case_id, self._cost = "", None
        return cost

    # -- recording -------------------------------------------------------
    def record_tool(
        self, name: str, wall_ms: float, pixels: int = 0, cached: bool = False
    ) -> None:
        cost = self._current()
        cost.tool_calls[name] = cost.tool_calls.get(name, 0) + 1
        cost.tool_wall_ms += float(wall_ms)
        cost.detector_pixels += int(pixels)
        if cached:
            cost.cached_hits += 1

    def record_llm(self, usage: dict | None, usd: float, turns: int) -> None:
        """Fold one LLM adjudication's spend in (SDK ``ResultMessage`` fields)."""
        cost = self._current()
        usage = usage or {}
        cost.llm_turns += int(turns)
        cost.llm_input_tokens += int(usage.get("input_tokens", 0) or 0)
        cost.llm_output_tokens += int(usage.get("output_tokens", 0) or 0)
        cost.llm_cache_read_tokens += int(usage.get("cache_read_input_tokens", 0) or 0)
        cost.llm_usd += float(usd or 0.0)
