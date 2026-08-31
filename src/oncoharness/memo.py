"""Content-hash tool memoization (U3: E6; axioms A3, A6, S8).

Every tool in the belt is deterministic and every artifact already carries a
sha256 (``store.py``), so exact memoization is trivial and *sound*: a cache
key is the tool name, its pinned version, and its kwargs with every
``art:*`` handle replaced by that artifact's content hash. Same bytes, same
parameters, same tool — same result, by construction. The harness recomputes
identical work constantly (TTA replays in nightly evals, zoom re-detections,
the gate's determinism double-run, ablation arms, failure-bank replays);
memoization makes the evaluation side of the RSI loop cheap enough to run
nightly on a laptop.

Scope rules (the soundness boundary):

- **Cross-run** memo only for tools whose results contain no handles:
  ``run_detector``, ``measure``, ``lookup_criteria``, ``retrieve_similar``.
  Their outputs are pure facts about content, valid in any store.
- **In-run** memo for handle-producing tools (``crop_region``, ``segment``):
  handles are store-scoped, so these entries live in memory only, keyed by
  the store instance, and never touch the file. A new run gets a miss.
- **Never** memoized: ``submit_review`` (a side effect), ``run_eval_gate``
  (the gate must run), ``describe_store`` (reads mutable state), and
  anything not explicitly listed — deny-by-default, like the registry.

Version-aware invalidation: ``tool_versions`` (fed from PolicyConfig's
``tool_versions`` pin plus :func:`derive_tool_versions`) is folded into the
key, so re-parameterizing a detector profile or swapping the criteria corpus
invalidates cleanly — no stale result can survive a policy change.

Ledger and audit semantics (S8): a hit still writes a ``tool_call`` entry
and a ``tool_result_cached`` entry citing the original result's ledger ref,
so the evidence chain stays complete, the U1 invariant (metered calls ==
ledger ``tool_call`` entries) holds, and an audit replay can distinguish
cached from fresh evidence. Only :meth:`~oncoharness.tools.Toolbelt.call`
reads or writes the memo — it is never a registered tool, and the memo file
lives under ``runs/memo/``, off the LLM's writable surface: a corrupted
entry can at worst reproduce an old *tool* output, never a fabricated one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
import weakref
from collections import OrderedDict
from pathlib import Path

CROSS_RUN_TOOLS = frozenset({"run_detector", "measure", "lookup_criteria", "retrieve_similar"})
IN_RUN_TOOLS = frozenset({"crop_region", "segment"})

DEFAULT_MAX_BYTES = 200 * 1024 * 1024  # spec'd LRU cap for the on-disk memo


def derive_tool_versions(toolbelt) -> dict[str, str]:
    """Version pins from what actually determines each tool's output (E6).

    - ``run_detector``: hash of every detector profile's parameters — bump a
      sigma or threshold (a PolicyConfig change) and every cached detection
      misses cleanly.
    - ``retrieve_similar`` / ``lookup_criteria``: hash of the atlas/corpus
      file bytes, so editing the reference data invalidates too.
    - geometry tools pin a code-owned constant.

    Merge ``PolicyConfig.tool_versions`` over this dict to pin further:
    explicit policy pins outrank derived ones.
    """
    profiles = {
        name: {k: v for k, v in sorted(vars(det).items())}
        for name, det in sorted(getattr(toolbelt, "detector_profiles", {}).items())
    }
    versions = {
        "run_detector": _short_hash(json.dumps(profiles, sort_keys=True, default=repr)),
        "crop_region": "v1",
        "segment": "v1",
        "measure": "v1",
    }
    for tool, path in (
        ("retrieve_similar", getattr(toolbelt, "atlas_path", None)),
        ("lookup_criteria", getattr(toolbelt, "criteria_path", None)),
    ):
        if path is not None and Path(path).exists():
            versions[tool] = _short_hash(Path(path).read_bytes())
        else:
            versions[tool] = "absent"
    return versions


def _short_hash(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]


class ToolMemo:
    """Append-only JSONL memo with an in-memory index, LRU-capped (E6).

    ``get``/``put`` speak opaque keys produced by :meth:`key`; the scope
    prefix inside the key routes cross-run entries to disk and in-run
    entries to a per-store, memory-only dict. Results are deep-copied on
    both put and get, so downstream mutation (the pipeline annotates result
    dicts freely) can never poison the cache.
    """

    def __init__(
        self,
        path: str | Path,
        tool_versions: dict[str, str] | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.path = Path(path)
        self.tool_versions = dict(tool_versions or {})
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict[str, dict] = OrderedDict()  # key -> record (LRU order)
        self._bytes = 0
        self._in_run: dict[str, dict] = {}
        self._store_tokens: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self.hits = 0
        self.misses = 0
        self._load()

    # -- keys -------------------------------------------------------------
    def scope_of(self, tool: str) -> str | None:
        if tool in CROSS_RUN_TOOLS:
            return "cross"
        if tool in IN_RUN_TOOLS:
            return "inrun"
        return None  # deny-by-default: unlisted tools are never memoized

    def key(self, tool: str, kwargs: dict, store) -> str | None:
        """Canonical content key, or None when this call must not be memoized.

        Every ``art:*`` string anywhere in ``kwargs`` is replaced by the
        artifact's sha256 before hashing — the key names *content*, never a
        handle, so byte-identical pixels hit regardless of which run minted
        the handle. In-run keys are additionally namespaced by a per-store
        token, which is how handle-producing results stay store-scoped.
        """
        scope = self.scope_of(tool)
        if scope is None:
            return None
        try:
            resolved = _resolve_handles(kwargs, store)
        except KeyError:
            return None  # unknown handle: let the tool itself raise, unmemoized
        blob = json.dumps(
            {"tool": tool, "version": self.tool_versions.get(tool, ""), "kwargs": resolved},
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
        digest = hashlib.sha256(blob.encode()).hexdigest()
        if scope == "inrun":
            token = self._store_tokens.get(store)
            if token is None:
                token = uuid.uuid4().hex
                self._store_tokens[store] = token
            return f"inrun:{token}:{digest}"
        return f"cross:{digest}"

    # -- lookups ----------------------------------------------------------
    def get(self, key: str) -> dict | None:
        """The memoized record ``{"result", "original_ref"}``, or None on miss."""
        if key.startswith("inrun:"):
            record = self._in_run.get(key)
        else:
            record = self._entries.get(key)
            if record is not None:
                self._entries.move_to_end(key)  # LRU touch
        if record is None:
            self.misses += 1
            return None
        self.hits += 1
        return copy.deepcopy(record)

    def put(self, key: str, result: dict) -> None:
        """Store one tool result under ``key`` (minus its evidence ref).

        The ledger ref of the original ``tool_result`` entry is kept as
        ``original_ref`` so a later hit can cite the evidence it replays.
        """
        record = {
            "result": {k: copy.deepcopy(v) for k, v in result.items() if k != "evidence_ref"},
            "original_ref": str(result.get("evidence_ref", "")),
        }
        if key.startswith("inrun:"):
            self._in_run[key] = record
            return
        # Normalize through JSON once (numpy scalars -> plain numbers), so the
        # in-memory record and the file replay byte-for-byte the same result.
        record = json.loads(json.dumps(record, sort_keys=True, default=_json_default))
        line = json.dumps({"key": key, "ts": time.time(), **record}, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(line + "\n")
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = record
        self._bytes += len(line) + 1
        if self._bytes > self.max_bytes:
            self._evict_and_compact()

    def clear_in_run(self) -> None:
        self._in_run.clear()

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["key"]
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = {"result": row["result"], "original_ref": row["original_ref"]}
            self._bytes += len(line) + 1

    def _evict_and_compact(self) -> None:
        """Drop least-recently-used entries until under cap; rewrite the file.

        Compaction also squeezes out superseded duplicate lines, so the file
        converges to one line per live key.
        """
        lines: list[str] = []
        size = 0
        kept: list[tuple[str, dict, str]] = []
        for key, record in reversed(self._entries.items()):  # most recent first
            line = json.dumps({"key": key, "ts": time.time(), **record}, sort_keys=True)
            if size + len(line) + 1 > self.max_bytes and kept:
                break
            kept.append((key, record, line))
            size += len(line) + 1
        kept.reverse()
        self._entries = OrderedDict((k, r) for k, r, _ in kept)
        lines = [line for _, _, line in kept]
        self._bytes = size
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
        tmp.replace(self.path)


def _json_default(obj):
    """Unbox numpy scalars; refuse anything else loudly (a silent key fork is a bug)."""
    if hasattr(obj, "item") and not hasattr(obj, "__len__"):
        return obj.item()
    raise TypeError(f"unmemoizable value of type {type(obj).__name__}")


def _resolve_handles(obj, store):
    """Recursively swap ``art:*`` handles for their content sha256."""
    if isinstance(obj, str) and obj.startswith("art:"):
        return {"__artifact_sha256__": store.info(obj).sha256}
    if isinstance(obj, dict):
        return {str(k): _resolve_handles(v, store) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_resolve_handles(v, store) for v in obj]
    if isinstance(obj, float):
        return round(obj, 12)  # canonical float form; repr noise must not fork keys
    return obj
