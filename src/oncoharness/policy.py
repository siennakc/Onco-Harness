"""Versioned policy registry: content-hashed configs, lineage, rollback (U4; A8, A14, T-5.5/T-5.6 substrate).

A change you cannot name, diff, and roll back is not an experiment. Every
mutable lever the self-improvement loop may pull — pipeline thresholds,
router settings, adjudicator parameters, registered prompt texts, pinned
tool versions — lives inside one frozen :class:`PolicyConfig`, and the
policy's name IS its content hash: identical configs collapse to one id,
any edit mints a new one. This also bounds the reachable policy family
(S9): the loop tunes parameters and registered prompt text inside a
declared schema, never arbitrary code — ``pipeline_from_policy`` refuses
unknown knobs outright.

Promotion is a registry pointer move and rollback is one command (A8):
``promote`` refuses without a passing gate summary (self-improving, never
self-certifying), ``rollback`` returns the champion pointer to a policy
that has actually held the title before. Both are recorded — along with
every ``register`` — in a hash-chained lineage ledger (reusing
:class:`~oncoharness.ledger.EvidenceLedger`), so the full history of who
was champion, why, and on what evidence survives any crash and betrays
any rewrite.

The promote/rollback paths should ultimately live behind the same
service-account boundary as ``gates/`` (T-3.3 discipline, S8). For v1 a
CI test asserts no other harness code path writes the lineage ledger:
``tests/test_policy.py::test_only_policy_module_writes_lineage``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .ledger import EvidenceLedger

POLICY_ID_LEN = 12


class PolicyConfig(BaseModel):
    """The entire reachable self-modification family, as one frozen record.

    Free-form code is deliberately unrepresentable here: ``prompt_ids``
    holds content hashes of texts registered under ``runs/registry/prompts``
    (never the text itself), and every other field is a parameter dict that
    ``pipeline_from_policy`` validates against an explicit allowlist.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    pipeline: dict[str, Any] = Field(
        default_factory=dict,
        description="HarnessPipeline knobs: consistency_reads, min_reproduced, zoom_*, ...",
    )
    router: dict[str, Any] = Field(
        default_factory=dict, description="difficulty-router settings (U2; empty until built)"
    )
    adjudicator: dict[str, Any] = Field(
        default_factory=dict, description="kind: rule|llm plus that adjudicator's parameters"
    )
    prompt_ids: dict[str, str] = Field(
        default_factory=dict, description="role -> sha256 of a registered prompt text"
    )
    tool_versions: dict[str, str] = Field(
        default_factory=dict, description="tool name -> version pin (feeds memoization, U3)"
    )

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def policy_id(self) -> str:
        """sha256 of the canonical JSON, truncated: the policy's one true name."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:POLICY_ID_LEN]


def default_policy(prompt_ids: dict[str, str] | None = None) -> PolicyConfig:
    """The current hardcoded defaults, captured as generation zero."""
    return PolicyConfig(
        pipeline={
            "consistency_reads": 5,
            "min_reproduced": 3,
            "zoom_margin": 0.75,
            "zoom_score_ratio": 0.4,
            "max_zoom_crops": 8,
            "fp_max_aspect_ratio": 3.0,
            "blindspot_min_score": 0.55,
        },
        router={},
        adjudicator={
            "kind": "rule",
            "recall_threshold": 0.65,
            "deferral_band": [0.35, 0.65],
            "max_disagreement": 0.4,
        },
        prompt_ids=prompt_ids or {},
        tool_versions={},
    )


class PolicyRegistry:
    """Immutable policy store + hash-chained lineage + the champion pointer.

    Layout under ``root`` (default ``runs/registry``):

    - ``policies/<id>.json``   immutable, content-addressed configs
    - ``prompts/<sha256>.txt`` immutable, content-addressed prompt texts
    - ``lineage.jsonl``        hash-chained register/promote/rollback events
    - ``CHAMPION``             the pointer; promotion moves it, nothing else
    """

    def __init__(self, root: str | Path = "runs/registry") -> None:
        self.root = Path(root)
        self.policies_dir = self.root / "policies"
        self.prompts_dir = self.root / "prompts"
        self.champion_path = self.root / "CHAMPION"
        self.policies_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_ledger = EvidenceLedger(self.root / "lineage.jsonl")

    # -- immutable stores -------------------------------------------------
    def register(self, cfg: PolicyConfig, parent_id: str | None = None, note: str = "") -> str:
        """Persist a config under its content hash; append a lineage event.

        Content addressing makes registration idempotent: re-registering an
        identical config is a no-op returning the same id (no duplicate
        lineage entry). A different config can never collide with an
        existing file short of a sha256 collision, which we treat as
        corruption and refuse.
        """
        if parent_id is not None:
            self.get(parent_id)  # unknown parents are lineage holes; refuse early
        pid = cfg.policy_id()
        path = self.policies_dir / f"{pid}.json"
        payload = cfg.canonical_json()
        if path.exists():
            if path.read_text() != payload:
                raise RuntimeError(f"registry corruption: {path} does not match its id")
            return pid
        path.write_text(payload)
        self.lineage_ledger.append(
            "register", {"event": "register", "policy_id": pid, "parent_id": parent_id, "note": note}
        )
        return pid

    def register_prompt(self, text: str) -> str:
        """Content-address a prompt text; returns the sha256 for ``prompt_ids``."""
        sha = hashlib.sha256(text.encode()).hexdigest()
        path = self.prompts_dir / f"{sha}.txt"
        if not path.exists():
            path.write_text(text)
        return sha

    def get_prompt(self, sha: str) -> str:
        path = self.prompts_dir / f"{sha}.txt"
        text = path.read_text()
        if hashlib.sha256(text.encode()).hexdigest() != sha:
            raise RuntimeError(f"prompt {sha[:12]} was edited on disk; content hash mismatch")
        return text

    def get(self, policy_id: str) -> PolicyConfig:
        """Load a config and re-verify its content hash (tamper detection)."""
        path = self.policies_dir / f"{policy_id}.json"
        if not path.exists():
            raise KeyError(f"unknown policy_id {policy_id!r}")
        cfg = PolicyConfig(**json.loads(path.read_text()))
        if cfg.policy_id() != policy_id:
            raise RuntimeError(
                f"policy {policy_id} was edited on disk; content now hashes to {cfg.policy_id()}"
            )
        return cfg

    # -- the pointer ------------------------------------------------------
    def champion(self) -> str | None:
        if not self.champion_path.exists():
            return None
        return self.champion_path.read_text().strip() or None

    def _move_pointer(self, policy_id: str) -> None:
        tmp = self.champion_path.with_suffix(".tmp")
        tmp.write_text(policy_id + "\n")
        os.replace(tmp, self.champion_path)

    def promote(self, policy_id: str, gate_summary: str, gate_passed: bool) -> None:
        """Move the champion pointer — only over a passing gate.

        The registry re-checks the summary text itself: a caller claiming
        ``gate_passed=True`` while handing over a summary that says FAIL is
        refused (the loop is never the judge of its own promotion).
        """
        self.get(policy_id)
        if not gate_passed:
            raise PermissionError(f"refusing to promote {policy_id}: gate did not pass")
        if not gate_summary.strip():
            raise PermissionError(f"refusing to promote {policy_id}: no gate summary provided")
        if "GATE: FAIL" in gate_summary:
            raise PermissionError(
                f"refusing to promote {policy_id}: gate summary reports FAIL "
                "despite gate_passed=True"
            )
        prev = self.champion()
        self.lineage_ledger.append(
            "promote",
            {
                "event": "promote",
                "policy_id": policy_id,
                "previous_champion": prev,
                "gate_passed": True,
                "gate_summary": gate_summary,
            },
        )
        self._move_pointer(policy_id)

    def rollback(self, to_policy_id: str, reason: str) -> None:
        """One command back to a previous champion (A8: rehearsed rollback).

        The target must have actually held the champion title (a prior
        promote or rollback event); rolling "back" to a policy that was
        never champion would be a promotion dodging the gate.
        """
        self.get(to_policy_id)
        if not reason.strip():
            raise ValueError("rollback requires a reason for the lineage record")
        held_title = any(
            e["kind"] in ("promote", "rollback") and e["payload"].get("policy_id") == to_policy_id
            for e in self.lineage_ledger.entries()
        )
        if not held_title:
            raise PermissionError(
                f"refusing rollback to {to_policy_id}: it was never champion "
                "(promotion must go through the gate)"
            )
        prev = self.champion()
        self.lineage_ledger.append(
            "rollback",
            {
                "event": "rollback",
                "policy_id": to_policy_id,
                "previous_champion": prev,
                "reason": reason,
            },
        )
        self._move_pointer(to_policy_id)

    # -- history ----------------------------------------------------------
    def lineage(self) -> list[dict]:
        return self.lineage_ledger.entries()

    def verify_chain(self) -> bool:
        return self.lineage_ledger.verify_chain()


def pipeline_from_policy(cfg: PolicyConfig, toolbelt, head=None, conformal=None):
    """Instantiate a :class:`~oncoharness.state_machine.HarnessPipeline` from a policy.

    This is the S9 boundary made executable: only knobs the pipeline and
    adjudicator declare are accepted, so a "policy" can never smuggle in
    behavior outside the registered family. Reports produced by the
    returned pipeline are stamped with ``cfg.policy_id()`` (S2).
    """
    import inspect

    from .state_machine import HarnessPipeline, RuleBasedAdjudicator

    allowed_pipeline = set(inspect.signature(HarnessPipeline.__init__).parameters) - {
        "self", "toolbelt", "adjudicator", "head", "conformal", "policy_id",
    }
    unknown = set(cfg.pipeline) - allowed_pipeline
    if unknown:
        raise ValueError(
            f"policy {cfg.policy_id()} names unknown pipeline knob(s) {sorted(unknown)}; "
            f"the reachable family is {sorted(allowed_pipeline)}"
        )

    adj_cfg = dict(cfg.adjudicator)
    kind = adj_cfg.pop("kind", "rule")
    if kind != "rule":
        raise ValueError(
            f"policy {cfg.policy_id()} requests adjudicator kind {kind!r}; "
            "only 'rule' is instantiable here (the LLM adjudicator is wired "
            "by the runtime with its registered prompt, not by this factory)"
        )
    allowed_adj = set(inspect.signature(RuleBasedAdjudicator.__init__).parameters) - {"self"}
    unknown_adj = set(adj_cfg) - allowed_adj
    if unknown_adj:
        raise ValueError(
            f"policy {cfg.policy_id()} names unknown adjudicator knob(s) {sorted(unknown_adj)}; "
            f"the reachable family is {sorted(allowed_adj)}"
        )
    if "deferral_band" in adj_cfg:
        adj_cfg["deferral_band"] = tuple(adj_cfg["deferral_band"])
    adjudicator = RuleBasedAdjudicator(**adj_cfg)

    return HarnessPipeline(
        toolbelt,
        adjudicator=adjudicator,
        head=head,
        conformal=conformal,
        policy_id=cfg.policy_id(),
        **cfg.pipeline,
    )
