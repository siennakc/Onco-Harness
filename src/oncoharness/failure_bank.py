"""Auto-growing failure bank + regression replay gate (U5; A7, A8, T-3.2's unbuilt half).

"Every confirmed error becomes a locked test" is the ratchet's teeth. This
module is the runtime half T-3.2's static pytest files never covered: a
typed, append-only, hash-chained bank of failure records that the loop can
grow but never quietly edit, a replay runner that re-executes every active
record against a candidate policy, and a gate check that fails promotion
the moment a confirmed record the champion passed starts failing again.

Hygiene follows the Ratchet recipe: near-duplicate failures are
canonicalized by cosine similarity (>= ``merge_cosine``, default 0.85)
into one record with a growing ``cluster_size`` — a merge appends a
``merge`` event, it never rewrites — and a hard ``active_cap`` bounds the
store so curation stays a decision, not an accident.

Truth-channel discipline (A7): only human adjudication and clinical
outcomes are truth channels. A record mined by the loop itself
(``eval_miss``) cannot even be *constructed* as confirmed — the schema
refuses — and reaches ``confirmed`` only through :meth:`FailureBank.confirm`,
the human-gated call that demands an evidence reference. Until then it is
probationary and excluded from gating (the tasksheet's "mined FPs may be
unreported cancers" pitfall, enforced at the schema level).

Records are typed and imperative-free (S6): every field is an enum, a
number, an id, or a constrained tag — nothing the model wrote can ever be
rendered back into a prompt as an instruction. The bank feeds replays and
statistics, not context.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from .gate import GateCheck
from .ledger import EvidenceLedger
from .schemas import CaseDecision

RECORD_ID_LEN = 12
# Typed, imperative-free tags: ids and enum-ish tokens only, no prose.
_TAG_RE = re.compile(r"^[A-Za-z0-9_.:+/=-]{1,64}$")


class FailureStatus(str, Enum):
    probation = "probation"
    confirmed = "confirmed"
    retired = "retired"


class FailureSource(str, Enum):
    human_adjudication = "human_adjudication"  # A7 truth channel
    outcome = "outcome"                        # A7 truth channel (pathology, registry)
    eval_miss = "eval_miss"                    # loop-mined: probationary until a human confirms


class FailureRecord(BaseModel):
    """One confirmed-or-suspected error, as data the loop can replay but not read aloud."""

    record_id: str = Field(default="", description="content hash; assigned by the bank")
    case_ref: str = Field(description="dataset case id or artifact sha256")
    label: int = Field(ge=0, le=1, description="ground truth (1 = cancer)")
    expected: CaseDecision = Field(description="what a correct policy must output")
    observed: CaseDecision = Field(description="what the failing policy output")
    policy_id: str = Field(default="", description="which policy failed (U4)")
    slice_tags: dict[str, str] = Field(
        default_factory=dict, description="site, density_band, size_band, ... (constrained tokens)"
    )
    embedding: list[float] | None = Field(
        default=None, description="embed_crop features for cosine canonicalization"
    )
    status: FailureStatus = FailureStatus.probation
    source: FailureSource
    cluster_size: int = 1
    added_ts: str = ""

    @field_validator("slice_tags")
    @classmethod
    def _tags_are_tokens_not_prose(cls, v: dict[str, str]) -> dict[str, str]:
        for key, value in v.items():
            if not _TAG_RE.match(key) or not _TAG_RE.match(value):
                raise ValueError(
                    f"slice tag {key!r}={value!r} is not a constrained token; "
                    "free text is banned from bank records (S6)"
                )
        return v

    @model_validator(mode="after")
    def _schema_teeth(self) -> "FailureRecord":
        if self.observed == self.expected:
            raise ValueError("not a failure: observed decision equals expected decision")
        if self.source == FailureSource.eval_miss and self.status == FailureStatus.confirmed:
            raise ValueError(
                "eval_miss is not an A7 truth channel: loop-mined records enter on "
                "probation and reach confirmed only via FailureBank.confirm"
            )
        return self

    def failure_type(self) -> tuple[int, str, str]:
        """Canonicalization bucket: merges only happen within one failure mode."""
        return (self.label, self.expected.value, self.observed.value)

    def content_id(self) -> str:
        basis = json.dumps(
            [self.case_ref, self.label, self.expected.value, self.observed.value],
            sort_keys=True,
        )
        return hashlib.sha256(basis.encode()).hexdigest()[:RECORD_ID_LEN]


@dataclass
class BankReport:
    """Per-record pass vector for one policy's replay of the bank."""

    policy_id: str
    results: dict[str, bool] = field(default_factory=dict)    # record_id -> passed
    statuses: dict[str, str] = field(default_factory=dict)    # record_id -> status at replay

    @property
    def confirmed_results(self) -> dict[str, bool]:
        return {
            rid: ok
            for rid, ok in self.results.items()
            if self.statuses.get(rid) == FailureStatus.confirmed.value
        }

    @property
    def pass_rate(self) -> float:
        confirmed = self.confirmed_results
        if not confirmed:
            return 1.0
        return float(sum(confirmed.values()) / len(confirmed))


class FailureBank:
    """Append-only failure store, event-sourced from a hash-chained ledger.

    Layout under ``root`` (default ``runs/failure_bank``):

    - ``bank.jsonl``            add/merge/confirm/retire events (EvidenceLedger chain)
    - ``replays/<policy>.json`` pass vectors, one per replayed policy (feeds S11)

    State is rebuilt from the event stream on every open, so the file is
    the record and any in-memory shortcut is disposable.
    """

    def __init__(
        self,
        root: str | Path = "runs/failure_bank",
        merge_cosine: float = 0.85,
        active_cap: int = 512,
    ) -> None:
        self.root = Path(root)
        self.replays_dir = self.root / "replays"
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = EvidenceLedger(self.root / "bank.jsonl")
        self.merge_cosine = float(merge_cosine)
        self.active_cap = int(active_cap)
        self._records: dict[str, FailureRecord] = {}
        self._replay_events()

    # -- event sourcing ---------------------------------------------------
    def _replay_events(self) -> None:
        for entry in self.ledger.entries():
            kind, payload = entry["kind"], entry["payload"]
            if kind == "add":
                rec = FailureRecord(**payload["record"])
                self._records[rec.record_id] = rec
            elif kind == "merge":
                rec = self._records[payload["record_id"]]
                self._records[rec.record_id] = rec.model_copy(
                    update={"cluster_size": rec.cluster_size + 1}
                )
            elif kind == "confirm":
                rec = self._records[payload["record_id"]]
                self._records[rec.record_id] = rec.model_copy(
                    update={"status": FailureStatus.confirmed}
                )
            elif kind == "retire":
                rec = self._records[payload["record_id"]]
                self._records[rec.record_id] = rec.model_copy(
                    update={"status": FailureStatus.retired}
                )

    # -- growth -----------------------------------------------------------
    def add(self, rec: FailureRecord) -> str:
        """Canonicalized append: returns the id now holding this failure.

        An exact re-report of a known case, or a near-duplicate at cosine
        >= ``merge_cosine`` within the same failure type, merges into the
        existing record (``cluster_size`` += 1 via an appended ``merge``
        event — history is never rewritten). Anything else appends a new
        record, subject to the hard active cap.
        """
        rid = rec.content_id()
        if rid in self._records:
            self._append_merge(rid, rec)
            return rid
        target = self._nearest_duplicate(rec)
        if target is not None:
            self._append_merge(target, rec)
            return target
        active = [r for r in self._records.values() if r.status != FailureStatus.retired]
        if len(active) >= self.active_cap:
            raise ValueError(
                f"failure bank active cap ({self.active_cap}) reached: curate/retire "
                "before adding — an unbounded self-written store drifts (Ratchet)"
            )
        stamped = rec.model_copy(
            update={
                "record_id": rid,
                "added_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
        )
        self.ledger.append("add", {"record": stamped.model_dump(mode="json")})
        self._records[rid] = stamped
        return rid

    def _append_merge(self, target_id: str, rec: FailureRecord) -> None:
        self.ledger.append(
            "merge",
            {"record_id": target_id, "case_ref": rec.case_ref, "policy_id": rec.policy_id},
        )
        existing = self._records[target_id]
        self._records[target_id] = existing.model_copy(
            update={"cluster_size": existing.cluster_size + 1}
        )

    def _nearest_duplicate(self, rec: FailureRecord) -> str | None:
        if rec.embedding is None:
            return None
        q = np.asarray(rec.embedding, dtype=np.float64)
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return None
        best_id, best_cos = None, self.merge_cosine
        for rid, other in sorted(self._records.items()):
            if other.embedding is None or other.failure_type() != rec.failure_type():
                continue
            if other.status == FailureStatus.retired:
                continue
            v = np.asarray(other.embedding, dtype=np.float64)
            vn = float(np.linalg.norm(v))
            if vn == 0.0 or len(v) != len(q):
                continue
            cos = float(q @ v / (qn * vn))
            if cos >= best_cos:
                best_id, best_cos = rid, cos
        return best_id

    # -- curation (human-gated) -------------------------------------------
    def confirm(self, record_id: str, evidence_ref: str) -> None:
        """Probation -> confirmed, only with a citation to the human evidence."""
        if not evidence_ref.strip():
            raise ValueError("confirmation requires an evidence reference (A7)")
        rec = self._require(record_id)
        if rec.status != FailureStatus.probation:
            raise ValueError(f"record {record_id} is {rec.status.value}, not probation")
        self.ledger.append("confirm", {"record_id": record_id, "evidence_ref": evidence_ref})
        self._records[record_id] = rec.model_copy(update={"status": FailureStatus.confirmed})

    def retire(self, record_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("retirement requires a reason for the ledger")
        rec = self._require(record_id)
        self.ledger.append("retire", {"record_id": record_id, "reason": reason})
        self._records[record_id] = rec.model_copy(update={"status": FailureStatus.retired})

    def _require(self, record_id: str) -> FailureRecord:
        if record_id not in self._records:
            raise KeyError(f"unknown failure record {record_id!r}")
        return self._records[record_id]

    # -- reads ------------------------------------------------------------
    def records(self, status: str = "confirmed") -> list[FailureRecord]:
        """Records filtered by status; ``status='all'`` returns everything."""
        out = [
            r
            for r in self._records.values()
            if status == "all" or r.status.value == status
        ]
        return sorted(out, key=lambda r: r.record_id)

    def verify_chain(self) -> bool:
        return self.ledger.verify_chain()

    # -- replay -----------------------------------------------------------
    def replay(self, run_case_fn, load_case_fn, policy_id: str = "") -> BankReport:
        """Re-run every active record through a policy; persist its pass vector.

        ``load_case_fn(case_ref)`` returns the case payload (e.g. pixels);
        ``run_case_fn(case_ref, payload)`` returns a report with a
        ``decision`` (or the decision itself). A record passes iff the
        decision equals the record's ``expected``. When ``policy_id`` is
        empty, it is taken from the first stamped report (U4/S2).
        """
        report = BankReport(policy_id=policy_id)
        for rec in self.records("all"):
            if rec.status == FailureStatus.retired:
                continue
            case_report = run_case_fn(rec.case_ref, load_case_fn(rec.case_ref))
            decision = getattr(case_report, "decision", case_report)
            if not report.policy_id:
                report.policy_id = getattr(case_report, "policy_id", "") or "unregistered"
            report.results[rec.record_id] = CaseDecision(decision) == rec.expected
            report.statuses[rec.record_id] = rec.status.value
        if not report.policy_id:
            report.policy_id = "unregistered"
        out = self.replays_dir / f"{report.policy_id}.json"
        out.write_text(
            json.dumps(
                {
                    "policy_id": report.policy_id,
                    "results": report.results,
                    "statuses": report.statuses,
                    "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                },
                sort_keys=True,
                indent=1,
            )
        )
        return report

    def transfer_matrix(self) -> dict[str, dict[str, bool]]:
        """policy_id -> record_id -> passed, from every persisted replay (S11).

        Negative transfer — a record fixed under one policy and regressed
        under a later one — shows up as a False after a True in a column.
        """
        matrix: dict[str, dict[str, bool]] = {}
        for path in sorted(self.replays_dir.glob("*.json")):
            data = json.loads(path.read_text())
            matrix[data["policy_id"]] = {k: bool(v) for k, v in data["results"].items()}
        return matrix


def gate_check_failure_bank(
    bank_report: BankReport,
    champion_vector: dict[str, bool],
    max_confirmed_regressions: int = 0,
) -> GateCheck:
    """FAIL when a confirmed record the champion passed now fails (bank-level negative flip).

    New confirmed records the champion never replayed are teeth for the
    future, not regressions today; probation records never gate (their
    labels are unconfirmed and mined FPs may be unreported cancers).
    """
    confirmed = bank_report.confirmed_results
    regressions = sorted(
        rid for rid, ok in confirmed.items() if not ok and champion_vector.get(rid, False)
    )
    failed_total = sum(1 for ok in confirmed.values() if not ok)
    detail = (
        f"{len(regressions)} confirmed-record regression(s) vs champion "
        f"(allowed {max_confirmed_regressions}); "
        f"{failed_total}/{len(confirmed)} confirmed records failing overall"
    )
    if regressions:
        detail += f"; regressed: {', '.join(regressions[:5])}"
    return GateCheck(
        "failure_bank_regression",
        passed=len(regressions) <= max_confirmed_regressions,
        detail=detail,
    )
