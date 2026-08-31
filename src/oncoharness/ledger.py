"""Append-only, hash-chained evidence ledger (technique 2G, axiom A3).

Every tool call, tool output, and claim lands here with provenance. No agent
may raise a finding's confidence without citing a new ledger entry. The hash
chain makes silent edits detectable: entry N commits to the digest of entry
N-1, so any rewrite breaks verification from that point forward.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

GENESIS = "0" * 64


class EvidenceLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev = GENESIS
        if self.path.exists():
            for line in self.path.open():
                self._prev = json.loads(line)["entry_sha"]

    def append(self, kind: str, payload: dict) -> str:
        """Append one entry; returns its id for citation (e.g. by findings)."""
        body = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "kind": kind,             # tool_call | tool_result | claim | decision
            "payload": payload,
            "prev_sha": self._prev,
        }
        canonical = json.dumps(body, sort_keys=True)
        entry_sha = hashlib.sha256(canonical.encode()).hexdigest()
        body["entry_sha"] = entry_sha
        with self.path.open("a") as f:
            f.write(json.dumps(body, sort_keys=True) + "\n")
        self._prev = entry_sha
        return entry_sha

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.open()]

    def verify_chain(self) -> bool:
        prev = GENESIS
        for entry in self.entries():
            claimed = entry.pop("entry_sha")
            if entry.get("prev_sha") != prev:
                return False
            canonical = json.dumps(entry, sort_keys=True)
            if hashlib.sha256(canonical.encode()).hexdigest() != claimed:
                return False
            prev = claimed
        return True
