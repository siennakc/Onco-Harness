"""Handle-passing artifact store (T-4.1, axiom A3).

The LLM never sees raw pixels. Images, crops, masks, and heatmaps live here
under opaque handles ("art:3f2a..."); tools accept and return handles plus
structured facts. Anything the orchestrator receives is text — the store is
the only place pixels exist.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ArtifactInfo:
    handle: str
    kind: str                  # image | crop | mask | heatmap
    shape: tuple[int, ...]
    sha256: str
    meta: dict


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._manifest: dict[str, dict] = (
            json.loads(self._manifest_path.read_text())
            if self._manifest_path.exists()
            else {}
        )

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(json.dumps(self._manifest, indent=1, sort_keys=True))

    def put(self, array: np.ndarray, kind: str, meta: dict | None = None) -> ArtifactInfo:
        handle = f"art:{uuid.uuid4().hex[:12]}"
        digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        np.save(self.root / f"{handle.split(':')[1]}.npy", array)
        record = {
            "kind": kind,
            "shape": list(array.shape),
            "sha256": digest,
            "meta": meta or {},
        }
        self._manifest[handle] = record
        self._save_manifest()
        return ArtifactInfo(
            handle=handle, kind=kind, shape=tuple(array.shape), sha256=digest, meta=record["meta"]
        )

    def get(self, handle: str) -> np.ndarray:
        if handle not in self._manifest:
            raise KeyError(f"unknown artifact handle {handle!r}")
        return np.load(self.root / f"{handle.split(':')[1]}.npy")

    def info(self, handle: str) -> ArtifactInfo:
        rec = self._manifest[handle]
        return ArtifactInfo(
            handle=handle,
            kind=rec["kind"],
            shape=tuple(rec["shape"]),
            sha256=rec["sha256"],
            meta=rec["meta"],
        )

    def describe(self) -> list[dict]:
        """Text-safe listing for the LLM: handles and facts, never pixels."""
        return [
            {"handle": h, "kind": r["kind"], "shape": r["shape"], "meta": r["meta"]}
            for h, r in sorted(self._manifest.items())
        ]
