"""Patient- and site-grouped splits, persisted and hashed (T-1.2, axiom A9).

Image-level splitting inflates AUROC by 2-20+ points and is the most common
invalidating bug in the field. Grouping is by patient always, and by site when
more than one site exists. Splits are written to a versioned JSON file whose
hash is recorded, so every eval is reproducible against an exact membership.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

SPLIT_NAMES = ("train", "calibration", "threshold", "slice_discovery", "test")


@dataclass
class SplitManifest:
    version: str
    assignment: dict[str, str]  # patient_id -> split name
    sha256: str

    def split_of(self, patient_id: str) -> str:
        return self.assignment[patient_id]


def make_splits(
    patients: dict[str, str],
    fractions: dict[str, float],
    seed: int = 20260829,
    version: str = "v1",
) -> SplitManifest:
    """Assign each patient to exactly one split, stratified by site.

    ``patients`` maps patient_id -> site_id. Every image of a patient inherits
    the patient's split; no patient appears in two splits, ever.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")
    unknown = set(fractions) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"unknown split names: {unknown}")

    rng = random.Random(seed)
    by_site: dict[str, list[str]] = {}
    for pid, site in patients.items():
        by_site.setdefault(site, []).append(pid)

    assignment: dict[str, str] = {}
    for site, pids in sorted(by_site.items()):
        pids = sorted(pids)
        rng.shuffle(pids)
        # Largest-remainder allocation so small sites still fill every split.
        n = len(pids)
        exact = {name: n * frac for name, frac in fractions.items()}
        counts = {name: int(v) for name, v in exact.items()}
        remainder = n - sum(counts.values())
        for name, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
            if remainder <= 0:
                break
            counts[name] += 1
            remainder -= 1
        i = 0
        for name in SPLIT_NAMES:
            for _ in range(counts.get(name, 0)):
                assignment[pids[i]] = name
                i += 1

    payload = json.dumps({"version": version, "assignment": assignment}, sort_keys=True)
    sha = hashlib.sha256(payload.encode()).hexdigest()
    return SplitManifest(version=version, assignment=assignment, sha256=sha)


def save_manifest(manifest: SplitManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": manifest.version,
                "sha256": manifest.sha256,
                "assignment": manifest.assignment,
            },
            sort_keys=True,
            indent=1,
        )
    )


def load_manifest(path: str | Path) -> SplitManifest:
    raw = json.loads(Path(path).read_text())
    manifest = SplitManifest(
        version=raw["version"], assignment=raw["assignment"], sha256=raw["sha256"]
    )
    payload = json.dumps(
        {"version": manifest.version, "assignment": manifest.assignment}, sort_keys=True
    )
    if hashlib.sha256(payload.encode()).hexdigest() != manifest.sha256:
        raise ValueError(f"split manifest {path} failed hash verification — do not proceed")
    return manifest
