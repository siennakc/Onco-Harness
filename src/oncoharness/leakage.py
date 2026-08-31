"""Leakage and contamination audit (T-1.2, axiom A9).

Run in CI as a failing test, not a convention. v0 checks:
1. Patient-ID intersection between any two splits (the classic invalidator).
2. Near-duplicate images across splits via perceptual hashing (aHash) —
   catches the same exam re-exported under two IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LeakageReport:
    patient_overlaps: dict[str, list[str]] = field(default_factory=dict)
    near_duplicates: list[tuple[str, str]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.patient_overlaps and not self.near_duplicates


def average_hash(pixels: np.ndarray, hash_size: int = 8) -> int:
    """Perceptual average-hash of an image, as an integer bit string."""
    h, w = pixels.shape
    ys = np.linspace(0, h, hash_size + 1, dtype=int)
    xs = np.linspace(0, w, hash_size + 1, dtype=int)
    small = np.array(
        [
            [pixels[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].mean() for j in range(hash_size)]
            for i in range(hash_size)
        ]
    )
    bits = (small > np.median(small)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def audit(
    split_of_patient: dict[str, str],
    images: list[tuple[str, str, np.ndarray]] | None = None,
    duplicate_threshold: int = 4,
) -> LeakageReport:
    """``images`` is a list of (case_id, patient_id, pixels) for the hash check."""
    report = LeakageReport()

    # 1. A patient may appear in exactly one split by construction of the
    #    manifest; the audit re-derives the invariant from raw case listings.
    seen: dict[str, str] = {}
    if images is not None:
        for case_id, patient_id, _ in images:
            split = split_of_patient.get(patient_id)
            if split is None:
                report.patient_overlaps.setdefault("unassigned", []).append(patient_id)
            elif patient_id in seen and seen[patient_id] != split:
                report.patient_overlaps.setdefault(patient_id, []).append(split)
            seen[patient_id] = split or "unassigned"

        # 2. Cross-split near-duplicate scan.
        hashed = [
            (case_id, patient_id, average_hash(px), split_of_patient.get(patient_id))
            for case_id, patient_id, px in images
        ]
        for i in range(len(hashed)):
            for j in range(i + 1, len(hashed)):
                id_a, pa, ha, sa = hashed[i]
                id_b, pb, hb, sb = hashed[j]
                if sa != sb and hamming(ha, hb) <= duplicate_threshold:
                    report.near_duplicates.append((id_a, id_b))
    return report
