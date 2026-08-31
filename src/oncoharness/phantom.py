"""Synthetic phantom dataset (T-3.2): plumbing tests without real data.

Generates grayscale "scans" with optional inserted lesions (bright Gaussian
blobs on structured background) plus patient/site metadata, so splits,
leakage audits, the detector, the harness state machine, and the eval gate all
run end-to-end on a laptop. Phantoms are excluded from every real eval by
construction — they exist to test the measurement apparatus, not the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PhantomCase:
    case_id: str
    patient_id: str
    site: str
    pixels: np.ndarray                    # float32 [0,1], HxW
    label: int                            # 1 = lesion present
    lesion_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    pixel_spacing_mm: tuple[float, float] = (0.1, 0.1)
    density_band: str = "b"
    age_band: str = "50-59"


def _background(rng: np.random.Generator, size: int) -> np.ndarray:
    """Smooth structured background: sum of large-scale random cosines + noise."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / size
    img = np.zeros((size, size))
    for _ in range(4):
        fx, fy = rng.uniform(0.5, 2.5, 2)
        px, py = rng.uniform(0, np.pi, 2)
        img += rng.uniform(0.05, 0.15) * np.cos(2 * np.pi * fx * xx + px) * np.cos(
            2 * np.pi * fy * yy + py
        )
    img += rng.normal(0, 0.02, (size, size))
    img -= img.min()
    return img / max(img.max(), 1e-9)


def _insert_lesion(
    img: np.ndarray, rng: np.random.Generator, size: int
) -> tuple[int, int, int, int]:
    r = int(rng.integers(size // 32, size // 12))
    cy = int(rng.integers(r + 2, size - r - 2))
    cx = int(rng.integers(r + 2, size - r - 2))
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r / 2.0) ** 2)))
    img += 0.55 * blob
    return (cx - r, cy - r, cx + r, cy + r)  # x0, y0, x1, y1


def generate_dataset(
    n_patients: int = 60,
    images_per_patient: int = 2,
    prevalence: float = 0.3,
    sites: tuple[str, ...] = ("site_a", "site_b"),
    size: int = 128,
    seed: int = 7,
) -> list[PhantomCase]:
    """Positive/negative is decided per patient so grouped splits matter."""
    rng = np.random.default_rng(seed)
    cases: list[PhantomCase] = []
    for p in range(n_patients):
        patient_id = f"P{p:04d}"
        site = sites[p % len(sites)]
        positive_patient = rng.random() < prevalence
        for i in range(images_per_patient):
            img = _background(rng, size)
            boxes: list[tuple[int, int, int, int]] = []
            label = 0
            if positive_patient:
                boxes.append(_insert_lesion(img, rng, size))
                label = 1
            img = np.clip(img, 0.0, 1.0)
            cases.append(
                PhantomCase(
                    case_id=f"{patient_id}_img{i}",
                    patient_id=patient_id,
                    site=site,
                    pixels=img.astype(np.float32),
                    label=label,
                    lesion_boxes=boxes,
                    density_band=("a" if p % 3 == 0 else "b" if p % 3 == 1 else "c"),
                    age_band=("40-49" if p % 2 == 0 else "50-59"),
                )
            )
    return cases
