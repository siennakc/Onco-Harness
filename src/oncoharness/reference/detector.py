"""Candidate detector (axiom A2: detector proposes, VLM adjudicates).

v0 ships a deterministic difference-of-Gaussians blob detector: it runs
anywhere, is fully reproducible, and exercises every downstream contract
(scored boxes, high sensitivity / low precision). A real specialist
(nnDetection, nnU-Net ResEnc) drops in behind the same ``Detector`` protocol
without touching the harness — the detector's recall is the ceiling of the
whole system, so upgrading this module is the main Phase 7 lever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Candidate:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in source pixels
    score: float                    # calibrated-ish [0,1]; higher = more suspicious


class Detector(Protocol):
    def propose(self, pixels: np.ndarray) -> list[Candidate]: ...


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur in pure numpy (no scipy dependency)."""
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    padded = np.pad(img, radius, mode="reflect")
    blurred = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded)
    blurred = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, blurred)
    return blurred


class DoGBlobDetector:
    """Difference-of-Gaussians peak proposer, tuned for sensitivity over precision."""

    def __init__(
        self,
        sigma_small: float = 2.0,
        sigma_large: float = 6.0,
        score_threshold: float = 0.10,
        max_candidates: int = 16,
        contrast_scale: float = 0.15,
    ) -> None:
        self.sigma_small = sigma_small
        self.sigma_large = sigma_large
        self.score_threshold = score_threshold
        self.max_candidates = max_candidates
        self.contrast_scale = contrast_scale

    def propose(self, pixels: np.ndarray) -> list[Candidate]:
        dog = _gaussian_blur(pixels, self.sigma_small) - _gaussian_blur(
            pixels, self.sigma_large
        )
        dog = np.clip(dog, 0, None)
        if dog.max() <= 0:
            return []
        # Score by ABSOLUTE contrast squashed to [0,1) — never self-normalized:
        # normalizing by the image's own max would give every image a score-1.0
        # peak and destroy all case-level separation.
        response = 1.0 - np.exp(-dog / self.contrast_scale)

        candidates: list[Candidate] = []
        working = response.copy()
        h, w = working.shape
        radius = int(3 * self.sigma_large)
        for _ in range(self.max_candidates):
            peak = float(working.max())
            if peak < self.score_threshold:
                break
            cy, cx = np.unravel_index(int(working.argmax()), working.shape)
            box = (
                max(0, cx - radius),
                max(0, cy - radius),
                min(w - 1, cx + radius),
                min(h - 1, cy + radius),
            )
            # Score blends peak response with local contrast for a monotone signal.
            candidates.append(Candidate(box=box, score=round(peak, 4)))
            y0, y1 = max(0, cy - 2 * radius), min(h, cy + 2 * radius)
            x0, x1 = max(0, cx - 2 * radius), min(w, cx + 2 * radius)
            working[y0:y1, x0:x1] = 0.0  # non-max suppression
        return candidates
