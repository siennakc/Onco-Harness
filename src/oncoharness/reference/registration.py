"""Translation registration with a QC gate (2I, inside the compare_prior tool).

v0 sits low on the correspondence ladder: FFT phase correlation recovers a
global translation, and a QC gate decides whether the alignment is usable.
When QC fails the tool REFUSES the comparison (ladder rung 5) — an
unregistered "comparison" is fluent, confident, and unfounded, so refusal is
a first-class result, not an error. Deformable registration (ANTs/ConvexAdam)
slots in behind the same QC contract in Phase 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegistrationResult:
    shift: tuple[int, int]         # (dy, dx) applied to the moving image
    ncc_before: float
    ncc_after: float
    passed_qc: bool
    reason: str


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def register_translation(
    fixed: np.ndarray,
    moving: np.ndarray,
    min_ncc_after: float = 0.35,
    min_improvement: float = -0.02,
) -> RegistrationResult:
    """Phase-correlation translation of ``moving`` onto ``fixed`` + QC verdict."""
    if fixed.shape != moving.shape:
        return RegistrationResult((0, 0), 0.0, 0.0, False, "shape mismatch — resample first")

    f = np.fft.fft2(fixed - fixed.mean())
    m = np.fft.fft2(moving - moving.mean())
    cross_power = f * np.conj(m)
    denom = np.abs(cross_power)
    denom[denom == 0] = 1.0
    corr = np.real(np.fft.ifft2(cross_power / denom))
    dy, dx = np.unravel_index(int(corr.argmax()), corr.shape)
    h, w = fixed.shape
    dy = dy - h if dy > h // 2 else dy
    dx = dx - w if dx > w // 2 else dx

    aligned = np.roll(np.roll(moving, dy, axis=0), dx, axis=1)
    ncc_before = _ncc(fixed, moving)
    ncc_after = _ncc(fixed, aligned)

    if ncc_after < min_ncc_after:
        return RegistrationResult(
            (int(dy), int(dx)), ncc_before, ncc_after, False,
            f"post-alignment similarity {ncc_after:.3f} below floor {min_ncc_after}",
        )
    if ncc_after - ncc_before < min_improvement:
        return RegistrationResult(
            (int(dy), int(dx)), ncc_before, ncc_after, False,
            "alignment made similarity worse — geometry is not a pure translation",
        )
    return RegistrationResult((int(dy), int(dx)), ncc_before, ncc_after, True, "ok")


def apply_shift(moving: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    dy, dx = shift
    return np.roll(np.roll(moving, dy, axis=0), dx, axis=1)
