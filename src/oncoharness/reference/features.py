"""Feature extraction standing in for cached frozen-FM embeddings (T-2.1).

The v0 workflow is frozen-encoder + light head on cached embeddings. On real
data the encoder is a pathology/radiology foundation model with its outputs
cached to disk; here a deterministic hand-crafted embedding keeps the exact
same downstream shape (fixed-length float vector per crop) so the head,
calibration, DFR refits, and kNN atlas are all real code from day one.
"""

from __future__ import annotations

import numpy as np

EMBED_DIM = 20


def embed_crop(crop: np.ndarray) -> np.ndarray:
    """Fixed-length descriptor of one image crop. Deterministic, [0,1]-scaled."""
    crop = crop.astype(np.float64)
    if crop.size == 0:
        return np.zeros(EMBED_DIM)
    hist, _ = np.histogram(crop, bins=8, range=(0.0, 1.0), density=False)
    hist = hist / max(crop.size, 1)
    gy, gx = np.gradient(crop)
    grad_mag = np.hypot(gx, gy)
    center = crop[
        crop.shape[0] // 4 : 3 * crop.shape[0] // 4,
        crop.shape[1] // 4 : 3 * crop.shape[1] // 4,
    ]
    ring_mean = (crop.sum() - center.sum()) / max(crop.size - center.size, 1)
    feats = np.array(
        [
            crop.mean(),
            crop.std(),
            crop.max(),
            crop.min(),
            np.median(crop),
            grad_mag.mean(),
            grad_mag.std(),
            float(center.mean() if center.size else 0.0),
            ring_mean,
            float(center.mean() - ring_mean if center.size else 0.0),  # blob contrast
            crop.shape[0] / 256.0,
            crop.shape[1] / 256.0,
        ]
    )
    return np.concatenate([feats, hist])
