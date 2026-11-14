"""Ironbow-style false colour for thermal frames (dashboard + overlays)."""

from __future__ import annotations

import numpy as np

# Key colours approximating a FLIR-style ironbow (black → blue → magenta → orange → yellow → white).
_KEYS = np.array(
    [
        [0, 0, 0],
        [16, 0, 64],
        [32, 0, 160],
        [128, 0, 176],
        [200, 24, 80],
        [255, 80, 16],
        [255, 160, 0],
        [255, 224, 80],
        [255, 255, 255],
    ],
    dtype=np.float32,
)


def _build_lut() -> np.ndarray:
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    stops = np.linspace(0, 255, len(_KEYS))
    for c in range(3):
        lut[:, 0, c] = np.clip(np.interp(np.arange(256), stops, _KEYS[:, c]), 0, 255).astype(np.uint8)
    return lut


IRONBOW_LUT = _build_lut()


def temps_to_ironbow(
    temps: np.ndarray,
    lo: float | None = None,
    hi: float | None = None,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """Map a Celsius matrix to a BGR ironbow image (uint8)."""
    import cv2

    finite = temps[np.isfinite(temps)]
    if finite.size == 0:
        return np.zeros((*temps.shape, 3), dtype=np.uint8)
    if lo is None:
        lo = float(np.percentile(finite, p_lo))
    if hi is None:
        hi = float(np.percentile(finite, p_hi))
    if hi - lo < 0.25:
        hi = lo + 0.25
    norm = np.clip((temps - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(cv2.merge([norm, norm, norm]), IRONBOW_LUT)
