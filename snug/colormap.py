"""Ironbow-style false colour for thermal frames (dashboard + overlays)."""

from __future__ import annotations

import numpy as np

# Key colours of the Snug thermal ramp (same as the pitch deck): near-black violet -> 3B0F6E -> A3207F ->
# EC5A26 -> F9B233 -> pale cream. Still a classic ironbow-style monotonic hot-is-brighter ramp.
_KEYS = np.array(
    [
        [9, 6, 24],
        [59, 15, 110],
        [110, 22, 121],
        [163, 32, 127],
        [212, 55, 78],
        [236, 90, 38],
        [249, 178, 51],
        [255, 233, 168],
        [255, 250, 232],
    ],
    dtype=np.float32,
)


def _build_lut() -> np.ndarray:
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    stops = np.linspace(0, 255, len(_KEYS))
    for c in range(3):
        lut[:, 0, c] = np.clip(np.interp(np.arange(256), stops, _KEYS[:, c]), 0, 255).astype(np.uint8)
    return lut


# cv2 images are BGR, the key colours above are written as RGB.
IRONBOW_LUT = np.ascontiguousarray(_build_lut()[:, :, ::-1])


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
