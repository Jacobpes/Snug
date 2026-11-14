"""Tiny FOMO-style CNN: 1x192x256 → 6-class 12x16 grid.

Inference is implemented in NumPy so the demo does not require PyTorch or
onnxruntime on the UNO Q. An ONNX graph is also exported for optional use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

CLASS_NAMES = ("background", "wall", "window", "door", "roof", "corner")
INPUT_H, INPUT_W = 192, 256
GRID_H, GRID_W = 12, 16
CELL = INPUT_H // GRID_H  # 16


def temps_to_input(temps: np.ndarray, ambient_c: float | None = None) -> np.ndarray:
    """Normalize a Celsius frame to a 1x192x256 float32 tensor in roughly [0, 1]."""
    t = temps.astype(np.float32)
    if t.shape != (INPUT_H, INPUT_W):
        import cv2

        t = cv2.resize(t, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    ref = float(np.median(t)) if ambient_c is None else float(ambient_c)
    # Typical model-house ΔT is a few degrees; 12 °C span keeps contrast after warmup.
    x = np.clip((t - ref) / 12.0 + 0.25, 0.0, 1.0)
    return x[None, ...].astype(np.float32)


def _pad(x: np.ndarray, p: int) -> np.ndarray:
    if p <= 0:
        return x
    return np.pad(x, ((0, 0), (p, p), (p, p)), mode="constant")


def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """x: (C,H,W), weight: (Cout,Cin,k,k) → (Cout,Hout,Wout)."""
    x = _pad(x, padding)
    cin, h, w = x.shape
    cout, cin_w, k, _ = weight.shape
    assert cin == cin_w, f"channel mismatch {cin} vs {cin_w}"
    hout = (h - k) // stride + 1
    wout = (w - k) // stride + 1
    # As-strided windows: (hout, wout, cin, k, k)
    s_c, s_h, s_w = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(cin, hout, wout, k, k),
        strides=(s_c, s_h * stride, s_w * stride, s_h, s_w),
        writeable=False,
    )
    # windows: Cin, Ho, Wo, k, k  · weight: Cout, Cin, k, k
    y = np.einsum("ihwyx,oiyx->ohw", windows, weight, optimize=True)
    y += bias[:, None, None]
    return y.astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=np.float32)


class TinyFomo:
    """Four stride-2 3x3 convs + 1x1 classifier."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights

    @classmethod
    def load(cls, path: str | Path) -> "TinyFomo":
        data = np.load(path, allow_pickle=False)
        return cls({k: data[k] for k in data.files})

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (1,H,W) → logits (C,12,16)."""
        w = self.w
        h = relu(conv2d(x, w["c1_w"], w["c1_b"], stride=2, padding=1))
        h = relu(conv2d(h, w["c2_w"], w["c2_b"], stride=2, padding=1))
        h = relu(conv2d(h, w["c3_w"], w["c3_b"], stride=2, padding=1))
        h = relu(conv2d(h, w["c4_w"], w["c4_b"], stride=2, padding=1))
        logits = conv2d(h, w["head_w"], w["head_b"], stride=1, padding=0)
        return logits

    def predict_grid(self, temps: np.ndarray, ambient_c: float | None = None) -> np.ndarray:
        logits = self.forward(temps_to_input(temps, ambient_c))
        return np.argmax(logits, axis=0).astype(np.uint8)


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def grid_to_boxes(
    grid: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    min_cells: int = 3,
    min_score: float = 0.35,
) -> list[dict[str, Any]]:
    """Connected components on the class grid → axis-aligned boxes in pixel space."""
    import cv2

    boxes: list[dict[str, Any]] = []
    h, w = grid.shape
    for cls_id, name in enumerate(CLASS_NAMES):
        if cls_id == 0:
            continue
        mask = (grid == cls_id).astype(np.uint8)
        if scores is not None:
            mask = np.logical_and(mask, scores[cls_id] >= min_score).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < min_cells:
                continue
            boxes.append(
                {
                    "name": f"{name}_{len([b for b in boxes if b['element'] == name]) + 1}",
                    "element": name,
                    "x": int(x * CELL),
                    "y": int(y * CELL),
                    "w": int(bw * CELL),
                    "h": int(bh * CELL),
                    "cells": int(area),
                    "source": "model",
                }
            )
    return boxes
