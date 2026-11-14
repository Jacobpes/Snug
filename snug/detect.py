"""Element detection: tiny FOMO CNN with a JSON ROI fallback (best for the live demo)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from snug.nn import CLASS_NAMES, TinyFomo, grid_to_boxes, softmax
from snug.simulate import default_layout

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "detector.npz"
DEFAULT_ONNX = ROOT / "models" / "detector.onnx"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rois_from_config(cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not cfg:
        return [r.as_roi() for r in default_layout()]
    rois = cfg.get("rois")
    if rois:
        out = []
        for r in rois:
            item = {
                "name": r["name"],
                "element": r.get("element") or r.get("type") or "wall",
                "x": int(r["x"]),
                "y": int(r["y"]),
                "w": int(r["w"]),
                "h": int(r["h"]),
                "source": "config",
            }
            out.append(item)
        return out
    return [r.as_roi() | {"source": "config"} for r in default_layout()]


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0


def merge_with_rois(predicted: list[dict[str, Any]], rois: list[dict[str, Any]], iou_thresh: float = 0.15) -> list[dict[str, Any]]:
    """Prefer configured names/boxes when they overlap a prediction of the same element."""
    if not predicted:
        return [dict(r) for r in rois]
    used = set()
    merged: list[dict[str, Any]] = []
    for roi in rois:
        best_i, best = -1, 0.0
        for i, pred in enumerate(predicted):
            if i in used:
                continue
            if pred["element"] != roi["element"] and not (
                roi["element"] == "wall" and pred["element"] == "wall"
            ):
                continue
            score = _iou(roi, pred)
            if score > best:
                best, best_i = score, i
        if best_i >= 0 and best >= iou_thresh:
            used.add(best_i)
            pred = predicted[best_i]
            # Keep the stable config name and (for the demo house) the config box,
            # which is more reliable than a coarse 16 px grid on a known facade.
            item = dict(roi)
            item["source"] = "hybrid"
            item["model_box"] = {k: pred[k] for k in ("x", "y", "w", "h")}
            item["cells"] = pred.get("cells")
            merged.append(item)
        else:
            merged.append(dict(roi))
    # In hybrid mode the configured house layout is the source of truth; stray
    # coarse-grid boxes are ignored. Use mode "auto" to see raw model output.
    return merged


class ElementDetector:
    def __init__(
        self,
        *,
        weights: str | Path | None = None,
        config: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ):
        self.mode = mode
        self.rois = rois_from_config(config)
        self.model: TinyFomo | None = None
        self.onnx_session = None
        path = Path(weights) if weights else DEFAULT_WEIGHTS
        if path.is_file():
            self.model = TinyFomo.load(path)
        onnx_path = DEFAULT_ONNX
        # SNUG_NO_ONNX=1 forces the NumPy path (needed under qemu-user, where
        # onnxruntime aarch64 wheels can SIGSEGV even though they work on real A53).
        if onnx_path.is_file() and os.environ.get("SNUG_NO_ONNX") != "1":
            try:
                import onnxruntime as ort

                self.onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            except Exception:
                self.onnx_session = None

    @property
    def available(self) -> bool:
        return self.model is not None or self.onnx_session is not None

    def infer_grid(self, temps: np.ndarray, ambient_c: float | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        from snug.nn import temps_to_input

        x = temps_to_input(temps, ambient_c)
        if self.onnx_session is not None:
            inp = self.onnx_session.get_inputs()[0].name
            logits = self.onnx_session.run(None, {inp: x[None, ...]})[0][0]
        elif self.model is not None:
            logits = self.model.forward(x)
        else:
            raise RuntimeError("No detector weights loaded")
        grid = np.argmax(logits, axis=0).astype(np.uint8)
        return grid, softmax(logits, axis=0)

    def detect(self, temps: np.ndarray, ambient_c: float | None = None) -> list[dict[str, Any]]:
        if self.mode == "rois" or not self.available:
            return [dict(r) for r in self.rois]
        try:
            grid, scores = self.infer_grid(temps, ambient_c)
        except Exception:
            return [dict(r) for r in self.rois]
        predicted = grid_to_boxes(grid, scores)
        if self.mode == "auto":
            return predicted or [dict(r) for r in self.rois]
        return merge_with_rois(predicted, self.rois)
