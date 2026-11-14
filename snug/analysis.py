"""Per-region thermal analysis: normality, defect type, severity, heat-loss estimate.

Defect type is a rule/statistical classifier on the radiometric matrix:

- compact warm blob on a wall → insulation gap
- elongated warm line along a joint, corner, or frame → thermal bridge
- streaky / plume-like anomaly at a door or window edge, often with
  temporal fluctuation → air leak

Severity and Q = U·A·ΔT are demo-calibrated estimates, not a certified audit.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from snug.actions import ACTIONS, DEFAULT_U, action_for, verification_verdict

SEVERITY_RANK = {"ok": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class RegionFinding:
    name: str
    element: str
    x: int
    y: int
    w: int
    h: int
    mean_c: float
    max_c: float
    min_c: float
    std_c: float
    ambient_c: float
    reference_c: float
    delta_c: float
    indoor_c: float | None
    anomaly_pixels: int
    anomaly_area_m2: float
    anomaly_frac: float
    compactness: float
    aspect_ratio: float
    edge_overlap: float
    fluctuation_c: float | None
    normal: bool
    defect: str
    severity: str
    q_watts: float
    u_value: float
    priority: float
    fix: str
    verify: str
    hud: str
    reasons: list[str] = field(default_factory=list)
    t_surface_c: float | None = None
    t_ref_surface_c: float | None = None
    cost: dict[str, Any] | None = None
    cost_label: str | None = None
    cost_short: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def six_answers(self) -> dict[str, Any]:
        return {
            "element": self.element,
            "normal": self.normal,
            "defect": self.defect,
            "severity": self.severity if not self.normal else "ok",
            "fix": self.fix,
            "verify": self.verify,
            "cost": self.cost_short or (self.cost or {}).get("short"),
            "cost_label": self.cost_label or (self.cost or {}).get("label"),
        }


def _clip_box(x: int, y: int, w: int, h: int, height: int, width: int) -> tuple[int, int, int, int]:
    x0 = int(max(0, min(x, width - 1)))
    y0 = int(max(0, min(y, height - 1)))
    x1 = int(max(x0 + 1, min(x + max(w, 1), width)))
    y1 = int(max(y0 + 1, min(y + max(h, 1), height)))
    return x0, y0, x1, y1


def _inset(x0: int, y0: int, x1: int, y1: int, margin: int) -> tuple[int, int, int, int]:
    if margin <= 0:
        return x0, y0, x1, y1
    ix0, iy0 = x0 + margin, y0 + margin
    ix1, iy1 = x1 - margin, y1 - margin
    if ix1 <= ix0 + 2 or iy1 <= iy0 + 2:
        return x0, y0, x1, y1
    return ix0, iy0, ix1, iy1


def region_stats(temps: np.ndarray, box: dict[str, Any], inset: int = 0) -> dict[str, Any]:
    h, w = temps.shape
    x0, y0, x1, y1 = _clip_box(box["x"], box["y"], box["w"], box["h"], h, w)
    cx0, cy0, cx1, cy1 = _inset(x0, y0, x1, y1, inset)
    crop = temps[cy0:cy1, cx0:cx1]
    return {
        "x": x0,
        "y": y0,
        "w": x1 - x0,
        "h": y1 - y0,
        "mean_c": float(crop.mean()),
        "max_c": float(crop.max()),
        "min_c": float(crop.min()),
        "std_c": float(crop.std()),
        "crop": crop,
    }


def _anomaly_mask(crop: np.ndarray, reference_c: float, thresh_c: float) -> np.ndarray:
    return crop >= (reference_c + thresh_c)


def shape_features(mask: np.ndarray) -> tuple[float, float, float]:
    """Return (compactness, aspect_ratio, area_frac) for a boolean mask."""
    pixels = int(mask.sum())
    if pixels < 4:
        return 0.0, 1.0, float(pixels) / max(mask.size, 1)
    area = float(pixels)
    m8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peri = 0.0
    if contours:
        peri = float(sum(cv2.arcLength(c, True) for c in contours))
    compactness = float(4.0 * np.pi * area / (peri * peri)) if peri > 1e-3 else 0.0
    compactness = float(np.clip(compactness, 0.0, 2.0))
    ys, xs = np.nonzero(mask)
    height = float(ys.max() - ys.min() + 1)
    width = float(xs.max() - xs.min() + 1)
    aspect = max(height, width) / max(min(height, width), 1.0)
    return compactness, aspect, area / mask.size


def edge_overlap_fraction(mask: np.ndarray, border: int = 4) -> float:
    """How much of the anomaly sits in the outer border of the crop (openings)."""
    if mask.size == 0 or not mask.any():
        return 0.0
    h, w = mask.shape
    b = min(border, h // 3, w // 3)
    if b <= 0:
        return 0.0
    rim = np.zeros_like(mask)
    rim[:b, :] = True
    rim[-b:, :] = True
    rim[:, :b] = True
    rim[:, -b:] = True
    return float((mask & rim).sum()) / float(mask.sum())


def classify_defect(
    *,
    element: str,
    delta_c: float,
    compactness: float,
    aspect_ratio: float,
    edge_overlap: float,
    fluctuation_c: float | None,
    anomaly_frac: float,
    normal_dt: float,
) -> tuple[str, list[str]]:
    """Rule-based defect type. Returns (defect, reasons)."""
    reasons: list[str] = []
    if delta_c < normal_dt or anomaly_frac < 0.02:
        return "normal", [f"ΔT {delta_c:.2f} °C below normal threshold {normal_dt:.2f} °C"]

    # Air leak: openings + streaky / fluttering
    leak_shape = aspect_ratio >= 2.2 or (compactness < 0.35 and edge_overlap >= 0.35)
    leak_time = fluctuation_c is not None and fluctuation_c >= 0.12
    if element in {"door", "window"} and (leak_shape or edge_overlap >= 0.45 or leak_time):
        if leak_time:
            reasons.append(f"temporal fluctuation {fluctuation_c:.2f} °C")
        if edge_overlap >= 0.35:
            reasons.append(f"anomaly concentrated at the opening edge ({edge_overlap:.0%})")
        if leak_shape:
            reasons.append(f"streaky/elongated plume (aspect {aspect_ratio:.1f}, compactness {compactness:.2f})")
        return "air_leak", reasons

    # Thermal bridge: very elongated, typically along a joint/corner/frame
    if aspect_ratio >= 3.2 or (aspect_ratio >= 2.4 and compactness < 0.28):
        reasons.append(f"elongated pattern along a joint (aspect {aspect_ratio:.1f}, compactness {compactness:.2f})")
        if element in {"corner", "window", "door", "roof"}:
            reasons.append(f"located on {element}")
        return "thermal_bridge", reasons

    # Insulation gap: compact blob on an opaque wall/roof only.
    # A uniformly warm window pane or door leaf is expected (higher U), not missing insulation.
    if element in {"wall", "roof"} and compactness >= 0.32 and aspect_ratio < 3.0:
        reasons.append(f"compact warm area (compactness {compactness:.2f}, aspect {aspect_ratio:.1f})")
        reasons.append(f"on {element}")
        return "insulation_gap", reasons

    # Fallbacks
    if element in {"door", "window"}:
        reasons.append("residual warm anomaly at an opening, treated as a likely air leak")
        return "air_leak", reasons
    if element == "corner":
        reasons.append("warm anomaly at a corner joint, treated as a thermal bridge")
        return "thermal_bridge", reasons
    reasons.append("warm anomaly on an opaque element, treated as an insulation gap")
    return "insulation_gap", reasons


def severity_of(delta_c: float, anomaly_frac: float, thresholds: dict[str, float]) -> str:
    if delta_c < thresholds["normal_dt"]:
        return "ok"
    high = delta_c >= thresholds["high_dt"] or (
        delta_c >= thresholds["medium_dt"] and anomaly_frac >= thresholds["area_frac_high"]
    )
    if high:
        return "high"
    if delta_c >= thresholds["medium_dt"] or anomaly_frac >= thresholds["area_frac_medium"]:
        return "medium"
    return "low"


def estimate_q(u_value: float, area_m2: float, delta_c: float) -> float:
    """Q = U · A · ΔT (watts). Labelled as an estimate everywhere in the UI."""
    return max(0.0, float(u_value) * float(area_m2) * max(0.0, float(delta_c)))


class HouseAnalyzer:
    """Stateful analyzer: keeps a short history for air-leak fluctuation + verify snapshots."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self.meters_per_pixel = float(cfg.get("meters_per_pixel", 0.002))
        self.u_values = {**DEFAULT_U, **(cfg.get("u_values") or {})}
        self.thresholds = {
            "normal_dt": 1.5,
            "medium_dt": 3.0,
            "high_dt": 5.0,
            "area_frac_medium": 0.15,
            "area_frac_high": 0.30,
            "anomaly_offset_c": 1.2,
            "verify_dt_drop_frac": 0.4,
            "verify_area_drop_frac": 0.4,
            **(cfg.get("thresholds") or {}),
        }
        self.reference_name = cfg.get("reference_roi", "insulated_wall")
        self.history_s = float(cfg.get("fluctuation_window_s", 4.0))
        self._hist: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        self.before: dict[str, dict[str, Any]] | None = None
        self.last_verification: dict[str, Any] | None = None

    def _push(self, name: str, now: float, mean_c: float) -> None:
        hist = self._hist[name]
        hist.append((now, mean_c))
        cutoff = now - self.history_s - 0.5
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    def _fluctuation(self, name: str) -> float | None:
        vals = [v for _, v in self._hist[name]]
        if len(vals) < 6:
            return None
        return float(np.std(np.asarray(vals, dtype=np.float64)))

    def analyze(
        self,
        temps: np.ndarray,
        regions: list[dict[str, Any]],
        *,
        ambient_c: float | None = None,
        indoor_c: float | None = None,
        now: float = 0.0,
    ) -> list[RegionFinding]:
        if not regions:
            return []

        # Reference: named insulated wall, else the coolest 'wall' region mean.
        ref = None
        for r in regions:
            if r.get("name") == self.reference_name:
                ref = region_stats(temps, r, inset=8)
                break
        if ref is None:
            walls = [region_stats(temps, r, inset=6) for r in regions if r.get("element") == "wall"]
            ref = min(walls, key=lambda s: s["mean_c"]) if walls else region_stats(temps, regions[0])
        reference_c = ref["mean_c"]

        if ambient_c is None:
            # Use a 12-px border median as outdoor/ambient proxy when not provided.
            border = np.concatenate(
                [
                    temps[:8, :].ravel(),
                    temps[-8:, :].ravel(),
                    temps[:, :8].ravel(),
                    temps[:, -8:].ravel(),
                ]
            )
            ambient_c = float(np.median(border))

        findings: list[RegionFinding] = []
        px_area = self.meters_per_pixel**2
        for r in regions:
            if r.get("element") == "background":
                continue
            name = str(r.get("name") or r.get("element") or "region")
            st = region_stats(temps, r, inset=8 if name == self.reference_name else 0)
            element = str(r.get("element") or "wall")
            self._push(name, now, st["mean_c"])
            fluct = self._fluctuation(name)
            # Opaque elements: compare to the insulated reference (or ambient for that wall).
            # Openings: a warm pane/door leaf is expected; look for streaks hotter than the element itself.
            opening = element in {"window", "door"}
            if name == self.reference_name:
                baseline = float(ambient_c)
            elif opening:
                baseline = float(np.percentile(st["crop"], 40))
            else:
                baseline = float(reference_c)
            delta = st["mean_c"] - (float(ambient_c) if name == self.reference_name else (baseline if opening else reference_c))
            if opening:
                # Surface excess of the leak relative to the pane, not the whole-element mean vs the wall.
                delta = float(st["max_c"] - baseline) * 0.5 + float(st["mean_c"] - baseline) * 0.5

            mask = _anomaly_mask(st["crop"], baseline, self.thresholds["anomaly_offset_c"])
            compactness, aspect, _ = shape_features(mask)
            anomaly_px = int(mask.sum())
            anomaly_frac = anomaly_px / mask.size if mask.size else 0.0
            if anomaly_px:
                t_surface = float(st["crop"][mask].mean())
            else:
                t_surface = float(st["mean_c"])
            t_ref_surface = float(baseline) if opening else float(reference_c)
            if name == self.reference_name:
                t_ref_surface = float(reference_c)
            edge_ov = edge_overlap_fraction(mask)
            defect, reasons = classify_defect(
                element=element,
                delta_c=delta,
                compactness=compactness,
                aspect_ratio=aspect,
                edge_overlap=edge_ov,
                fluctuation_c=fluct,
                anomaly_frac=anomaly_frac,
                normal_dt=float(self.thresholds["normal_dt"]),
            )
            sev = severity_of(delta, anomaly_frac, self.thresholds)
            if defect == "normal":
                sev = "ok"
            u = float(self.u_values.get(element, 0.35))
            area_m2 = anomaly_px * px_area
            # Heat-loss estimate uses indoor-outdoor ΔT when known, else surface ΔT.
            dt_q = (indoor_c - ambient_c) if indoor_c is not None else max(delta, st["mean_c"] - ambient_c)
            q = estimate_q(u, area_m2, dt_q)
            act = action_for(defect)
            priority = SEVERITY_RANK[sev] * 10.0 + q + max(delta, 0.0)
            hud = act["hud"]
            if defect != "normal":
                hud = f"{act['hud']} at {name.replace('_', ' ')} · Severity {sev.upper()}"
            findings.append(
                RegionFinding(
                    name=name,
                    element=element,
                    x=st["x"],
                    y=st["y"],
                    w=st["w"],
                    h=st["h"],
                    mean_c=st["mean_c"],
                    max_c=st["max_c"],
                    min_c=st["min_c"],
                    std_c=st["std_c"],
                    ambient_c=float(ambient_c),
                    reference_c=float(reference_c),
                    delta_c=float(delta),
                    indoor_c=indoor_c,
                    anomaly_pixels=anomaly_px,
                    anomaly_area_m2=float(area_m2),
                    anomaly_frac=float(anomaly_frac),
                    compactness=float(compactness),
                    aspect_ratio=float(aspect),
                    edge_overlap=float(edge_ov),
                    fluctuation_c=fluct,
                    normal=defect == "normal",
                    defect=defect,
                    severity=sev,
                    q_watts=float(q),
                    u_value=u,
                    priority=float(priority),
                    fix=act["fix"],
                    verify=act["verify"],
                    hud=hud,
                    reasons=reasons,
                    t_surface_c=t_surface,
                    t_ref_surface_c=t_ref_surface,
                )
            )
        findings.sort(key=lambda f: f.priority, reverse=True)
        return findings

    def snapshot(self, findings: list[RegionFinding]) -> dict[str, dict[str, Any]]:
        return {
            f.name: {
                "name": f.name,
                "element": f.element,
                "defect": f.defect,
                "severity": f.severity,
                "delta_c": f.delta_c,
                "anomaly_pixels": f.anomaly_pixels,
                "anomaly_area_m2": f.anomaly_area_m2,
                "q_watts": f.q_watts,
                "mean_c": f.mean_c,
            }
            for f in findings
        }

    def mark_before(self, findings: list[RegionFinding]) -> dict[str, dict[str, Any]]:
        self.before = self.snapshot(findings)
        self.last_verification = None
        return self.before

    def compare_after(self, findings: list[RegionFinding]) -> dict[str, Any]:
        if not self.before:
            raise RuntimeError("No 'before' snapshot. Capture a baseline first.")
        after = self.snapshot(findings)
        regions = []
        for name, b in self.before.items():
            a = after.get(name) or {**b, "delta_c": 0.0, "anomaly_pixels": 0, "anomaly_area_m2": 0.0, "defect": "normal"}
            v = verification_verdict(
                b,
                a,
                dt_drop_frac=float(self.thresholds["verify_dt_drop_frac"]),
                area_drop_frac=float(self.thresholds["verify_area_drop_frac"]),
            )
            v["name"] = name
            v["element"] = b.get("element")
            v["defect_before"] = b.get("defect")
            v["defect_after"] = a.get("defect")
            regions.append(v)
        passed = all(r["passed"] or (self.before[r["name"]].get("defect") == "normal") for r in regions)
        # Require that every previously abnormal region improved.
        abnormal = [r for r in regions if self.before[r["name"]].get("defect") != "normal"]
        passed = all(r["passed"] for r in abnormal) if abnormal else passed
        report = {
            "passed": passed,
            "summary": (
                "Fix verified: ΔT and anomaly area dropped on the flagged regions."
                if passed
                else "Not verified yet — one or more flagged regions did not improve enough."
            ),
            "regions": regions,
        }
        self.last_verification = report
        return report
