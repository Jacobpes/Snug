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

# Sensitivity presets. "low" keeps the original demo-house thresholds exactly.
SENSITIVITY_OPTIONS = ["low", "normal", "high"]
DEFAULT_SENSITIVITY = "low"  # analyzer default; config.example.json / the server pick "high" for the live demo
SENSITIVITY_PRESETS: dict[str, dict[str, float]] = {
    "low": {
        "normal_dt": 1.5,
        "anomaly_offset_c": 1.2,
        "min_anomaly_frac": 0.02,
        "medium_dt": 3.0,
        "high_dt": 5.0,
        "min_anomaly_px": 0,
        "smooth_px": 0,
    },
    "normal": {
        "normal_dt": 1.0,
        "anomaly_offset_c": 0.8,
        "min_anomaly_frac": 0.02,
        "medium_dt": 2.5,
        "high_dt": 4.5,
        "min_anomaly_px": 6,
        "smooth_px": 3,
    },
    "high": {
        "normal_dt": 0.5,
        "anomaly_offset_c": 0.4,
        "min_anomaly_frac": 0.01,
        "medium_dt": 1.5,
        "high_dt": 3.0,
        "min_anomaly_px": 6,
        "smooth_px": 3,
    },
}
# Time constant (s) of the exponential smoothing of region deltas / anomaly fractions.
EMA_TAU_S = 1.0
DEFAULT_SCENE_T_OUT_C = 0.0


def normalize_sensitivity(value: Any) -> str:
    key = str(value or DEFAULT_SENSITIVITY).strip().lower()
    aliases = {"med": "normal", "medium": "normal", "default": "normal", "hi": "high", "max": "high", "lo": "low"}
    key = aliases.get(key, key)
    if key not in SENSITIVITY_PRESETS:
        raise ValueError(f"sensitivity must be one of {SENSITIVITY_OPTIONS}, got {value!r}")
    return key


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
    direction: str = "warm"  # "warm" (excess heat) or "cold" (cold spot seen from inside)
    live: bool = False  # True = live interior survey (no simulator meta)

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


def _cold_mask(crop: np.ndarray, baseline_c: float, thresh_c: float) -> np.ndarray:
    return crop <= (baseline_c - thresh_c)


def drop_small_blobs(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected components smaller than ``min_px`` pixels (sensor noise)."""
    if min_px <= 1 or not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[labels]


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


def _classify_impl(
    *,
    element: str,
    delta_c: float,
    compactness: float,
    aspect_ratio: float,
    edge_overlap: float,
    fluctuation_c: float | None,
    anomaly_frac: float,
    normal_dt: float,
    min_frac: float = 0.02,
) -> tuple[str, list[str]]:
    """Rule-based defect type. Returns (defect, reasons)."""
    reasons: list[str] = []
    if delta_c < normal_dt or anomaly_frac < min_frac:
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
    min_frac: float = 0.02,
    direction: str = "warm",
) -> tuple[str, list[str]]:
    """Rule-based defect type for a warm excess or (seen from inside) a cold deficit."""
    defect, reasons = _classify_impl(
        element=element,
        delta_c=delta_c,
        compactness=compactness,
        aspect_ratio=aspect_ratio,
        edge_overlap=edge_overlap,
        fluctuation_c=fluctuation_c,
        anomaly_frac=anomaly_frac,
        normal_dt=normal_dt,
        min_frac=min_frac,
    )
    if direction == "cold":
        reasons = [r.replace("warm", "cold") for r in reasons]
    return defect, reasons


def severity_of(delta_c: float, anomaly_frac: float, thresholds: dict[str, float]) -> str:
    if delta_c < thresholds["normal_dt"]:
        return "ok"
    high = delta_c >= thresholds["high_dt"] or (
        delta_c >= thresholds["medium_dt"] and anomaly_frac >= thresholds["area_frac_high"]
    )
    if high:
        return "high"
    # A large area alone only makes it "medium" when the deviation is also clearly above noise.
    if delta_c >= thresholds["medium_dt"] or (
        anomaly_frac >= thresholds["area_frac_medium"] and delta_c >= 0.5 * thresholds["medium_dt"]
    ):
        return "medium"
    return "low"


def estimate_q(u_value: float, area_m2: float, delta_c: float) -> float:
    """Q = U · A · ΔT (watts). Labelled as an estimate everywhere in the UI."""
    return max(0.0, float(u_value) * float(area_m2) * max(0.0, float(delta_c)))


class HouseAnalyzer:
    """Stateful analyzer: keeps a short history for air-leak fluctuation + verify snapshots.

    Two survey situations are handled:

    * simulator / model-house demo (``ambient_c`` and ``indoor_c`` supplied): exterior view,
      leaks are WARM anomalies relative to the insulated reference wall;
    * live camera (both ``None``): an interior survey. Leaks and cold bridges are COLD spots,
      so each region is tested in both directions and the larger deviation wins.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self.meters_per_pixel = float(cfg.get("meters_per_pixel", 0.002))
        self.u_values = {**DEFAULT_U, **(cfg.get("u_values") or {})}
        self._threshold_base = {
            "area_frac_medium": 0.15,
            "area_frac_high": 0.30,
            "verify_dt_drop_frac": 0.4,
            "verify_area_drop_frac": 0.4,
        }
        # Explicit values in cfg["thresholds"] win over the sensitivity preset.
        self._threshold_overrides = dict(cfg.get("thresholds") or {})
        self._smooth: dict[str, dict[str, float]] = {}
        self.sensitivity = DEFAULT_SENSITIVITY
        self.thresholds: dict[str, float] = {}
        self.set_sensitivity(cfg.get("sensitivity", DEFAULT_SENSITIVITY))
        self.reference_name = cfg.get("reference_roi", "insulated_wall")
        self.history_s = float(cfg.get("fluctuation_window_s", 4.0))
        self.ema_tau_s = float(cfg.get("smoothing_tau_s", EMA_TAU_S))
        self._hist: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        self._scene: tuple[float, float] | None = None  # (time, EMA of the frame median)
        self.scene_c: float | None = None
        self.live = False
        self.before: dict[str, dict[str, Any]] | None = None
        self.last_verification: dict[str, Any] | None = None

    # -- configuration -------------------------------------------------
    def set_sensitivity(self, value: Any) -> str:
        key = normalize_sensitivity(value)
        self.sensitivity = key
        self.thresholds = {**self._threshold_base, **SENSITIVITY_PRESETS[key], **self._threshold_overrides}
        self._smooth.clear()
        return key

    # -- state ---------------------------------------------------------
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

    def _alpha(self, last_t: float, now: float) -> float | None:
        dt = now - last_t
        if dt <= 1e-6 or dt > 3.0 * self.ema_tau_s:
            return None  # same/backwards timestamp or stale: restart from the raw value
        return float(1.0 - np.exp(-dt / max(self.ema_tau_s, 1e-3)))

    def _ema(self, name: str, now: float, vals: dict[str, float]) -> dict[str, float]:
        """Exponential smoothing (~``ema_tau_s``) of the per-region measurements."""
        prev = self._smooth.get(name)
        alpha = self._alpha(prev["_t"], now) if prev else None
        if prev is None or alpha is None:
            out = dict(vals)
        else:
            out = {k: prev[k] + alpha * (v - prev[k]) for k, v in vals.items()}
        out["_t"] = now
        self._smooth[name] = out
        return out

    def _update_scene(self, temps: np.ndarray, now: float) -> float:
        med = float(np.median(temps))
        prev = self._scene
        alpha = self._alpha(prev[0], now) if prev else None
        val = med if prev is None or alpha is None else prev[1] + alpha * (med - prev[1])
        self._scene = (now, val)
        self.scene_c = val
        return val

    def _blur(self, crop: np.ndarray) -> np.ndarray:
        k = int(self.thresholds.get("smooth_px", 0))
        c32 = np.ascontiguousarray(crop, dtype=np.float32)
        if k >= 2 and min(c32.shape) >= k:
            return cv2.blur(c32, (k, k))
        return c32

    # -- analysis ------------------------------------------------------
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
        live = ambient_c is None and indoor_c is None
        self.live = live
        self._update_scene(temps, now)
        th = self.thresholds
        offset = float(th["anomaly_offset_c"])
        min_px = int(th.get("min_anomaly_px", 0))
        min_frac = float(th.get("min_anomaly_frac", 0.02))

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
            # (Display only: the cost model never uses it for a live interior scan.)
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
            is_ref = name == self.reference_name
            st = region_stats(temps, r, inset=8 if is_ref else 0)
            element = str(r.get("element") or "wall")
            self._push(name, now, st["mean_c"])
            fluct = self._fluctuation(name)
            crop = st["crop"]
            sm = self._blur(crop)
            mean = float(st["mean_c"])
            opening = element in {"window", "door"}

            # Baselines and deviation measures for both directions.
            # Warm (exterior view / heat source): excess over the baseline.
            # Cold (interior view): deficit below the baseline.
            if opening:
                # A warm pane/door leaf is expected; look for streaks hotter (or colder) than the element itself.
                base_w = float(np.percentile(crop, 40))
                base_c = float(np.percentile(crop, 60))
                dw = 0.5 * (float(sm.max()) - base_w) + 0.5 * (mean - base_w)
                dc = 0.5 * (base_c - float(sm.min())) + 0.5 * (base_c - mean)
            else:
                # The reference wall is the healthy baseline: it is only checked for hot/cold spots
                # inside itself (it used to be compared against outdoor air, which flags a good wall
                # as soon as thresholds get sensitive).
                base_w = base_c = float(reference_c)
                dw = mean - base_w
                if live:
                    # Interior survey: a small local anomaly barely moves the mean, so weigh the tail too.
                    dw = max(dw, 0.5 * (float(np.percentile(sm, 98)) - base_w) + 0.5 * (mean - base_w))
                dc = 0.5 * (base_c - float(np.percentile(sm, 2))) + 0.5 * (base_c - mean)

            mask_w = drop_small_blobs(_anomaly_mask(sm, base_w, offset), min_px)
            frac_w = float(mask_w.sum()) / max(mask_w.size, 1)
            t_w = float(crop[mask_w].mean()) if mask_w.any() else mean
            if base_c is not None:
                mask_c = drop_small_blobs(_cold_mask(sm, base_c, offset), min_px)
                frac_c = float(mask_c.sum()) / max(mask_c.size, 1)
                t_c = float(crop[mask_c].mean()) if mask_c.any() else mean
            else:
                mask_c, frac_c, t_c = np.zeros_like(mask_w), 0.0, mean

            if live:
                # A localized anomaly barely moves the region mean: also rate it by how far its own
                # pixels sit from the baseline (>= the offset by construction).
                if frac_w >= min_frac and mask_w.any():
                    dw = max(dw, t_w - base_w)
                if base_c is not None and frac_c >= min_frac and mask_c.any():
                    dc = max(dc, base_c - t_c)

            sv = self._ema(name, now, {"dw": dw, "dc": dc, "fw": frac_w, "fc": frac_c, "tw": t_w, "tc": t_c})
            cold = base_c is not None and sv["dc"] > sv["dw"]
            direction = "cold" if cold else "warm"
            delta = float(sv["dc"] if cold else sv["dw"])
            anomaly_frac = float(sv["fc"] if cold else sv["fw"])
            t_surface = float(sv["tc"] if cold else sv["tw"])
            mask = mask_c if cold else mask_w
            baseline = base_c if cold else base_w
            anomaly_px = int(round(anomaly_frac * mask.size))
            compactness, aspect, _ = shape_features(mask)
            if is_ref:
                t_ref_surface = float(reference_c)
            else:
                t_ref_surface = float(baseline) if opening else float(reference_c)
            edge_ov = edge_overlap_fraction(mask)
            defect, reasons = classify_defect(
                element=element,
                delta_c=delta,
                compactness=compactness,
                aspect_ratio=aspect,
                edge_overlap=edge_ov,
                fluctuation_c=fluct,
                anomaly_frac=anomaly_frac,
                normal_dt=float(th["normal_dt"]),
                min_frac=min_frac,
                direction=direction,
            )
            sev = severity_of(delta, anomaly_frac, th)
            if defect == "normal":
                sev = "ok"
            else:
                reasons.append(
                    f"{direction} anomaly {delta:.1f} °C vs {'opening baseline' if opening else 'reference wall'}"
                    f" over {anomaly_frac:.0%} of the region"
                )
            if live and defect != "normal" and not cold:
                reasons.append("warm area seen from inside: likely a heat source, not a heat loss (no cost counted)")
            u = float(self.u_values.get(element, 0.35))
            area_m2 = anomaly_px * px_area
            # Heat-loss estimate uses indoor-outdoor ΔT when known, else the anomaly's surface ΔT.
            if live:
                dt_q = max(delta, 0.0)
            else:
                dt_q = (indoor_c - ambient_c) if indoor_c is not None else max(delta, st["mean_c"] - ambient_c)
            q = estimate_q(u, area_m2, dt_q)
            act = action_for(defect)
            fix, verify, hud = act["fix"], act["verify"], act["hud"]
            if cold:
                verify = verify.replace("warm", "cold")
            priority = SEVERITY_RANK[sev] * 10.0 + q + max(delta, 0.0)
            if defect != "normal":
                hud = f"{hud} at {name.replace('_', ' ')} · Severity {sev.upper()}"
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
                    fix=fix,
                    verify=verify,
                    hud=hud,
                    reasons=reasons,
                    t_surface_c=t_surface,
                    t_ref_surface_c=t_ref_surface,
                    direction=direction,
                    live=live,
                )
            )
        findings.sort(key=lambda f: f.priority, reverse=True)
        return findings

    def coldest_area_finding(self, temps: np.ndarray, indoor_c: float | None = None) -> RegionFinding | None:
        """Live interior scan: the coldest area in the frame as a cold-spot leak candidate.

        Takes the connected blob around the coldest (smoothed) pixel, i.e. pixels within a band
        above the minimum, and reports it if it is colder than the scene by more than the
        sensitivity's normal_dt. Seen from inside, cold = heat escaping (leak / cold bridge).
        """
        th = self.thresholds
        t = np.nan_to_num(np.asarray(temps, dtype=np.float32), nan=float(np.nanmedian(temps)))
        sm = self._blur(t)
        scene = float(self.scene_c) if self.scene_c is not None else float(np.median(sm))
        t_min = float(sm.min())
        deficit_max = scene - t_min
        if deficit_max < float(th["normal_dt"]):
            return None
        band = max(0.3, 0.4 * deficit_max)
        mask = (sm <= t_min + band).astype(np.uint8)
        my, mx = np.unravel_index(int(np.argmin(sm)), sm.shape)
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        blob = labels == labels[my, mx]
        ys, xs = np.nonzero(blob)
        if ys.size < 4:
            return None
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        pad = 3
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(t.shape[1] - 1, x1 + pad), min(t.shape[0] - 1, y1 + pad)
        crop_mask = blob[y0 : y1 + 1, x0 : x1 + 1]
        vals = t[blob]
        t_surface = float(vals.mean())
        delta = scene - t_surface
        if delta < float(th["normal_dt"]):
            return None
        frac = float(ys.size) / float(t.size)
        compactness, aspect, _ = shape_features(crop_mask.astype(np.uint8))
        edge_ov = edge_overlap_fraction(blob.astype(np.uint8))
        defect, reasons = classify_defect(
            element="wall", delta_c=delta, compactness=compactness, aspect_ratio=aspect,
            edge_overlap=edge_ov, fluctuation_c=None, anomaly_frac=max(frac, float(th.get("min_frac", 0.02))),
            normal_dt=float(th["normal_dt"]), min_frac=0.0, direction="cold",
        )
        if defect == "normal":
            defect = "thermal_bridge"
        sev = severity_of(delta, frac, th)
        if sev == "ok":
            sev = "low"
        reasons.append(f"coldest area in view: {delta:.1f} °C below the room surfaces ({scene:.1f} °C)")
        area_m2 = ys.size * self.meters_per_pixel**2
        u = float(self.u_values.get("wall", 0.35))
        q = estimate_q(u, area_m2, delta)
        act = action_for(defect)
        crop = t[y0 : y1 + 1, x0 : x1 + 1]
        return RegionFinding(
            name="coldest_area", element="wall", x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1,
            mean_c=float(crop.mean()), max_c=float(crop.max()), min_c=float(crop.min()), std_c=float(crop.std()),
            ambient_c=scene, reference_c=scene, delta_c=float(delta), indoor_c=indoor_c,
            anomaly_pixels=int(ys.size), anomaly_area_m2=float(area_m2), anomaly_frac=frac,
            compactness=float(compactness), aspect_ratio=float(aspect), edge_overlap=float(edge_ov),
            fluctuation_c=None, normal=False, defect=defect, severity=sev, q_watts=float(q), u_value=u,
            priority=float(SEVERITY_RANK[sev] * 10.0 + q + delta), fix=act["fix"],
            verify=act["verify"].replace("warm", "cold"),
            hud=f"{act['hud']} at coldest area · Severity {sev.upper()}", reasons=reasons,
            t_surface_c=t_surface, t_ref_surface_c=scene, direction="cold", live=True,
        )

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
