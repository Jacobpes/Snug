"""Realistic model-house thermal simulator (256x192, exterior view).

A heat source inside warms the house. From the outside, well-insulated
surfaces stay close to ambient; defects appear warmer:

- insulation gap: compact warm blob on a wall
- thermal bridge: elongated warm line along a joint/corner/sill
- air leak: streaky, time-varying plume at a door or window edge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from snug.colormap import temps_to_ironbow

SENSOR_W = 256
SENSOR_H = 192

# Semantic ids used as ground truth for the detector and tests.
CLASS_NAMES = ("background", "wall", "window", "door", "roof", "corner")
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class Rect:
    name: str
    element: str  # wall, window, door, roof, corner, background
    x: int
    y: int
    w: int
    h: int
    u_factor: float = 0.35  # W/m²K, used only to scale heat bleed in the sim

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def as_roi(self) -> dict[str, Any]:
        return {"name": self.name, "element": self.element, "x": self.x, "y": self.y, "w": self.w, "h": self.h}


def default_layout() -> list[Rect]:
    """Front facade of the demo house in pixel coordinates."""
    return [
        Rect("roof", "roof", 28, 8, 200, 24, u_factor=0.30),
        Rect("left_corner", "corner", 28, 32, 16, 140, u_factor=0.90),
        Rect("insulated_wall", "wall", 44, 32, 56, 140, u_factor=0.28),
        Rect("window", "window", 100, 44, 48, 72, u_factor=2.4),
        Rect("uninsulated_wall", "wall", 148, 32, 52, 80, u_factor=1.55),
        Rect("door", "door", 160, 112, 40, 60, u_factor=1.8),
        Rect("right_corner", "corner", 212, 32, 16, 140, u_factor=0.90),
    ]


@dataclass
class HouseSimConfig:
    outdoor_c: float = 18.0
    indoor_max_c: float = 36.0
    heat_delay_s: float = 6.0
    heat_ramp_s: float = 14.0
    noise_c: float = 0.18
    insulation_gap: bool = True
    thermal_bridge: bool = True
    air_leak: bool = True
    seed: int | None = 0
    layout: list[Rect] = field(default_factory=default_layout)
    # jitter used when generating training frames
    jitter: bool = False


def indoor_temperature(elapsed_s: float, cfg: HouseSimConfig) -> float:
    """Indoor air ramps from outdoor toward indoor_max after heat_delay_s."""
    if elapsed_s <= cfg.heat_delay_s:
        return cfg.outdoor_c
    t = min(1.0, (elapsed_s - cfg.heat_delay_s) / max(cfg.heat_ramp_s, 1e-3))
    # Smooth ease-in
    t = t * t * (3 - 2 * t)
    return cfg.outdoor_c + t * (cfg.indoor_max_c - cfg.outdoor_c)


def _rect_mask(shape: tuple[int, int], r: Rect) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    y0, y1 = max(0, r.y), min(shape[0], r.y + r.h)
    x0, x1 = max(0, r.x), min(shape[1], r.x + r.w)
    m[y0:y1, x0:x1] = True
    return m


def _gaussian_blob(shape: tuple[int, int], cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.exp(-((xx - cx) ** 2) / (2 * sx**2) - ((yy - cy) ** 2) / (2 * sy**2)).astype(np.float32)


def _line_kernel(shape: tuple[int, int], x0: float, y0: float, x1: float, y1: float, width: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    dx, dy = x1 - x0, y1 - y0
    length = max(np.hypot(dx, dy), 1e-3)
    t = ((xx - x0) * dx + (yy - y0) * dy) / (length**2)
    t_clamped = np.clip(t, 0.0, 1.0)
    px = x0 + t_clamped * dx
    py = y0 + t_clamped * dy
    dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    return np.exp(-(dist**2) / (2 * (width**2))).astype(np.float32)


def semantic_masks(layout: list[Rect], shape: tuple[int, int] = (SENSOR_H, SENSOR_W)) -> np.ndarray:
    """Integer class map (H, W) with CLASS_ID labels."""
    labels = np.zeros(shape, dtype=np.uint8)
    for r in layout:
        labels[_rect_mask(shape, r)] = CLASS_ID[r.element]
    return labels


def _jitter_layout(layout: list[Rect], rng: np.random.Generator) -> list[Rect]:
    out = []
    dx = int(rng.integers(-4, 5))
    dy = int(rng.integers(-3, 4))
    for r in layout:
        out.append(
            Rect(
                r.name,
                r.element,
                r.x + dx + int(rng.integers(-2, 3)),
                r.y + dy + int(rng.integers(-2, 3)),
                max(10, r.w + int(rng.integers(-3, 4))),
                max(10, r.h + int(rng.integers(-3, 4))),
                u_factor=r.u_factor,
            )
        )
    return out


def simulate_scene(
    elapsed_s: float,
    cfg: HouseSimConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (ironbow BGR, temps °C, ground-truth meta)."""
    cfg = cfg or HouseSimConfig()
    if rng is None:
        rng = np.random.default_rng(None if cfg.seed is None else (cfg.seed + int(elapsed_s * 25)))

    layout = _jitter_layout(cfg.layout, rng) if cfg.jitter else cfg.layout
    shape = (SENSOR_H, SENSOR_W)
    outdoor = cfg.outdoor_c + float(rng.normal(0, 0.05))
    indoor = indoor_temperature(elapsed_s, cfg)
    dT = max(0.0, indoor - outdoor)

    temps = np.full(shape, outdoor, dtype=np.float32)
    # faint ambient gradient (floor slightly cooler, sky a touch cooler)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    temps += ((xx / SENSOR_W) * 0.15 - (yy / SENSOR_H) * 0.25).astype(np.float32)

    labels = semantic_masks(layout, shape)
    # Bleed through each element proportional to a crude U-factor.
    # Surface excess ≈ k * U * dT, capped so insulation stays "normal".
    for r in layout:
        mask = _rect_mask(shape, r)
        # Empirical scale: insulated wall stays under the 1.5 °C 'normal' ΔT at full heat.
        k = 0.22
        temps[mask] += np.float32(k * r.u_factor * dT)

    defect_maps: dict[str, np.ndarray] = {}

    # --- insulation gap: compact blob on the uninsulated wall ---
    if cfg.insulation_gap:
        wall = next((r for r in layout if r.name == "uninsulated_wall"), None)
        if wall is not None:
            cx = wall.x + wall.w * 0.55
            cy = wall.y + wall.h * 0.45
            blob = _gaussian_blob(shape, cx, cy, sx=wall.w * 0.28, sy=wall.h * 0.28)
            temps += blob * np.float32(0.85 * dT)
            defect_maps["insulation_gap"] = blob

    # --- thermal bridge: vertical line at left corner / wall joint, plus window sill ---
    if cfg.thermal_bridge:
        corner = next((r for r in layout if r.name == "left_corner"), None)
        window = next((r for r in layout if r.name == "window"), None)
        bridge = np.zeros(shape, dtype=np.float32)
        if corner is not None:
            xj = corner.x + corner.w * 0.45
            bridge += _line_kernel(shape, xj, corner.y, xj, corner.y + corner.h, width=2.0)
        if window is not None:
            # sill and jambs as linear bridges
            bridge += 0.7 * _line_kernel(
                shape, window.x, window.y + window.h, window.x + window.w, window.y + window.h, width=1.8
            )
            bridge += 0.55 * _line_kernel(
                shape, window.x, window.y, window.x, window.y + window.h, width=1.6
            )
        temps += bridge * np.float32(0.70 * dT)
        defect_maps["thermal_bridge"] = bridge

    # --- air leak: streaky, temporally fluctuating plume at the door seal and window corner ---
    if cfg.air_leak:
        door = next((r for r in layout if r.name == "door"), None)
        window = next((r for r in layout if r.name == "window"), None)
        leak = np.zeros(shape, dtype=np.float32)
        pulse = 0.65 + 0.35 * np.sin(elapsed_s * 2.4)
        flutter = 0.85 + 0.15 * np.sin(elapsed_s * 7.0)
        if door is not None:
            x_edge = door.x + 1
            # several short, slightly offset streaks (irregular plume)
            for i, (oy, sx) in enumerate(((6, 1.3), (18, 1.6), (32, 1.2), (44, 1.7))):
                phase = np.sin(elapsed_s * 3.1 + i) * 2.0
                leak += _line_kernel(
                    shape,
                    x_edge,
                    door.y + oy,
                    x_edge - 6 - phase,
                    door.y + oy + 8,
                    width=sx,
                )
        if window is not None:
            leak += 0.6 * _line_kernel(
                shape,
                window.x + window.w - 2,
                window.y + 8,
                window.x + window.w + 5,
                window.y + 28,
                width=1.5,
            )
        temps += leak * np.float32(0.80 * dT * pulse * flutter)
        defect_maps["air_leak"] = leak

    temps += rng.normal(0, cfg.noise_c, shape).astype(np.float32)

    visual = temps_to_ironbow(temps, lo=outdoor - 0.5, hi=max(outdoor + 1.0, float(temps.max())))
    # Draw a faint house outline on the visual so the demo reads clearly.
    for r in layout:
        color = {
            "wall": (40, 40, 40),
            "window": (60, 50, 30),
            "door": (50, 40, 30),
            "roof": (35, 35, 40),
            "corner": (45, 45, 45),
        }.get(r.element, (30, 30, 30))
        cv2.rectangle(visual, (r.x, r.y), (r.x + r.w - 1, r.y + r.h - 1), color, 1)

    boxes = [r.as_roi() for r in layout]
    meta = {
        "labels": labels,
        "boxes": boxes,
        "indoor_c": indoor,
        "outdoor_c": outdoor,
        "dT_in_out": dT,
        "defects_enabled": {
            "insulation_gap": cfg.insulation_gap,
            "thermal_bridge": cfg.thermal_bridge,
            "air_leak": cfg.air_leak,
        },
        "defect_maps": defect_maps,
        "layout": layout,
    }
    return visual, temps, meta
