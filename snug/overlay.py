"""Draw HUD boxes, per-defect savings and a total banner onto an ironbow frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from snug.cost import format_range_eur

# BGR
COLORS = {
    "normal": (110, 190, 80),
    "ok": (110, 190, 80),
    "low": (80, 200, 230),
    "medium": (40, 160, 255),
    "high": (40, 60, 255),
    "insulation_gap": (0, 200, 255),
    "thermal_bridge": (0, 140, 255),
    "air_leak": (40, 40, 255),
}

_FONTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
_FONT_REG = _FONTS_DIR / "DejaVuSans.ttf"
_FONT_BOLD = _FONTS_DIR / "DejaVuSans-Bold.ttf"
_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}

_DEFECT_SHORT = {
    "insulation_gap": "INSULATION GAP",
    "thermal_bridge": "THERMAL BRIDGE",
    "air_leak": "AIR LEAK",
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (int(size), bool(bold))
    if key not in _FONT_CACHE:
        path = _FONT_BOLD if bold and _FONT_BOLD.is_file() else _FONT_REG
        _FONT_CACHE[key] = ImageFont.truetype(str(path), int(size))
    return _FONT_CACHE[key]


def overlay_save_caption(finding: Any) -> str:
    """Second line drawn next to a defect box, e.g. 'Fix → save ~€9–13/yr (Estimate)'."""
    is_dict = isinstance(finding, dict)
    get = finding.get if is_dict else lambda k, default=None: getattr(finding, k, default)
    if bool(get("normal") or get("defect") == "normal"):
        return ""
    cost = get("cost") or {}
    low = float((cost or {}).get("cost_eur_low") or 0.0)
    high = float((cost or {}).get("cost_eur_high") or 0.0)
    if high <= 0 and low <= 0:
        short = get("cost_short") or (cost or {}).get("short") or ""
        return str(short) if short else ""
    return f"Fix → save ~{format_range_eur(low, high, suffix='/yr')} (Estimate)"


def overlay_banner_text(total: dict[str, Any] | None) -> str:
    """Prominent total drawn on the thermogram itself."""
    if not total:
        return ""
    if total.get("banner"):
        return str(total["banner"])
    low = float(total.get("cost_eur_low") or 0.0)
    high = float(total.get("cost_eur_high") or 0.0)
    if high <= 0 and low <= 0:
        return "No leaks found"
    return f"Fixing these saves ~{format_range_eur(low, high, suffix='/ year')}"


def defect_hud_lines(finding: Any) -> list[str]:
    is_dict = isinstance(finding, dict)
    get = finding.get if is_dict else lambda k, default=None: getattr(finding, k, default)
    if bool(get("normal") or get("defect") == "normal"):
        return []
    defect = str(get("defect") or "")
    short = _DEFECT_SHORT.get(defect, defect.upper())
    sev = str(get("severity") or "").upper()
    line1 = f"{short} {sev}".strip()
    line2 = overlay_save_caption(finding)
    return [line for line in (line1, line2) if line]


def _bgr_to_rgb(color) -> tuple[int, int, int]:
    b, g, r = (int(color[0]), int(color[1]), int(color[2]))
    return r, g, b


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return int(box[2] - box[0]), int(box[3] - box[1])


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    stroke: int = 2,
    stroke_fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)



def _clamp(v: int, lo: int, hi: int) -> int:
    return int(min(max(lo, v), hi))


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 2) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + pad <= bx or bx + bw + pad <= ax or ay + ah + pad <= by or by + bh + pad <= ay)


def _place_block(
    box: tuple[int, int, int, int],
    block_w: int,
    block_h: int,
    canvas_w: int,
    canvas_h: int,
    top_bar: int,
    occupied: list[tuple[int, int, int, int]],
    scale: int,
) -> tuple[int, int]:
    x, y, fw, fh = box
    gap = 3 * scale
    candidates = [
        (x, y - block_h - gap),  # above
        (x, y + fh + gap),  # below
        (x + 3 * scale, y + 3 * scale),  # inside top-left
        (x + fw + gap, y),  # right
        (x - block_w - gap, y),  # left
        (x + fw - block_w, y - block_h - gap),  # above, right-aligned
        (x + fw + gap, y + max(0, fh - block_h)),  # right, bottom-aligned
        (x - block_w - gap, y + max(0, fh - block_h)),  # left, bottom-aligned
        (x, y + max(0, (fh - block_h) // 2)),  # inside vertically centered
    ]
    for tx, ty in candidates:
        tx = _clamp(tx, 2, max(2, canvas_w - block_w - 2))
        ty = _clamp(ty, top_bar, max(top_bar, canvas_h - block_h - 2))
        rect = (tx, ty, block_w, block_h)
        if any(_rects_overlap(rect, other, pad=2 * scale) for other in occupied):
            continue
        if ty < top_bar:
            continue
        return tx, ty
    # Last resort: clamp above the banner strip.
    return _clamp(x, 2, max(2, canvas_w - block_w - 2)), _clamp(y, top_bar, max(top_bar, canvas_h - block_h - 2))


def draw_findings(
    visual: np.ndarray,
    findings: Iterable[Any],
    temps: np.ndarray | None = None,
    total: dict[str, Any] | None = None,
    scale: int = 3,
) -> np.ndarray:
    """Paint boxes, savings captions and a total banner. Upscales for readable type."""
    scale = max(1, int(scale))
    h, w = visual.shape[:2]
    if scale != 1:
        canvas = cv2.resize(visual, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    else:
        canvas = visual.copy()
    ch, cw = canvas.shape[:2]

    banner = overlay_banner_text(total)
    banner_h = int(round(22 * scale)) if banner else 0
    top_bar = int(round(14 * scale)) + (banner_h if banner else 0)

    items = list(findings)
    for f in items:
        get = f.get if isinstance(f, dict) else lambda k, default=None, _f=f: getattr(_f, k, default)
        defect = str(get("defect") or "normal")
        color = COLORS.get(defect, COLORS["ok"])
        if str(get("severity") or "ok") == "high":
            color = COLORS["high"]
        x = int(get("x") or 0) * scale
        y = int(get("y") or 0) * scale
        fw = int(get("w") or 0) * scale
        fh = int(get("h") or 0) * scale
        thickness = 2 if str(get("severity") or "ok") == "high" else 1
        thickness = max(thickness, 1 if scale == 1 else 2)
        cv2.rectangle(canvas, (x, y), (x + max(fw, 1) - 1, y + max(fh, 1) - 1), color, thickness)

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font_temp = _font(max(10, int(round(7.0 * scale))), bold=True)
    font_label = _font(max(9, int(round(5.4 * scale))), bold=True)
    font_save = _font(max(9, int(round(5.0 * scale))), bold=True)
    font_banner = _font(max(10, int(round(6.2 * scale))), bold=True)

    if temps is not None:
        readout = f"{float(np.min(temps)):.1f}  {float(np.max(temps)):.1f} °C"
        _draw_text(draw, (6 * scale, 2 * scale), readout, font_temp, (240, 240, 240), stroke=max(1, scale))

    occupied: list[tuple[int, int, int, int]] = []
    if banner:
        y0 = int(round(12 * scale)) if temps is not None else int(round(2 * scale))
        pad_x, pad_y = 6 * scale, max(2, scale)
        tw, th = _text_size(draw, banner, font_banner)
        bar_top = y0
        bar_bot = min(ch, y0 + th + 2 * pad_y)
        draw.rectangle((0, bar_top, cw, bar_bot), fill=(20, 10, 8))
        tx = max(pad_x, (cw - tw) // 2)
        _draw_text(draw, (tx, bar_top + pad_y - 1), banner, font_banner, (255, 213, 106), stroke=max(1, scale // 2 or 1))
        occupied.append((0, bar_top, cw, bar_bot - bar_top))
        top_bar = bar_bot + 2

    pad = max(2, int(round(2.2 * scale)))
    for f in items:
        get = f.get if isinstance(f, dict) else lambda k, default=None, _f=f: getattr(_f, k, default)
        lines = defect_hud_lines(f)
        if not lines:
            continue
        defect = str(get("defect") or "normal")
        color = COLORS.get(defect, COLORS["ok"])
        if str(get("severity") or "ok") == "high":
            color = COLORS["high"]
        fill = _bgr_to_rgb(color)
        x = int(get("x") or 0) * scale
        y = int(get("y") or 0) * scale
        fw = int(get("w") or 0) * scale
        fh = int(get("h") or 0) * scale

        sizes = [_text_size(draw, line, font_label if i == 0 else font_save) for i, line in enumerate(lines)]
        line_gap = max(1, scale)
        text_h = sum(h for _, h in sizes) + line_gap * (len(lines) - 1)
        text_w = max((w for w, _ in sizes), default=0)
        block_w = text_w + 2 * pad
        block_h = text_h + 2 * pad
        tx, ty = _place_block((x, y, fw, fh), block_w, block_h, cw, ch, top_bar, occupied, scale)
        occupied.append((tx, ty, block_w, block_h))
        # Dark plate so type stays readable on any ironbow colour.
        draw.rounded_rectangle(
            (tx, ty, tx + block_w, ty + block_h),
            radius=max(2, scale),
            fill=(12, 8, 10, 255),
            outline=(40, 28, 24),
        )
        cy = ty + pad
        for i, line in enumerate(lines):
            font = font_label if i == 0 else font_save
            col = (255, 213, 106) if i > 0 else fill
            _draw_text(draw, (tx + pad, cy), line, font, col, stroke=max(1, scale))
            cy += sizes[i][1] + line_gap

    canvas = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return canvas
