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
_POPPINS_BOLD = _FONTS_DIR / "Poppins-Bold.ttf"
_POPPINS_SEMI = _FONTS_DIR / "Poppins-SemiBold.ttf"
_INTER = _FONTS_DIR / "Inter-Variable.ttf"
_FALLBACK_REG = _FONTS_DIR / "DejaVuSans.ttf"
_FALLBACK_BOLD = _FONTS_DIR / "DejaVuSans-Bold.ttf"
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

# Deck palette (RGB): graphite plate, ironbow amber/orange, severity chips.
PLATE = (14, 14, 17)
AMBER = (255, 200, 87)  # deck amber F9B233, a touch lighter for small type on dark plates
SEV_RGB = {
    "high": (236, 90, 38),
    "medium": (249, 178, 51),
    "low": (155, 107, 208),
    "ok": (61, 207, 142),
}
SEV_TEXT = {"high": (255, 255, 255), "medium": (20, 14, 8), "low": (255, 255, 255), "ok": (8, 24, 16)}

_DEFECT_SHORT = {
    "insulation_gap": "INSULATION GAP",
    "thermal_bridge": "THERMAL BRIDGE",
    "air_leak": "AIR LEAK",
}
_DEFECT_TITLE = {
    "insulation_gap": "Insulation gap",
    "thermal_bridge": "Thermal bridge",
    "air_leak": "Air leak",
}


def _font(size: int, bold: bool = False, kind: str = "inter") -> ImageFont.FreeTypeFont:
    """kind: 'poppins' (display, Bold/SemiBold) or 'inter' (text, 600/500). Falls back to DejaVu."""
    key = (f"{kind}{int(bold)}", int(size))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font: ImageFont.FreeTypeFont | None = None
    try:
        if kind == "poppins" and _POPPINS_BOLD.is_file():
            font = ImageFont.truetype(str(_POPPINS_BOLD if bold else _POPPINS_SEMI), int(size))
        elif kind == "inter" and _INTER.is_file():
            font = ImageFont.truetype(str(_INTER), int(size))
            try:
                font.set_variation_by_axes([14, 650 if bold else 500])
            except Exception:  # FreeType without variation support: default weight is fine
                pass
    except OSError:
        font = None
    if font is None:
        path = _FALLBACK_BOLD if bold and _FALLBACK_BOLD.is_file() else _FALLBACK_REG
        font = ImageFont.truetype(str(path), int(size))
    _FONT_CACHE[key] = font
    return font


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


def _sev_key(finding_get) -> str:
    sev = str(finding_get("severity") or "ok").lower()
    return sev if sev in SEV_RGB else "ok"


def _brackets(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color, width: int, arm: int) -> None:
    """Corner brackets, like the focus marks on the pitch-deck thermograms."""
    x, y, w, h = box
    x2, y2 = x + max(w, 2) - 1, y + max(h, 2) - 1
    arm = max(3, min(arm, (x2 - x) // 2, (y2 - y) // 2))
    corners = ((x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1))
    for (cx, cy, dx, dy) in corners:  # dark halo first so the marks read on hot (orange) areas
        draw.line((cx - dx, cy, cx + dx * (arm + 1), cy), fill=(9, 6, 20, 170), width=width + 2)
        draw.line((cx, cy - dy, cx, cy + dy * (arm + 1)), fill=(9, 6, 20, 170), width=width + 2)
    for (cx, cy, dx, dy) in corners:
        draw.line((cx, cy, cx + dx * arm, cy), fill=color, width=width)
        draw.line((cx, cy, cx, cy + dy * arm), fill=color, width=width)


HOT_RGB = (255, 70, 50)
COLD_RGB = (70, 190, 255)


def extreme_points(temps: np.ndarray) -> dict[str, tuple[int, int, float]]:
    """(x, y, °C) of the hottest and coldest pixel in the frame."""
    t = np.nan_to_num(np.asarray(temps, dtype=np.float32), nan=float(np.nanmedian(temps)))
    iy, ix = np.unravel_index(int(np.argmax(t)), t.shape)
    jy, jx = np.unravel_index(int(np.argmin(t)), t.shape)
    return {"max": (int(ix), int(iy), float(t[iy, ix])), "min": (int(jx), int(jy), float(t[jy, jx]))}


def _draw_extreme_markers(draw, temps, scale, cw, ch, font, occupied, top_bar) -> None:
    pts = extreme_points(temps)
    r = int(round(4 * scale))
    for key, color, word in (("max", HOT_RGB, "MAX"), ("min", COLD_RGB, "MIN")):
        x, y, t = pts[key]
        px, py = int((x + 0.5) * scale), int((y + 0.5) * scale)
        # crosshair with dark halo so it reads on any colour
        for col, wdt in (((9, 6, 20, 190), max(3, scale + 2)), (color + (255,), max(2, scale))):
            draw.ellipse((px - r, py - r, px + r, py + r), outline=col, width=wdt)
            draw.line((px - 2 * r, py, px - r // 2, py), fill=col, width=wdt)
            draw.line((px + r // 2, py, px + 2 * r, py), fill=col, width=wdt)
            draw.line((px, py - 2 * r, px, py - r // 2), fill=col, width=wdt)
            draw.line((px, py + r // 2, px, py + 2 * r), fill=col, width=wdt)
        text = f"{word} {t:.1f} °C"
        tw = int(draw.textlength(text, font=font))
        th = int(font.size)
        padx, pady = int(round(1.8 * scale)), int(round(1.0 * scale))
        bw, bh = tw + 2 * padx, th + 2 * pady
        bx = px + 2 * r + 2
        if bx + bw > cw - 2:
            bx = px - 2 * r - 2 - bw
        by = py - bh // 2
        by = max(top_bar if bx < cw // 3 else 2, min(by, ch - bh - 2))
        bx = max(2, bx)
        draw.rounded_rectangle((bx, by, bx + bw, by + bh), radius=bh // 2, fill=PLATE + (225,))
        draw.rounded_rectangle((bx, by, bx + bw, by + bh), radius=bh // 2, outline=color + (230,), width=max(1, scale // 2))
        draw.text((bx + padx, by + bh / 2), text, font=font, fill=color + (255,), anchor="lm")
        occupied.append((bx, by, bw, bh))


def draw_findings(
    visual: np.ndarray,
    findings: Iterable[Any],
    temps: np.ndarray | None = None,
    total: dict[str, Any] | None = None,
    scale: int = 3,
) -> np.ndarray:
    """Paint focus brackets, numbered savings labels and a total strip. Upscales for readable type.

    Defects are numbered 1..n in list order (already priority-sorted) so the dashboard cards can
    carry the same numbers.
    """
    scale = max(1, int(scale))
    h, w = visual.shape[:2]
    if scale != 1:
        canvas = cv2.resize(visual, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    else:
        canvas = visual.copy()
    ch, cw = canvas.shape[:2]

    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    f_tag = _font(max(8, int(round(4.1 * scale))), bold=True, kind="inter")
    f_title = _font(max(9, int(round(6.0 * scale))), bold=True, kind="poppins")
    f_chip = _font(max(8, int(round(4.1 * scale))), bold=True, kind="inter")
    f_save = _font(max(9, int(round(5.5 * scale))), bold=True, kind="inter")
    f_num = _font(max(8, int(round(5.1 * scale))), bold=True, kind="poppins")
    f_banner = _font(max(10, int(round(6.3 * scale))), bold=True, kind="poppins")

    items = list(findings)
    lw = max(2, scale)

    # --- focus marks -------------------------------------------------------------------------
    for f in items:
        get = f.get if isinstance(f, dict) else lambda k, default=None, _f=f: getattr(_f, k, default)
        normal = bool(get("normal") or get("defect") == "normal")
        x = int(get("x") or 0) * scale
        y = int(get("y") or 0) * scale
        fw = int(get("w") or 0) * scale
        fh = int(get("h") or 0) * scale
        if normal:
            draw.rectangle((x, y, x + max(fw, 1) - 1, y + max(fh, 1) - 1), outline=(255, 255, 255, 70), width=1)
        else:
            col = SEV_RGB[_sev_key(get)] + (255,)
            draw.rectangle((x, y, x + max(fw, 1) - 1, y + max(fh, 1) - 1), outline=col[:3] + (120,), width=1)
            _brackets(draw, (x, y, fw, fh), col, lw + 1, int(round(7 * scale)))

    occupied: list[tuple[int, int, int, int]] = []
    margin = int(round(3 * scale))

    # --- top-left tag (like the deck's THERMAL · SNUG AI chip) --------------------------------
    tag = "THERMAL · SNUG AI"
    if temps is not None:
        tag += f" · {float(np.min(temps)):.0f}–{float(np.max(temps)):.0f} °C"
    tw = int(draw.textlength(tag, font=f_tag))
    th = int(f_tag.size * 1.0)
    tp_x, tp_y = int(round(2.2 * scale)), int(round(1.4 * scale))
    tag_box = (margin, margin, tw + 2 * tp_x, th + 2 * tp_y)
    draw.rounded_rectangle(
        (tag_box[0], tag_box[1], tag_box[0] + tag_box[2], tag_box[1] + tag_box[3]),
        radius=tag_box[3] // 2,
        fill=PLATE + (225,),
    )
    draw.text((tag_box[0] + tp_x, tag_box[1] + tp_y - 1), tag, font=f_tag, fill=(240, 240, 244, 255))
    occupied.append(tag_box)
    top_bar = tag_box[1] + tag_box[3] + margin

    # --- bottom total strip -------------------------------------------------------------------
    banner = overlay_banner_text(total)
    bottom_limit = ch
    if banner:
        bw = int(draw.textlength(banner, font=f_banner))
        bh = int(f_banner.size)
        bpx, bpy = int(round(5 * scale)), int(round(2 * scale))
        pw, ph = bw + 2 * bpx, bh + 2 * bpy
        bx = (cw - pw) // 2
        by = ch - ph - margin
        has_sav = bool(total and (float(total.get("cost_eur_high") or 0) > 0))
        draw.rounded_rectangle((bx, by, bx + pw, by + ph), radius=ph // 2, fill=PLATE + (235,))
        draw.rounded_rectangle((bx, by, bx + pw, by + ph), radius=ph // 2, outline=(AMBER if has_sav else (150, 150, 160)) + (200,), width=1)
        draw.text((bx + bpx, by + bpy - 2), banner, font=f_banner, fill=(AMBER if has_sav else (210, 210, 218)) + (255,))
        occupied.append((bx, by, pw, ph))
        bottom_limit = by

    # --- hottest / coldest point markers ------------------------------------------------------
    if temps is not None and temps.size:
        _draw_extreme_markers(draw, temps, scale, cw, ch, f_tag, occupied, top_bar)

    # --- labels -------------------------------------------------------------------------------
    pad = int(round(2.4 * scale))
    bar_w = max(2, int(round(1.2 * scale)))
    n = 0
    for f in items:
        get = f.get if isinstance(f, dict) else lambda k, default=None, _f=f: getattr(_f, k, default)
        lines = defect_hud_lines(f)
        if not lines:
            continue
        n += 1
        defect = str(get("defect") or "normal")
        sev = _sev_key(get)
        sev_rgb = SEV_RGB[sev]
        title = _DEFECT_TITLE.get(defect, defect.replace("_", " ").capitalize())
        chip = sev.upper()
        save = lines[1] if len(lines) > 1 else ""
        num = str(n)

        circ = int(round(f_num.size * 1.35))
        chip_pad = int(round(1.8 * scale))
        chip_w = int(draw.textlength(chip, font=f_chip)) + 2 * chip_pad
        chip_h = int(round(f_chip.size * 1.45))
        title_w = int(draw.textlength(title, font=f_title))
        row1_h = max(circ, chip_h, int(f_title.size * 1.2))
        gap = int(round(1.8 * scale))
        row1_w = circ + gap + title_w + gap + chip_w
        save_w = int(draw.textlength(save, font=f_save)) if save else 0
        save_h = int(f_save.size * 1.2) if save else 0
        block_w = bar_w + pad + max(row1_w, save_w) + pad
        block_h = pad + row1_h + (int(round(1.2 * scale)) + save_h if save else 0) + pad

        x = int(get("x") or 0) * scale
        y = int(get("y") or 0) * scale
        fw = int(get("w") or 0) * scale
        fh = int(get("h") or 0) * scale
        tx, ty = _place_block((x, y, fw, fh), block_w, block_h, cw, bottom_limit, top_bar, occupied, scale)
        occupied.append((tx, ty, block_w, block_h))

        draw.rounded_rectangle((tx, ty, tx + block_w, ty + block_h), radius=max(3, int(1.6 * scale)), fill=PLATE + (232,))
        draw.rounded_rectangle((tx, ty, tx + bar_w + 2, ty + block_h), radius=max(2, scale), fill=sev_rgb + (255,))
        draw.rectangle((tx + bar_w, ty, tx + bar_w + 3, ty + block_h), fill=PLATE + (232,))
        cx0 = tx + bar_w + pad
        cy0 = ty + pad
        # number badge
        draw.ellipse((cx0, cy0 + (row1_h - circ) // 2, cx0 + circ, cy0 + (row1_h - circ) // 2 + circ), fill=sev_rgb + (255,))
        nw = draw.textlength(num, font=f_num)
        draw.text(
            (cx0 + (circ - nw) / 2, cy0 + (row1_h - circ) // 2 + circ / 2),
            num, font=f_num, fill=SEV_TEXT[sev] + (255,), anchor="lm",
        )
        # title
        draw.text((cx0 + circ + gap, cy0 + row1_h / 2), title, font=f_title, fill=(255, 255, 255, 255), anchor="lm")
        # severity chip
        chx = cx0 + circ + gap + title_w + gap
        chy = cy0 + (row1_h - chip_h) // 2
        draw.rounded_rectangle((chx, chy, chx + chip_w, chy + chip_h), radius=chip_h // 2, fill=sev_rgb + (255,))
        draw.text((chx + chip_w / 2, chy + chip_h / 2), chip, font=f_chip, fill=SEV_TEXT[sev] + (255,), anchor="mm")
        if save:
            draw.text((cx0, cy0 + row1_h + int(round(1.2 * scale)) + save_h / 2), save, font=f_save, fill=AMBER + (255,), anchor="lm")

    out = Image.alpha_composite(img, layer).convert("RGB")
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
