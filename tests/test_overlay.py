"""Overlay captions and the savings banner drawn onto the thermogram."""

from __future__ import annotations

import numpy as np

from snug.cost import CostSettings, attach_costs, format_range_eur
from snug.overlay import defect_hud_lines, draw_findings, overlay_banner_text, overlay_save_caption


class _F:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.cost = None
        self.cost_label = None
        self.cost_short = None


def _defect(**kw):
    defaults = dict(
        name="gap",
        element="wall",
        x=40,
        y=40,
        w=60,
        h=50,
        anomaly_frac=0.20,
        t_surface_c=16.0,
        t_ref_surface_c=19.0,
        mean_c=16.0,
        reference_c=19.0,
        defect="insulation_gap",
        severity="high",
        normal=False,
    )
    defaults.update(kw)
    return _F(**defaults)


def test_overlay_save_caption_matches_cost_range():
    f = _defect()
    attach_costs([f], CostSettings(locality="vaasa", heating_type="heat_pump"), t_in=21.0, t_out=-5.0)
    caption = overlay_save_caption(f)
    expected = f"Fix → save ~{format_range_eur(f.cost['cost_eur_low'], f.cost['cost_eur_high'], suffix='/yr')} (Estimate)"
    assert caption == expected
    assert "€" in caption and "yr" in caption
    lines = defect_hud_lines(f)
    assert lines[0] == "INSULATION GAP HIGH"
    assert lines[1] == caption


def test_overlay_skips_normal_regions():
    f = _defect(defect="normal", normal=True, severity="ok", anomaly_frac=0.0, t_surface_c=19.0)
    attach_costs([f], CostSettings(), t_in=21.0, t_out=-5.0)
    assert overlay_save_caption(f) == ""
    assert defect_hud_lines(f) == []


def test_banner_no_leaks_and_with_savings():
    assert overlay_banner_text({"cost_eur_low": 0, "cost_eur_high": 0}) == "No leaks found"
    assert overlay_banner_text(None) == ""
    f = _defect()
    total = attach_costs([f], CostSettings(locality="vaasa", heating_type="heat_pump"), t_in=21.0, t_out=-5.0)
    banner = overlay_banner_text(total)
    assert banner.startswith("Fixing these saves ~")
    assert "€" in banner and "/ year" in banner
    assert banner == total["banner"]


def test_draw_findings_includes_savings_text_and_banner():
    f = _defect()
    total = attach_costs([f], CostSettings(locality="vaasa", heating_type="heat_pump"), t_in=21.0, t_out=-5.0)
    frame = np.zeros((192, 256, 3), dtype=np.uint8)
    frame[:] = (40, 20, 80)
    temps = np.full((192, 256), 18.0, dtype=np.float32)
    temps[40:90, 40:100] = 16.0
    out = draw_findings(frame, [f], temps, total=total, scale=3)
    assert out.shape[0] == 192 * 3 and out.shape[1] == 256 * 3
    # The gold savings colour (BGR ~ 106, 213, 255) must appear for the caption/banner.
    gold = (out[:, :, 0] > 80) & (out[:, :, 1] > 180) & (out[:, :, 2] > 230)
    assert int(gold.sum()) > 50
    # And the frame should not be a uniform upscaled copy — boxes or type have been painted.
    assert out.std() > frame.std()
