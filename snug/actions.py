"""Action + verification copy for each defect type (demo-house language, English UI)."""

from __future__ import annotations

from typing import Any

# U-values are typical-order estimates (W/m²K), not measured for the model house.
DEFAULT_U = {
    "wall": 0.35,
    "window": 2.8,
    "door": 2.0,
    "roof": 0.30,
    "corner": 0.80,
    "background": 0.0,
}

ACTIONS: dict[str, dict[str, str]] = {
    "normal": {
        "fix": "No action needed",
        "verify": "Optional periodic re-scan from the same viewpoint to confirm the pattern stays stable.",
        "hud": "Normal thermal pattern",
    },
    "insulation_gap": {
        "fix": "Add or repair insulation in the marked wall area",
        "verify": "Re-scan the same wall after the fix. ΔT versus the insulated reference and the warm anomaly area should both drop.",
        "hud": "Insulation gap",
    },
    "thermal_bridge": {
        "fix": "Break the thermal path at the joint (demo: cover the corner/sill with insulation tape)",
        "verify": "Re-scan the joint from the same angle. The linear warm line should fade; ΔT along the edge and anomaly area should drop.",
        "hud": "Thermal bridge",
    },
    "air_leak": {
        "fix": "Replace weatherstrip / seal the gap at the opening",
        "verify": "Re-scan the door or window seal. Streaks should disappear; ΔT and anomaly area should drop.",
        "hud": "Air leak",
    },
}


def action_for(defect: str) -> dict[str, str]:
    return ACTIONS.get(defect, ACTIONS["normal"])


def verification_verdict(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    dt_drop_frac: float = 0.4,
    area_drop_frac: float = 0.4,
) -> dict[str, Any]:
    """Compare a region's before/after ΔT and anomaly area."""
    dt0 = float(before.get("delta_c") or 0.0)
    dt1 = float(after.get("delta_c") or 0.0)
    a0 = float(before.get("anomaly_area_m2") or 0.0)
    a1 = float(after.get("anomaly_area_m2") or 0.0)
    px0 = float(before.get("anomaly_pixels") or 0.0)
    px1 = float(after.get("anomaly_pixels") or 0.0)
    area0, area1 = (a0, a1) if a0 > 0 or a1 > 0 else (px0, px1)

    dt_drop = (dt0 - dt1) / dt0 if dt0 > 1e-3 else 0.0
    area_drop = (area0 - area1) / area0 if area0 > 1e-9 else 0.0
    improved = (dt_drop >= dt_drop_frac) and (area_drop >= area_drop_frac or area0 <= 1e-9)
    if dt1 <= 0.6 and (after.get("defect") == "normal" or dt_drop >= dt_drop_frac):
        improved = True
    if dt1 > dt0 + 0.3 and area1 > area0:
        status = "worse"
    elif improved:
        status = "improved"
    else:
        status = "unchanged"
    return {
        "status": status,
        "delta_c_before": round(dt0, 3),
        "delta_c_after": round(dt1, 3),
        "delta_c_drop_frac": round(dt_drop, 3),
        "anomaly_area_before": area0,
        "anomaly_area_after": area1,
        "area_drop_frac": round(area_drop, 3),
        "passed": status == "improved",
        "note": (
            "ΔT and anomaly area dropped — treat the fix as verified for this demo viewpoint."
            if status == "improved"
            else "Pattern did not drop enough. Re-check the seal/insulation and re-scan from the same distance and angle."
        ),
    }
