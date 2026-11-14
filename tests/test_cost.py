"""Known-input checks for the Finland heating-cost estimate."""

from __future__ import annotations

import math

import pytest

from snug.cost import (
    DEFAULT_LOCALITY,
    HEATING_TYPES,
    KIT_CHECKED,
    LOCALITIES,
    MIN_SCAN_DT_C,
    R_SE,
    R_SI,
    CostSettings,
    KitSettings,
    annual_energy_kwh,
    attach_costs,
    effective_u,
    estimate_defect_cost,
    format_price_eur,
    format_range_eur,
    nice_eur,
    payback_label,
    payback_months,
    s17_for,
    scale_area_m2,
)


def test_rsi_and_official_s17_tables():
    assert R_SI == 0.13
    assert R_SE == 0.04
    assert s17_for("vaasa") == 4276 / 1.02
    assert s17_for("helsinki") == 3689 / 0.97
    assert s17_for("espoo") == 3689 / 0.95
    assert s17_for("pietarsaari") == 4276 / 0.98
    assert s17_for("jakobstad") == s17_for("pietarsaari")
    assert s17_for("oulu") == 4834 / 0.96
    assert s17_for("jyvaskyla") == 4622 / 1.02
    assert s17_for("rovaniemi") == 5924 / 1.05
    assert s17_for("sodankyla") == 5924 / 0.97
    # Pietarsaari / Espoo / Rovaniemi are not comparison towns — notes must say so.
    from snug.cost import resolve_locality

    for key in ("pietarsaari", "espoo", "rovaniemi"):
        note = resolve_locality(key)["note"]
        assert note and "not one of FMI" in note
    assert DEFAULT_LOCALITY in LOCALITIES
    assert "heat_pump" in HEATING_TYPES


def test_scale_area_maps_fraction_to_real_element():
    assert scale_area_m2(0.2, "wall") == 2.0  # 0.2 × 10 m²
    assert scale_area_m2(0.5, "window") == 0.75  # 0.5 × 1.5 m²
    assert scale_area_m2(1.0, "door") == 2.0
    assert scale_area_m2(0.0, "wall") == 0.0
    assert scale_area_m2(1.5, "wall") == 10.0  # clamped


def test_interior_u_and_annual_energy_known_values():
    # Interior survey: T_in=21 °C, T_out=-5 °C, defect 16 °C, reference 19 °C.
    u_def, q_def, side = effective_u(16.0, 21.0, -5.0, "interior")
    u_ref, q_ref, _ = effective_u(19.0, 21.0, -5.0, "interior")
    assert side == "interior"
    assert q_def == (21.0 - 16.0) / R_SI
    assert u_def == q_def / 26.0
    assert math.isclose(u_def, 1.4792899408284024, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(u_ref, 0.5917159763313609, rel_tol=0, abs_tol=1e-12)
    delta_u = u_def - u_ref
    energy = annual_energy_kwh(delta_u, 2.0, 4276 / 1.02)
    # E = ΔU · A · S17 · 24 / 1000
    expected = delta_u * 2.0 * (4276 / 1.02) * 24.0 / 1000.0
    assert math.isclose(energy, expected, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(energy, 178.60076575008702, rel_tol=1e-12, abs_tol=1e-9)


def test_estimate_vaasa_heat_pump_range():
    settings = CostSettings(locality="vaasa", heating_type="heat_pump")
    est = estimate_defect_cost(
        element="wall",
        anomaly_frac=0.20,
        t_surface=16.0,
        t_ref_surface=19.0,
        t_in=21.0,
        t_out=-5.0,
        settings=settings,
        defect="insulation_gap",
    )
    assert est.ok and est.reliable
    assert est.survey_side == "interior"
    assert est.area_m2 == 2.0
    assert math.isclose(est.energy_kwh, 178.60076575008702, rel_tol=1e-12, abs_tol=1e-8)
    # Heat pump COP 2.5–3.0, electricity 0.14892–0.17910 €/kWh (Tilastokeskus 2026-06).
    energy = est.energy_kwh
    expected_low = energy / 3.0 * 0.14892
    expected_high = energy / 2.5 * 0.17910
    assert math.isclose(est.cost_eur_low, expected_low, rel_tol=1e-12, abs_tol=1e-8)
    assert math.isclose(est.cost_eur_high, expected_high, rel_tol=1e-12, abs_tol=1e-8)
    assert "Vaasa" in est.label
    assert "heat pump" in est.label.lower() or "air-source" in est.label.lower()
    assert est.label.startswith("Fix → save")
    assert est.short.startswith("Fix → save")
    assert "€" in est.short
    assert est.sources["s17"].startswith("https://www.ilmatieteenlaitos.fi/")
    assert est.sources["prices"].startswith("https://stat.fi/")


def test_exterior_auto_side_on_cold_facade():
    # Facade warmer than outdoor but closer to T_out than T_in → exterior, R_se.
    u, q, side = effective_u(5.0, 21.0, -5.0, "auto")
    assert side == "exterior"
    assert math.isclose(q, (5.0 - (-5.0)) / R_SE)
    assert math.isclose(u, q / 26.0)


def test_normal_or_better_than_reference_is_zero():
    settings = CostSettings()
    z = estimate_defect_cost(
        element="wall",
        anomaly_frac=0.3,
        t_surface=19.0,
        t_ref_surface=19.0,
        t_in=21.0,
        t_out=-5.0,
        settings=settings,
        normal=True,
    )
    assert z.energy_kwh == 0 and z.cost_eur_high == 0
    better = estimate_defect_cost(
        element="wall",
        anomaly_frac=0.3,
        t_surface=19.5,  # closer to indoor than the reference 19.0 in an interior survey
        t_ref_surface=16.0,
        t_in=21.0,
        t_out=-5.0,
        settings=settings,
        defect="insulation_gap",
    )
    # Interior: higher surface temp → lower U, so ΔU = 0.
    assert better.delta_u == 0
    assert better.cost_eur_high == 0


def test_small_delta_t_is_flagged_unreliable():
    est = estimate_defect_cost(
        element="wall",
        anomaly_frac=0.2,
        t_surface=18.5,
        t_ref_surface=19.5,
        t_in=21.0,
        t_out=16.0,  # only 5 °C
        settings=CostSettings(survey_side="interior"),
        defect="insulation_gap",
    )
    assert est.scan_dt_c == 5.0
    assert est.reliable is False
    assert any("10" in n for n in est.notes)
    assert MIN_SCAN_DT_C == 10.0


def test_air_leak_notes_infiltration_limitation():
    est = estimate_defect_cost(
        element="door",
        anomaly_frac=0.15,
        t_surface=16.0,
        t_ref_surface=19.0,
        t_in=21.0,
        t_out=-5.0,
        settings=CostSettings(),
        defect="air_leak",
    )
    assert any("infiltration" in n.lower() or "air leak" in n.lower() for n in est.notes)


def test_price_override_and_electric_range():
    settings = CostSettings(locality="helsinki", heating_type="electric", price_eur_per_kwh=0.16, cop=1.0)
    est = estimate_defect_cost(
        element="window",
        anomaly_frac=0.4,  # 0.4 × 1.5 = 0.6 m²
        t_surface=16.0,
        t_ref_surface=19.0,
        t_in=21.0,
        t_out=-5.0,
        settings=settings,
        defect="air_leak",
    )
    assert est.area_m2 == pytest.approx(0.6)
    assert est.locality == "helsinki"
    # Single price + COP=1 still yields a range via method uncertainty.
    assert est.cost_eur_low < est.cost_eur_mid < est.cost_eur_high


def test_attach_costs_totals_only_defects():
    class F:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.cost = None
            self.cost_label = None
            self.cost_short = None

    findings = [
        F(name="ok", element="wall", anomaly_frac=0.0, t_surface_c=19.0, t_ref_surface_c=19.0, mean_c=19.0, reference_c=19.0, defect="normal", normal=True),
        F(name="gap", element="wall", anomaly_frac=0.20, t_surface_c=16.0, t_ref_surface_c=19.0, mean_c=16.0, reference_c=19.0, defect="insulation_gap", normal=False),
        F(name="gap2", element="wall", anomaly_frac=0.20, t_surface_c=16.0, t_ref_surface_c=19.0, mean_c=16.0, reference_c=19.0, defect="insulation_gap", normal=False),
    ]
    total = attach_costs(findings, CostSettings(locality="vaasa", heating_type="heat_pump"), t_in=21.0, t_out=-5.0)
    assert findings[0].cost["energy_kwh"] == 0
    assert findings[1].cost["energy_kwh"] > 0
    assert math.isclose(total["energy_kwh"], 2 * findings[1].cost["energy_kwh"], rel_tol=1e-12)
    assert "Fixing these saves" in total["label"]
    assert "Fixing these saves" in total["banner"]
    assert "Vaasa" in total["label"]
    assert nice_eur(0) == 0
    assert "€" in format_range_eur(40, 60)


def test_payback_known_values_and_zero_savings():
    # Mid savings €11 / year, kit €284.90 → 284.90 / 11 * 12 months.
    kit = 225.0 + 59.90
    pb = payback_months(kit, 8.87, 12.79)
    mid = 0.5 * (8.87 + 12.79)
    expected = kit / mid * 12.0
    assert pb["months"] == pytest.approx(expected)
    assert pb["months_fast"] == pytest.approx(kit / 12.79 * 12.0)
    assert pb["months_slow"] == pytest.approx(kit / 8.87 * 12.0)
    assert pb["label"].startswith("Pays back in ~")
    assert "(Estimate)" in pb["label"]

    none = payback_months(kit, 0.0, 0.0)
    assert none["months"] is None
    assert none["label"] == "No leaks found"
    also = payback_months(kit, 0, -1)
    assert also["label"] == "No leaks found"
    assert payback_label(0.4) == "Pays back in under a month (Estimate)"
    assert payback_label(1.0) == "Pays back in ~1 month (Estimate)"
    assert payback_label(3.2) == "Pays back in ~3 months (Estimate)"
    assert "years" in payback_label(24.0)


def test_kit_defaults_and_config_override():
    kit = KitSettings()
    pub = kit.to_public({"cost_eur_low": 0, "cost_eur_high": 0})
    names = [i["name"] for i in pub["items"]]
    assert names == ["Topdon TC001", "Arduino UNO Q 2GB", "Snug software (open source)"]
    assert pub["items"][0]["price_eur"] == 225.0
    assert pub["items"][0]["vat"] == "excluded"
    assert pub["items"][0]["source_url"] == "https://eu.topdon.com/products/tc001"
    assert pub["items"][1]["price_eur"] == pytest.approx(59.90)
    assert pub["items"][1]["vat"] == "included"
    assert pub["items"][1]["source_url"] == "https://store.arduino.cc/products/uno-q"
    assert pub["items"][2]["price_eur"] == 0
    assert pub["total_eur"] == pytest.approx(284.90)
    assert pub["total_label"] == format_price_eur(284.90)
    assert pub["payback"]["label"] == "No leaks found"
    assert pub["checked"] == KIT_CHECKED
    assert "excl. VAT" in pub["vat_note"]
    assert pub["alternatives"]["uno_q_4gb"]["price_eur"] == pytest.approx(82.90)
    assert pub["alternatives"]["uno_q_4gb"]["source_url"] == "https://store.arduino.cc/products/uno-q-4gb"

    custom = KitSettings.from_config({
        "kit": {
            "uno_q_variant": "4gb",
            "tc001_price_eur": 200,
            "uno_q_4gb_price_eur": 80,
            "software_price_eur": 0,
        }
    })
    pub2 = custom.to_public({"cost_eur_low": 400, "cost_eur_high": 400})
    assert custom.uno_q_variant == "4gb"
    assert pub2["items"][1]["name"] == "Arduino UNO Q 4GB"
    assert pub2["total_eur"] == pytest.approx(280.0)
    # 280 / 400 * 12 = 8.4 months → rounds to 8
    assert pub2["payback"]["months"] == pytest.approx(8.4)
    assert "8 months" in pub2["payback"]["label"]


def test_attach_costs_zero_defects_says_no_leaks():
    class F:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.cost = None
            self.cost_label = None
            self.cost_short = None

    findings = [
        F(name="ok", element="wall", anomaly_frac=0.0, t_surface_c=19.0, t_ref_surface_c=19.0,
          mean_c=19.0, reference_c=19.0, defect="normal", normal=True),
    ]
    total = attach_costs(findings, CostSettings(), t_in=21.0, t_out=-5.0)
    assert total["banner"] == "No leaks found"
    assert total["label"] == "No leaks found"
    assert findings[0].cost_short.startswith("Fix → save €0")



def test_payback_over_12_months_shows_years_and_months():
    assert payback_label(12.0) == "Pays back in ~12 months (Estimate)"
    assert payback_label(13.0) == "Pays back in ~1 year 1 month (Estimate)"
    assert payback_label(27.4) == "Pays back in ~2 years 3 months (Estimate)"
    assert payback_label(24.0) == "Pays back in ~2 years (Estimate)"
