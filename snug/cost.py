"""Annual extra heating-cost estimate for a detected thermal defect (Finland).

Surface method (labelled Estimate everywhere it is shown)
--------------------------------------------------------
    q      = (T_in - T_surface) / R_si          interior survey
    q      = (T_surface - T_out) / R_se         exterior survey
    U_eff  = q / (T_in - T_out)
    ΔU     = max(0, U_defect - U_ref)
    A      = (anomaly fraction of the element region) × assumed real area
    E_kWh  = ΔU · A · S17 · 24 / 1000
    €/year = E_kWh / η_or_COP × price

R_si = 0.13 m²K/W and R_se = 0.04 m²K/W for horizontal heat flow,
EN ISO 6946 as tabulated by the Finnish Ministry of the Environment
(Ympäristöministeriö, Energiatehokkuus, Table 2 — 0.10 up, 0.13
horizontal, 0.17 down; R_se 0.04 all directions).

S17 is the Finnish Meteorological Institute heating degree-day
(lämmitystarveluku) for the 1991–2020 normal, municipality-adjusted
with the official K1 coefficient: S_local = S_comparison / K1.
See https://www.ilmatieteenlaitos.fi/lammitystarveluvut

The method is approximate. It needs a meaningful indoor–outdoor ΔT
(about 10 °C or more) during the scan, maps a scale-model anomaly
onto an assumed real element area, and underestimates true air-leak
losses (infiltration is not modelled — only the surface signature).
FMI S17 assumes a 17 °C indoor balance temperature; a scan T_in of
21 °C is therefore not a contradiction, but it is a limitation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# EN ISO 6946 / Ympäristöministeriö Energiatehokkuus Table 2 (horizontal).
R_SI = 0.13  # m²K/W, internal surface resistance
R_SE = 0.04  # m²K/W, external surface resistance
MIN_SCAN_DT_C = 10.0
HOURS_PER_DAY = 24.0

# Official comparison-station annual S17, 1991–2020 normal (°C·d).
# Source CSV: https://cdn.fmi.fi/documents/climate/wau2020/lammitystarveluvut_1991-2020.csv
# Linked from https://www.ilmatieteenlaitos.fi/lammitystarveluvut
FMI_S17_SOURCE_PAGE = "https://www.ilmatieteenlaitos.fi/lammitystarveluvut"
FMI_S17_CSV = "https://cdn.fmi.fi/documents/climate/wau2020/lammitystarveluvut_1991-2020.csv"
FMI_K1_CSV = "https://cdn.fmi.fi/documents/climate/wau2020/kuntakertoimet_2023.csv"
YM_UVALUE_GUIDANCE = (
    "https://ym.fi/documents/1410903/0/"
    "Rakennuksen%20U-arvo_laskenta_Energiatehokkuus%2020250311%20(5).pdf"
)

# Hardware kit list prices, checked 2026-11-14. Do not invent figures:
# only official store (or clearly sourced retailer) prices belong here.
KIT_CHECKED = "2026-11-14"
TC001_OFFICIAL = {
    "id": "tc001",
    "name": "Topdon TC001",
    "price_eur": 225.00,
    "vat": "excluded",
    "source_url": "https://eu.topdon.com/products/tc001",
    "source_label": "Official TOPDON Europe store",
    "checked": KIT_CHECKED,
    "note": "Listed as €225.00 EUR (excl. VAT) on 2026-11-14.",
}
UNO_Q_VARIANTS = {
    "2gb": {
        "id": "uno_q_2gb",
        "name": "Arduino UNO Q 2GB",
        "price_eur": 59.90,
        "vat": "included",
        "source_url": "https://store.arduino.cc/products/uno-q",
        "source_label": "Official Arduino Store",
        "checked": KIT_CHECKED,
        "note": "VAT included, as shown on the official store on 2026-11-14.",
    },
    "4gb": {
        "id": "uno_q_4gb",
        "name": "Arduino UNO Q 4GB",
        "price_eur": 82.90,
        "vat": "included",
        "source_url": "https://store.arduino.cc/products/uno-q-4gb",
        "source_label": "Official Arduino Store",
        "checked": KIT_CHECKED,
        "note": "VAT included, as shown on the official store on 2026-11-14.",
    },
}
SOFTWARE_ITEM = {
    "id": "snug",
    "name": "Snug software (open source)",
    "price_eur": 0.0,
    "vat": "n/a",
    "source_url": "",
    "source_label": "This project",
    "checked": KIT_CHECKED,
    "note": "Open source. €0.",
}
DEFAULT_UNO_Q_VARIANT = "2gb"

# Municipality rows from kuntakertoimet_2023.csv (latin-1, comma decimals).
# s17_local = s17_ref / k1   (FMI worked example: 1000 / 1.02 = 980).
LOCALITIES: dict[str, dict[str, Any]] = {
    "helsinki": {
        "id": "helsinki",
        "name_en": "Helsinki",
        "name_sv": "Helsingfors",
        "station": "Helsinki-Kaisaniemi",
        "s17_ref": 3689,
        "k1": 0.97,
        "note": None,
    },
    "espoo": {
        "id": "espoo",
        "name_en": "Espoo",
        "name_sv": "Esbo",
        "station": "Helsinki-Kaisaniemi",
        "s17_ref": 3689,
        "k1": 0.95,
        "note": "Espoo is not one of FMI's 16 comparison towns; official comparison station is Helsinki-Kaisaniemi, adjusted with K1 = 0.95.",
    },
    "vaasa": {
        "id": "vaasa",
        "name_en": "Vaasa",
        "name_sv": "Vasa",
        "station": "Vaasa",
        "s17_ref": 4276,
        "k1": 1.02,
        "note": None,
    },
    "pietarsaari": {
        "id": "pietarsaari",
        "name_en": "Pietarsaari / Jakobstad",
        "name_sv": "Jakobstad",
        "station": "Vaasa",
        "s17_ref": 4276,
        "k1": 0.98,
        "note": "Pietarsaari is not one of FMI's 16 comparison towns; official comparison station is Vaasa, adjusted with K1 = 0.98.",
    },
    "pedersore": {
        "id": "pedersore",
        "name_en": "Pedersöre",
        "name_sv": "Pedersöre",
        "station": "Vaasa",
        "s17_ref": 4276,
        "k1": 0.95,
        "note": "Pedersöre is not one of FMI's 16 comparison towns; official comparison station is Vaasa, adjusted with K1 = 0.95.",
    },
    "kokkola": {
        "id": "kokkola",
        "name_en": "Kokkola / Karleby",
        "name_sv": "Karleby",
        "station": "Vaasa",
        "s17_ref": 4276,
        "k1": 0.93,
        "note": "Kokkola is not one of FMI's 16 comparison towns; official comparison station is Vaasa, adjusted with K1 = 0.93 (Kruunupyy is 0.94, same station).",
    },
    "oulu": {
        "id": "oulu",
        "name_en": "Oulu",
        "name_sv": "Uleåborg",
        "station": "Oulu",
        "s17_ref": 4834,
        "k1": 0.96,
        "note": None,
    },
    "jyvaskyla": {
        "id": "jyvaskyla",
        "name_en": "Jyväskylä",
        "name_sv": "Jyväskylä",
        "station": "Jyväskylä",
        "s17_ref": 4622,
        "k1": 1.02,
        "note": None,
    },
    "rovaniemi": {
        "id": "rovaniemi",
        "name_en": "Rovaniemi",
        "name_sv": "Rovaniemi",
        "station": "Sodankylä",
        "s17_ref": 5924,
        "k1": 1.05,
        "note": "Rovaniemi is not one of FMI's 16 comparison towns; official comparison station is Sodankylä, adjusted with K1 = 1.05.",
    },
    "sodankyla": {
        "id": "sodankyla",
        "name_en": "Sodankylä",
        "name_sv": "Sodankylä",
        "station": "Sodankylä",
        "s17_ref": 5924,
        "k1": 0.97,
        "note": None,
    },
}

DEFAULT_REAL_AREAS_M2: dict[str, float] = {
    "window": 1.5,
    "door": 2.0,
    "wall": 10.0,
    "roof": 10.0,
    "corner": 2.0,
}

# Prices: Tilastokeskus, Energian hinnat, June 2026 (published 24 Sep 2026).
# Electricity and district heating defaults from table 13nl (purchased heating
# energy, detached house). Ranges from neighbouring official tables (13rb, 12gd).
# Oil fuel price from table 12ge (light fuel oil €/MWh). Boiler efficiency is
# an assumed typical range, not a Statistics Finland figure.
STATFIN_EHI = "https://stat.fi/fi/tilasto/ehi"
STATFIN_NEWS_2026_Q2 = (
    "https://stat.fi/fi/uutinen/energian-hinnat-nousivat-kesakuussa-"
    "kotitalouksien-sahko-kallistui-noin-kymmenyksella"
)
PRICE_PERIOD = "2026-06"
PRICE_PUBLISHED = "2026-09-24"

HEATING_TYPES: dict[str, dict[str, Any]] = {
    "electric": {
        "id": "electric",
        "name_en": "direct electric",
        "cop_low": 1.0,
        "cop_high": 1.0,
        "efficiency_note": "COP = 1 (resistive heating).",
        "price_low": 0.14892,   # 13rb, household > 15 000 kWh, 14.892 snt/kWh
        "price_high": 0.17910,  # 13rb, household 5 000–15 000 kWh, 17.91 snt/kWh
        "price_default": 0.160,  # 13nl, detached house electricity, 160 €/MWh
        "price_unit": "EUR/kWh",
        "price_note": (
            "Tilastokeskus Energian hinnat June 2026: purchased electricity for "
            "a detached house 160 €/MWh (13nl); household total price 14.892–17.91 "
            "snt/kWh for >15 000 kWh and 5 000–15 000 kWh classes (13rb). Incl. tax and network."
        ),
    },
    "heat_pump": {
        "id": "heat_pump",
        "name_en": "air-source heat pump",
        "cop_low": 2.5,
        "cop_high": 3.0,
        "efficiency_note": "Typical Finnish seasonal COP / SPF range 2.5–3.0 (assumption, not a measured SPF).",
        "price_low": 0.14892,
        "price_high": 0.17910,
        "price_default": 0.160,
        "price_unit": "EUR/kWh",
        "price_note": "Same electricity prices as direct electric; divided by seasonal COP.",
    },
    "ground_source": {
        "id": "ground_source",
        "name_en": "ground-source heat pump",
        "cop_low": 3.0,
        "cop_high": 3.5,
        "efficiency_note": "Typical seasonal COP / SPF around 3–3.5 (assumption, not a measured SPF).",
        "price_low": 0.14892,
        "price_high": 0.17910,
        "price_default": 0.160,
        "price_unit": "EUR/kWh",
        "price_note": "Same electricity prices as direct electric; divided by seasonal COP.",
    },
    "district": {
        "id": "district",
        "name_en": "district heating",
        "cop_low": 1.0,
        "cop_high": 1.0,
        "efficiency_note": "Price is for delivered heat (η = 1).",
        "price_low": 0.10949,    # 12gd, terraced/small block, 109.49 €/MWh
        "price_high": 0.11439,   # 12gd, detached house, 114.39 €/MWh
        "price_default": 0.114,  # 13nl, detached house district heat, 114 €/MWh
        "price_unit": "EUR/kWh",
        "price_note": (
            "Tilastokeskus Energian hinnat June 2026: detached-house district heat "
            "114 €/MWh (13nl) / 114.39 €/MWh (12gd typitalo); terraced/small block 109.49 €/MWh."
        ),
    },
    "oil": {
        "id": "oil",
        "name_en": "oil",
        "cop_low": 0.85,  # boiler efficiency range
        "cop_high": 0.90,
        "efficiency_note": "Assumed oil-boiler efficiency 0.85–0.90 (not a Tilastokeskus figure). Fuel price is per kWh of oil energy content.",
        "price_low": 0.1675,     # 12ge light fuel oil 167.5 €/MWh
        "price_high": 0.1675,
        "price_default": 0.1675,
        "price_unit": "EUR/kWh",
        "price_note": (
            "Tilastokeskus Energian hinnat June 2026, table 12ge: light fuel oil "
            "167.5 €/MWh (1.67 €/l). 13nl detached-house oil 167 €/MWh."
        ),
    },
}

DEFAULT_LOCALITY = "vaasa"
DEFAULT_HEATING = "heat_pump"
DEFAULT_T_IN_C = 21.0
DEFAULT_LIVE_T_OUT_C = 0.0  # °C, typical Finnish heating-season day; configurable
METHOD_UNCERTAINTY = 0.15  # widen a collapsed range so we don't imply false precision


def s17_for(locality: str) -> float:
    loc = resolve_locality(locality)
    return float(loc["s17_ref"]) / float(loc["k1"])


def resolve_locality(locality: str | None) -> dict[str, Any]:
    key = (locality or DEFAULT_LOCALITY).strip().lower()
    aliases = {
        "helsingfors": "helsinki",
        "esbo": "espoo",
        "vasa": "vaasa",
        "jakobstad": "pietarsaari",
        "jakobstad/pietarsaari": "pietarsaari",
        "pietarsaari/jakobstad": "pietarsaari",
        "pedersöre": "pedersore",
        "pedersoren kunta": "pedersore",
        "karleby": "kokkola",
        "kruunupyy": "kokkola",
        "kronoby": "kokkola",
        "uleaborg": "oulu",
        "uleåborg": "oulu",
        "jyvaskyla": "jyvaskyla",
        "jyväskylä": "jyvaskyla",
        "sodankylä": "sodankyla",
    }
    key = aliases.get(key, key)
    if key not in LOCALITIES:
        raise KeyError(f"Unknown locality '{locality}'. Known: {', '.join(LOCALITIES)}")
    loc = dict(LOCALITIES[key])
    loc["s17"] = float(loc["s17_ref"]) / float(loc["k1"])
    return loc


def resolve_heating(heating_type: str | None) -> dict[str, Any]:
    key = (heating_type or DEFAULT_HEATING).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "direct_electric": "electric",
        "directelectric": "electric",
        "electricity": "electric",
        "ashp": "heat_pump",
        "air_source": "heat_pump",
        "air_source_heat_pump": "heat_pump",
        "ilmalampopumppu": "heat_pump",
        "gshp": "ground_source",
        "ground_source_heat_pump": "ground_source",
        "maalampo": "ground_source",
        "district_heating": "district",
        "kaukolampo": "district",
        "kaukolämpö": "district",
        "oljy": "oil",
        "öljy": "oil",
    }
    key = aliases.get(key, key)
    if key not in HEATING_TYPES:
        raise KeyError(f"Unknown heating type '{heating_type}'. Known: {', '.join(HEATING_TYPES)}")
    return dict(HEATING_TYPES[key])


def real_area_m2(element: str, areas: dict[str, float] | None = None) -> float:
    table = {**DEFAULT_REAL_AREAS_M2, **(areas or {})}
    return float(table.get(element, table["wall"]))


def scale_area_m2(anomaly_frac: float, element: str, areas: dict[str, float] | None = None) -> float:
    frac = min(max(float(anomaly_frac or 0.0), 0.0), 1.0)
    return frac * real_area_m2(element, areas)


def choose_survey_side(
    t_surface: float,
    t_in: float,
    t_out: float,
    side: str = "auto",
) -> str:
    requested = (side or "auto").lower()
    if requested in {"interior", "inside", "internal"}:
        return "interior"
    if requested in {"exterior", "outside", "external"}:
        return "exterior"
    # Auto: surfaces closer to indoor air (typical interior thermogram) use R_si;
    # surfaces closer to outdoor air (facade / scale-model demo) use R_se.
    if abs(t_in - t_surface) <= abs(t_surface - t_out):
        return "interior"
    return "exterior"


def heat_flux_wm2(t_surface: float, t_in: float, t_out: float, side: str) -> float:
    if side == "interior":
        return (t_in - t_surface) / R_SI
    return (t_surface - t_out) / R_SE


def effective_u(t_surface: float, t_in: float, t_out: float, side: str = "auto") -> tuple[float, float, str]:
    """Return (U_eff, q, side_used). U is clamped at 0 if the flux sign is wrong."""
    used = choose_survey_side(t_surface, t_in, t_out, side)
    dt = t_in - t_out
    if dt <= 0.5:
        return 0.0, 0.0, used
    q = heat_flux_wm2(t_surface, t_in, t_out, used)
    u = q / dt
    if u < 0 or q < 0:
        return 0.0, 0.0, used
    return float(u), float(q), used


def annual_energy_kwh(delta_u: float, area_m2: float, s17: float) -> float:
    if delta_u <= 0 or area_m2 <= 0 or s17 <= 0:
        return 0.0
    return float(delta_u) * float(area_m2) * float(s17) * HOURS_PER_DAY / 1000.0


def annual_cost_eur(energy_kwh: float, cop_or_eff: float, price: float) -> float:
    if energy_kwh <= 0 or cop_or_eff <= 0 or price < 0:
        return 0.0
    return energy_kwh / cop_or_eff * price


def nice_eur(value: float) -> int:
    """Round a euro amount to avoid false precision in the UI."""
    x = max(0.0, float(value))
    if x == 0:
        return 0
    if x < 10:
        return max(1, int(round(x)))
    if x < 50:
        return int(5 * round(x / 5.0))
    if x < 200:
        return int(10 * round(x / 10.0))
    if x < 1000:
        return int(20 * round(x / 20.0))
    return int(50 * round(x / 50.0))


def format_range_eur(low: float, high: float, *, suffix: str = "/ year") -> str:
    a, b = nice_eur(low), nice_eur(high)
    if b < a:
        a, b = b, a
    if a == b:
        return f"€{a} {suffix}".replace("  ", " ")
    return f"€{a}–{b} {suffix}".replace("  ", " ")


@dataclass
class CostSettings:
    locality: str = DEFAULT_LOCALITY
    heating_type: str = DEFAULT_HEATING
    t_in_c: float = DEFAULT_T_IN_C
    t_out_c: float | None = None  # None = use measured / simulator ambient
    survey_side: str = "auto"
    price_eur_per_kwh: float | None = None
    cop: float | None = None
    real_areas_m2: dict[str, float] = field(default_factory=dict)
    # Live camera (interior survey, no simulator meta): "auto" = use the measured scene
    # temperature as T_in; "manual" = use t_in_c as set by the user (CLI / config / dashboard).
    t_in_source: str = "auto"
    # Outdoor temperature assumed for a live survey when t_out_c is not set (Finnish heating season).
    default_t_out_c: float = DEFAULT_LIVE_T_OUT_C

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "CostSettings":
        raw = dict((cfg or {}).get("cost") or {})
        areas = dict(DEFAULT_REAL_AREAS_M2)
        areas.update(raw.get("real_areas_m2") or {})
        t_out = raw.get("t_out_c", raw.get("t_out"))
        t_in = raw.get("t_in_c", raw.get("t_in"))
        source = str(raw.get("t_in_source") or ("manual" if t_in not in (None, "") else "auto")).lower()
        if source not in {"auto", "manual"}:
            source = "auto"
        if t_in in (None, ""):
            t_in = DEFAULT_T_IN_C
            source = "auto"
        d_out = raw.get("default_t_out_c")
        return cls(
            locality=str(raw.get("locality", DEFAULT_LOCALITY)),
            heating_type=str(raw.get("heating_type", DEFAULT_HEATING)),
            t_in_c=float(t_in),
            t_out_c=None if t_out is None or t_out == "" else float(t_out),
            survey_side=str(raw.get("survey_side", "auto")),
            price_eur_per_kwh=(
                None if raw.get("price_eur_per_kwh") in (None, "") else float(raw["price_eur_per_kwh"])
            ),
            cop=None if raw.get("cop") in (None, "") else float(raw["cop"]),
            real_areas_m2=areas,
            t_in_source=source,
            default_t_out_c=DEFAULT_LIVE_T_OUT_C if d_out in (None, "") else float(d_out),
        )

    def updated(self, patch: dict[str, Any] | None) -> "CostSettings":
        data = self.to_public()
        if patch:
            for key in (
                "locality",
                "heating_type",
                "t_in_c",
                "t_out_c",
                "survey_side",
                "price_eur_per_kwh",
                "cop",
                "real_areas_m2",
                "t_in_source",
                "default_t_out_c",
            ):
                if key in patch:
                    data[key] = patch[key]
            # Changing the indoor temperature to a different value pins it ("manual");
            # re-posting the unchanged value keeps automatic scene-temperature mode.
            if "t_in_c" in patch and "t_in_source" not in patch:
                try:
                    changed = patch["t_in_c"] not in (None, "") and abs(float(patch["t_in_c"]) - float(self.t_in_c)) > 1e-9
                except (TypeError, ValueError):
                    changed = False
                if changed:
                    data["t_in_source"] = "manual"
            if "t_in_c" in patch and patch["t_in_c"] in (None, ""):
                data["t_in_c"] = None
                data["t_in_source"] = "auto"
        # Re-parse through from_config for type coercion.
        return CostSettings.from_config({"cost": data})

    def to_public(self) -> dict[str, Any]:
        loc = resolve_locality(self.locality)
        heat = resolve_heating(self.heating_type)
        return {
            "locality": loc["id"],
            "locality_name": loc["name_en"],
            "heating_type": heat["id"],
            "heating_name": heat["name_en"],
            "t_in_c": self.t_in_c,
            "t_out_c": self.t_out_c,
            "survey_side": self.survey_side,
            "t_in_source": self.t_in_source,
            "default_t_out_c": self.default_t_out_c,
            "price_eur_per_kwh": self.price_eur_per_kwh,
            "cop": self.cop,
            "real_areas_m2": dict(self.real_areas_m2 or DEFAULT_REAL_AREAS_M2),
            "s17": round(loc["s17"], 1),
            "s17_ref": loc["s17_ref"],
            "k1": loc["k1"],
            "station": loc["station"],
            "locality_note": loc.get("note"),
        }


@dataclass
class CostEstimate:
    ok: bool
    label: str
    short: str
    energy_kwh: float
    cost_eur_low: float
    cost_eur_high: float
    cost_eur_mid: float
    delta_u: float
    u_defect: float
    u_ref: float
    area_m2: float
    anomaly_frac: float
    s17: float
    locality: str
    locality_name: str
    heating_type: str
    heating_name: str
    survey_side: str
    t_in_c: float
    t_out_c: float
    t_surface_c: float
    t_ref_surface_c: float
    scan_dt_c: float
    reliable: bool
    notes: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_eur_low_display"] = nice_eur(self.cost_eur_low)
        d["cost_eur_high_display"] = nice_eur(self.cost_eur_high)
        return d


def _cost_range(energy_kwh: float, heat: dict[str, Any], price_override: float | None, cop_override: float | None) -> tuple[float, float, float]:
    price_low = float(heat["price_low"])
    price_high = float(heat["price_high"])
    if price_override is not None:
        price_low = price_high = float(price_override)
    cop_low = float(heat["cop_low"])
    cop_high = float(heat["cop_high"])
    if cop_override is not None and cop_override > 0:
        cop_low = cop_high = float(cop_override)
    # Cheapest = highest COP/efficiency × lowest price; dearest is the opposite.
    c_low = annual_cost_eur(energy_kwh, cop_high, price_low)
    c_high = annual_cost_eur(energy_kwh, cop_low, price_high)
    if c_high < c_low:
        c_low, c_high = c_high, c_low
    mid = 0.5 * (c_low + c_high)
    if mid > 0 and (c_high - c_low) < METHOD_UNCERTAINTY * 2.0 * mid:
        c_low = mid * (1.0 - METHOD_UNCERTAINTY)
        c_high = mid * (1.0 + METHOD_UNCERTAINTY)
    return c_low, c_high, mid


def estimate_defect_cost(
    *,
    element: str,
    anomaly_frac: float,
    t_surface: float,
    t_ref_surface: float,
    t_in: float,
    t_out: float,
    settings: CostSettings | None = None,
    defect: str | None = None,
    normal: bool = False,
) -> CostEstimate:
    settings = settings or CostSettings()
    loc = resolve_locality(settings.locality)
    heat = resolve_heating(settings.heating_type)
    s17 = loc["s17"]
    area = scale_area_m2(anomaly_frac, element, settings.real_areas_m2)
    # Decide interior vs exterior once from the reference (normal) surface so a
    # hot defect on an outdoor facade is not flipped to the interior formula.
    side_probe = t_ref_surface if (settings.survey_side or "auto").lower() == "auto" else t_surface
    side = choose_survey_side(side_probe, t_in, t_out, settings.survey_side)
    u_def, _q_def, side = effective_u(t_surface, t_in, t_out, side)
    u_ref, _q_ref, _ = effective_u(t_ref_surface, t_in, t_out, side)
    delta_u = max(0.0, u_def - u_ref)
    energy = annual_energy_kwh(delta_u, area, s17)
    notes: list[str] = []
    if loc.get("note"):
        notes.append(str(loc["note"]))
    if defect == "air_leak":
        notes.append(
            "Air leaks are estimated with the same surface-temperature method; "
            "extra infiltration losses are not modelled and may be underestimated."
        )
    scan_dt = t_in - t_out
    reliable = scan_dt >= MIN_SCAN_DT_C
    if not reliable:
        notes.append(
            f"Scan indoor–outdoor ΔT is {scan_dt:.1f} °C; the surface method needs "
            f"about {MIN_SCAN_DT_C:.0f} °C or more to be trustworthy."
        )
    if normal or delta_u <= 0 or energy <= 0:
        energy = 0.0
        c_low = c_high = mid = 0.0
        label = "Fix → save €0 / year (no extra loss vs reference)"
        short = "Fix → save €0/yr (Estimate)"
        ok = True
    else:
        c_low, c_high, mid = _cost_range(energy, heat, settings.price_eur_per_kwh, settings.cop)
        climate = loc["name_en"]
        label = (
            f"Fix → save ~{format_range_eur(c_low, c_high)} "
            f"(Estimate, {climate} climate, {heat['name_en']})"
        )
        short = f"Fix → save ~{format_range_eur(c_low, c_high, suffix='/yr')} (Estimate)"
        ok = True
        if not reliable:
            label += " — uncertain"
            short = short.replace("(Estimate)", "(uncertain Estimate)")

    sources = {
        "s17": FMI_S17_SOURCE_PAGE,
        "s17_csv": FMI_S17_CSV,
        "k1_csv": FMI_K1_CSV,
        "r_si": YM_UVALUE_GUIDANCE,
        "prices": STATFIN_EHI,
        "prices_news": STATFIN_NEWS_2026_Q2,
        "price_period": PRICE_PERIOD,
        "price_published": PRICE_PUBLISHED,
    }
    return CostEstimate(
        ok=ok,
        label=label,
        short=short,
        energy_kwh=float(energy),
        cost_eur_low=float(c_low),
        cost_eur_high=float(c_high),
        cost_eur_mid=float(mid),
        delta_u=float(delta_u),
        u_defect=float(u_def),
        u_ref=float(u_ref),
        area_m2=float(area),
        anomaly_frac=float(anomaly_frac or 0.0),
        s17=float(s17),
        locality=loc["id"],
        locality_name=loc["name_en"],
        heating_type=heat["id"],
        heating_name=heat["name_en"],
        survey_side=side,
        t_in_c=float(t_in),
        t_out_c=float(t_out),
        t_surface_c=float(t_surface),
        t_ref_surface_c=float(t_ref_surface),
        scan_dt_c=float(scan_dt),
        reliable=reliable,
        notes=notes,
        sources=sources,
    )


def attach_costs(
    findings: list[Any],
    settings: CostSettings,
    *,
    t_in: float,
    t_out: float,
) -> dict[str, Any]:
    """Annotate RegionFinding objects (or dicts) and return a scan-level total."""
    per: list[CostEstimate] = []
    if (settings.survey_side or "auto").lower() == "auto" and findings:
        probes = []
        for f in findings:
            val = f.get("reference_c") if isinstance(f, dict) else getattr(f, "reference_c", None)
            if val is not None:
                probes.append(float(val))
        if probes:
            probe = sum(probes) / len(probes)
            side = choose_survey_side(probe, t_in, t_out, "auto")
            settings = settings.updated({"survey_side": side})
    for f in findings:
        is_dict = isinstance(f, dict)
        getter = (lambda k, default=None, _f=f: _f.get(k, default)) if is_dict else (lambda k, default=None, _f=f: getattr(_f, k, default))
        est = estimate_defect_cost(
            element=str(getter("element") or "wall"),
            anomaly_frac=float(getter("anomaly_frac") or 0.0),
            t_surface=float(getter("t_surface_c") if getter("t_surface_c") is not None else getter("mean_c") or t_in),
            t_ref_surface=float(
                getter("t_ref_surface_c") if getter("t_ref_surface_c") is not None else getter("reference_c") or t_out
            ),
            t_in=t_in,
            t_out=t_out,
            settings=settings,
            defect=getter("defect"),
            normal=bool(getter("normal") or getter("defect") == "normal"),
        )
        per.append(est)
        payload = est.to_dict()
        if is_dict:
            f["cost"] = payload
            f["cost_label"] = est.label
            f["cost_short"] = est.short
        else:
            f.cost = payload
            f.cost_label = est.label
            f.cost_short = est.short

    defects = [e for e, f in zip(per, findings) if not (getattr(f, "normal", False) if not isinstance(f, dict) else f.get("normal") or f.get("defect") == "normal")]
    # Also include only those with energy > 0 in the total (normal already 0).
    total_low = sum(e.cost_eur_low for e in defects)
    total_high = sum(e.cost_eur_high for e in defects)
    total_kwh = sum(e.energy_kwh for e in defects)
    loc = resolve_locality(settings.locality)
    heat = resolve_heating(settings.heating_type)
    if total_kwh <= 0:
        total_label = "No leaks found"
        total_short = "No leaks found"
        banner = "No leaks found"
    else:
        rng = format_range_eur(total_low, total_high)
        total_label = (
            f"Fixing these saves ~{rng} "
            f"(Estimate, {loc['name_en']} climate, {heat['name_en']})"
        )
        total_short = f"Fixing these saves ~{rng}"
        banner = f"Fixing these saves ~{rng}"
    reliable = all(e.reliable for e in per) if per else True
    return {
        "label": total_label,
        "short": total_short,
        "banner": banner,
        "energy_kwh": float(total_kwh),
        "cost_eur_low": float(total_low),
        "cost_eur_high": float(total_high),
        "cost_eur_low_display": nice_eur(total_low),
        "cost_eur_high_display": nice_eur(total_high),
        "locality": loc["id"],
        "locality_name": loc["name_en"],
        "heating_type": heat["id"],
        "heating_name": heat["name_en"],
        "s17": round(loc["s17"], 1),
        "t_in_c": float(t_in),
        "t_out_c": float(t_out),
        "reliable": reliable,
        "n_defects": len(defects),
    }


def catalogue() -> dict[str, Any]:
    locs = []
    for loc in LOCALITIES.values():
        item = dict(loc)
        item["s17"] = round(float(loc["s17_ref"]) / float(loc["k1"]), 1)
        locs.append(item)
    heats = []
    for h in HEATING_TYPES.values():
        heats.append(
            {
                "id": h["id"],
                "name_en": h["name_en"],
                "cop_low": h["cop_low"],
                "cop_high": h["cop_high"],
                "price_default": h["price_default"],
                "price_low": h["price_low"],
                "price_high": h["price_high"],
                "efficiency_note": h["efficiency_note"],
                "price_note": h["price_note"],
            }
        )
    return {
        "localities": locs,
        "heating_types": heats,
        "defaults": {
            "locality": DEFAULT_LOCALITY,
            "heating_type": DEFAULT_HEATING,
            "t_in_c": DEFAULT_T_IN_C,
            "real_areas_m2": dict(DEFAULT_REAL_AREAS_M2),
            "r_si": R_SI,
            "r_se": R_SE,
            "min_scan_dt_c": MIN_SCAN_DT_C,
        },
        "sources": {
            "s17": FMI_S17_SOURCE_PAGE,
            "s17_csv": FMI_S17_CSV,
            "k1_csv": FMI_K1_CSV,
            "r_si": YM_UVALUE_GUIDANCE,
            "prices": STATFIN_EHI,
            "prices_news": STATFIN_NEWS_2026_Q2,
            "price_period": PRICE_PERIOD,
            "price_published": PRICE_PUBLISHED,
        },
    }


def format_price_eur(value: float) -> str:
    """Format a kit price: €225 or €59.90."""
    x = float(value)
    if abs(x - round(x)) < 1e-9:
        return f"€{int(round(x))}"
    return f"€{x:.2f}"


def payback_months(kit_total_eur: float, cost_eur_low: float, cost_eur_high: float) -> dict[str, Any]:
    """Months to recoup the kit from estimated yearly savings.

    Uses the mid-point of the savings range for the headline figure.
    Zero or negative savings → 'No leaks found' (no division by zero).
    """
    low = max(0.0, float(cost_eur_low or 0.0))
    high = max(0.0, float(cost_eur_high or 0.0))
    if high < low:
        low, high = high, low
    mid = 0.5 * (low + high)
    kit = float(kit_total_eur or 0.0)
    if mid <= 0 or kit < 0:
        return {
            "months": None,
            "months_fast": None,
            "months_slow": None,
            "label": "No leaks found",
        }
    months = kit / mid * 12.0
    months_fast = kit / high * 12.0
    months_slow = kit / max(low, 1e-9) * 12.0
    return {
        "months": float(months),
        "months_fast": float(months_fast),
        "months_slow": float(months_slow),
        "label": payback_label(months),
    }


def payback_label(months: float) -> str:
    if months < 1:
        return "Pays back in under a month (Estimate)"
    if months < 18:
        n = max(1, int(round(months)))
        unit = "month" if n == 1 else "months"
        return f"Pays back in ~{n} {unit} (Estimate)"
    years = months / 12.0
    if years < 10:
        y = round(years, 1)
        shown = int(y) if abs(y - int(y)) < 1e-9 else y
        return f"Pays back in ~{shown} years (Estimate)"
    return f"Pays back in ~{int(round(years))} years (Estimate)"


def _money(raw: Any, default: float) -> float:
    if raw in (None, ""):
        return float(default)
    return float(raw)


@dataclass
class KitSettings:
    """Editable official kit prices shown in the demo."""

    uno_q_variant: str = DEFAULT_UNO_Q_VARIANT
    tc001_price_eur: float = TC001_OFFICIAL["price_eur"]
    tc001_vat: str = TC001_OFFICIAL["vat"]
    tc001_source_url: str = TC001_OFFICIAL["source_url"]
    uno_q_2gb_price_eur: float = UNO_Q_VARIANTS["2gb"]["price_eur"]
    uno_q_4gb_price_eur: float = UNO_Q_VARIANTS["4gb"]["price_eur"]
    uno_q_vat: str = UNO_Q_VARIANTS["2gb"]["vat"]
    uno_q_2gb_source_url: str = UNO_Q_VARIANTS["2gb"]["source_url"]
    uno_q_4gb_source_url: str = UNO_Q_VARIANTS["4gb"]["source_url"]
    software_price_eur: float = SOFTWARE_ITEM["price_eur"]
    checked: str = KIT_CHECKED

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "KitSettings":
        raw = dict((cfg or {}).get("kit") or {})
        variant = str(raw.get("uno_q_variant", DEFAULT_UNO_Q_VARIANT)).lower()
        if variant not in UNO_Q_VARIANTS:
            variant = DEFAULT_UNO_Q_VARIANT
        return cls(
            uno_q_variant=variant,
            tc001_price_eur=_money(raw.get("tc001_price_eur"), TC001_OFFICIAL["price_eur"]),
            tc001_vat=str(raw.get("tc001_vat", TC001_OFFICIAL["vat"])),
            tc001_source_url=str(raw.get("tc001_source_url", TC001_OFFICIAL["source_url"])),
            uno_q_2gb_price_eur=_money(raw.get("uno_q_2gb_price_eur"), UNO_Q_VARIANTS["2gb"]["price_eur"]),
            uno_q_4gb_price_eur=_money(raw.get("uno_q_4gb_price_eur"), UNO_Q_VARIANTS["4gb"]["price_eur"]),
            uno_q_vat=str(raw.get("uno_q_vat", UNO_Q_VARIANTS["2gb"]["vat"])),
            uno_q_2gb_source_url=str(raw.get("uno_q_2gb_source_url", UNO_Q_VARIANTS["2gb"]["source_url"])),
            uno_q_4gb_source_url=str(raw.get("uno_q_4gb_source_url", UNO_Q_VARIANTS["4gb"]["source_url"])),
            software_price_eur=_money(raw.get("software_price_eur"), SOFTWARE_ITEM["price_eur"]),
            checked=str(raw.get("checked", KIT_CHECKED)),
        )

    def items(self) -> list[dict[str, Any]]:
        variant = self.uno_q_variant if self.uno_q_variant in UNO_Q_VARIANTS else DEFAULT_UNO_Q_VARIANT
        uno = dict(UNO_Q_VARIANTS[variant])
        uno["price_eur"] = float(self.uno_q_2gb_price_eur if variant == "2gb" else self.uno_q_4gb_price_eur)
        uno["vat"] = self.uno_q_vat
        uno["source_url"] = self.uno_q_2gb_source_url if variant == "2gb" else self.uno_q_4gb_source_url
        tc001 = dict(TC001_OFFICIAL)
        tc001["price_eur"] = float(self.tc001_price_eur)
        tc001["vat"] = self.tc001_vat
        tc001["source_url"] = self.tc001_source_url
        software = dict(SOFTWARE_ITEM)
        software["price_eur"] = float(self.software_price_eur)
        out = []
        for item in (tc001, uno, software):
            row = dict(item)
            row["price_label"] = format_price_eur(row["price_eur"])
            out.append(row)
        return out

    def total_eur(self) -> float:
        return float(sum(item["price_eur"] for item in self.items()))

    def to_public(self, cost_total: dict[str, Any] | None = None) -> dict[str, Any]:
        items = self.items()
        total = self.total_eur()
        low = float((cost_total or {}).get("cost_eur_low") or 0.0)
        high = float((cost_total or {}).get("cost_eur_high") or 0.0)
        pb = payback_months(total, low, high)
        return {
            "items": items,
            "total_eur": total,
            "total_label": format_price_eur(total),
            "uno_q_variant": self.uno_q_variant if self.uno_q_variant in UNO_Q_VARIANTS else DEFAULT_UNO_Q_VARIANT,
            "alternatives": {
                "uno_q_4gb": {
                    **UNO_Q_VARIANTS["4gb"],
                    "price_label": format_price_eur(self.uno_q_4gb_price_eur),
                    "price_eur": float(self.uno_q_4gb_price_eur),
                },
                "uno_q_2gb": {
                    **UNO_Q_VARIANTS["2gb"],
                    "price_label": format_price_eur(self.uno_q_2gb_price_eur),
                    "price_eur": float(self.uno_q_2gb_price_eur),
                },
            },
            "payback": pb,
            "checked": self.checked,
            "vat_note": (
                "TC001 official list price is excl. VAT; UNO Q official store price is VAT included. "
                f"Prices checked {self.checked}."
            ),
        }

