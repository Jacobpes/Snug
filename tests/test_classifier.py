import numpy as np

from snug.analysis import classify_defect, estimate_q, severity_of, shape_features
from snug.actions import verification_verdict


def test_compact_blob_is_insulation_gap():
    mask = np.zeros((40, 40), dtype=bool)
    mask[12:28, 12:28] = True
    compact, aspect, _ = shape_features(mask)
    defect, _ = classify_defect(
        element="wall",
        delta_c=4.0,
        compactness=compact,
        aspect_ratio=aspect,
        edge_overlap=0.05,
        fluctuation_c=0.02,
        anomaly_frac=0.2,
        normal_dt=1.5,
    )
    assert defect == "insulation_gap"


def test_long_thin_line_is_thermal_bridge():
    mask = np.zeros((60, 20), dtype=bool)
    mask[:, 9:12] = True
    compact, aspect, _ = shape_features(mask)
    defect, _ = classify_defect(
        element="corner",
        delta_c=3.5,
        compactness=compact,
        aspect_ratio=aspect,
        edge_overlap=0.1,
        fluctuation_c=0.02,
        anomaly_frac=0.15,
        normal_dt=1.5,
    )
    assert defect == "thermal_bridge"


def test_door_edge_flutter_is_air_leak():
    mask = np.zeros((50, 30), dtype=bool)
    mask[5:45, 0:3] = True
    compact, aspect, _ = shape_features(mask)
    defect, _ = classify_defect(
        element="door",
        delta_c=3.0,
        compactness=compact,
        aspect_ratio=aspect,
        edge_overlap=0.7,
        fluctuation_c=0.25,
        anomaly_frac=0.12,
        normal_dt=1.5,
    )
    assert defect == "air_leak"


def test_small_delta_is_normal():
    defect, _ = classify_defect(
        element="wall",
        delta_c=0.4,
        compactness=0.8,
        aspect_ratio=1.1,
        edge_overlap=0.0,
        fluctuation_c=0.01,
        anomaly_frac=0.01,
        normal_dt=1.5,
    )
    assert defect == "normal"


def test_severity_thresholds():
    th = {"normal_dt": 1.5, "medium_dt": 3.0, "high_dt": 5.0, "area_frac_medium": 0.15, "area_frac_high": 0.30}
    assert severity_of(1.0, 0.0, th) == "ok"
    assert severity_of(2.0, 0.05, th) == "low"
    assert severity_of(3.5, 0.1, th) == "medium"
    assert severity_of(6.0, 0.1, th) == "high"


def test_heat_loss_formula():
    # U=1, A=2 m², dT=3 → 6 W
    assert estimate_q(1.0, 2.0, 3.0) == 6.0
    assert estimate_q(1.0, 2.0, -3.0) == 0.0


def test_verification_requires_drop():
    before = {"delta_c": 5.0, "anomaly_area_m2": 0.01, "defect": "air_leak"}
    after_good = {"delta_c": 1.0, "anomaly_area_m2": 0.002, "defect": "normal"}
    after_bad = {"delta_c": 4.8, "anomaly_area_m2": 0.009, "defect": "air_leak"}
    assert verification_verdict(before, after_good)["passed"]
    assert not verification_verdict(before, after_bad)["passed"]


def test_uniform_window_pane_is_not_insulation_gap():
    # Almost the whole pane warm and compact — expected glazing, not a wall defect.
    defect, _ = classify_defect(
        element="window",
        delta_c=6.0,
        compactness=0.8,
        aspect_ratio=1.2,
        edge_overlap=0.1,
        fluctuation_c=0.02,
        anomaly_frac=0.8,
        normal_dt=1.5,
    )
    assert defect != "insulation_gap"
