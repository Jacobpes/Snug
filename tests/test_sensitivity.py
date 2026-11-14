"""Live-camera (interior survey) sensitivity: cold spots, presets, live t_in/t_out, no noise false positives."""

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from snug.analysis import SENSITIVITY_OPTIONS, SENSITIVITY_PRESETS, HouseAnalyzer, drop_small_blobs, normalize_sensitivity
from snug.cost import CostSettings, attach_costs
from snug.detect import ElementDetector, load_config
from snug.server import Pipeline, create_app, parse_args
from snug.simulate import HouseSimConfig, simulate_scene

ROOM_C = 27.7


@pytest.fixture(scope="module")
def cfg():
    import copy
    from pathlib import Path

    return copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config.example.json"))


def _frame(rng, spots=(), noise=0.1):
    t = ROOM_C + rng.normal(0.0, noise, (192, 256)).astype(np.float32)
    for x, y, w, h, d in spots:
        t[y : y + h, x : x + w] += d
    return t


def _run_live(cfg, spots, sensitivity="high", n=15, seed=1, noise=0.1):
    cfg = {**cfg, "sensitivity": sensitivity}
    rng = np.random.default_rng(seed)
    ana = HouseAnalyzer(cfg)
    det = ElementDetector(config=cfg, mode="rois")
    findings = []
    for i in range(n):
        t = _frame(rng, spots, noise)
        findings = ana.analyze(t, det.detect(t), now=i * 0.12)  # no ambient/indoor => live camera
    return ana, findings


def test_sensitivity_presets_map_to_thresholds():
    assert SENSITIVITY_OPTIONS == ["low", "normal", "high"]
    assert SENSITIVITY_PRESETS["high"]["normal_dt"] == 0.5
    assert SENSITIVITY_PRESETS["high"]["anomaly_offset_c"] == 0.4
    assert SENSITIVITY_PRESETS["high"]["min_anomaly_frac"] == 0.01
    assert SENSITIVITY_PRESETS["high"]["medium_dt"] == 1.5 and SENSITIVITY_PRESETS["high"]["high_dt"] == 3.0
    assert SENSITIVITY_PRESETS["low"]["normal_dt"] == 1.5 and SENSITIVITY_PRESETS["low"]["anomaly_offset_c"] == 1.2
    ana = HouseAnalyzer({"sensitivity": "high"})
    assert ana.thresholds["normal_dt"] == 0.5 and ana.thresholds["anomaly_offset_c"] == 0.4
    assert ana.set_sensitivity("normal") == "normal"
    assert ana.thresholds["normal_dt"] == 1.0 and ana.thresholds["medium_dt"] == 2.5 and ana.thresholds["high_dt"] == 4.5
    assert HouseAnalyzer().sensitivity == "low"
    with pytest.raises(ValueError):
        normalize_sensitivity("extreme")
    # explicit thresholds in the config override the preset
    assert HouseAnalyzer({"sensitivity": "high", "thresholds": {"normal_dt": 0.9}}).thresholds["normal_dt"] == 0.9


def test_small_blobs_are_dropped():
    m = np.zeros((20, 20), dtype=bool)
    m[2, 2] = True
    m[10:14, 10:14] = True
    out = drop_small_blobs(m, 6)
    assert not out[2, 2] and out[10:14, 10:14].all()


def test_uniform_frame_with_noise_has_no_false_positives_at_high(cfg):
    for seed in range(5):
        ana, findings = _run_live(cfg, [], "high", n=25, seed=seed)
        assert findings and all(f.normal for f in findings), [(f.name, f.delta_c, f.anomaly_frac) for f in findings]
        assert all(f.anomaly_frac == 0.0 for f in findings)
        cs = CostSettings.from_config(cfg)
        total = attach_costs(findings, cs, t_in=ana.scene_c, t_out=0.0)
        assert total["energy_kwh"] == 0 and total["short"] == "No leaks found"


def test_cold_spot_on_wall_detected_with_cost(cfg):
    spots = [(148, 40, 20, 32, -1.5)]  # inside uninsulated_wall
    ana, findings = _run_live(cfg, spots, "high")
    f = {x.name: x for x in findings}["uninsulated_wall"]
    assert not f.normal and f.direction == "cold" and f.live
    assert f.defect == "insulation_gap"
    assert 1.0 < f.delta_c < 2.0
    assert f.t_surface_c < f.t_ref_surface_c - 1.0
    assert all(x.normal for x in findings if x.name != "uninsulated_wall")


def test_cold_spot_on_window_edge_is_air_leak(cfg):
    ana, findings = _run_live(cfg, [(100, 44, 48, 6, -1.2)], "high")
    f = {x.name: x for x in findings}["window"]
    assert not f.normal and f.direction == "cold" and f.defect == "air_leak"


def test_modest_cold_spot_is_missed_at_low_but_found_at_high(cfg):
    spots = [(148, 40, 20, 32, -0.8)]
    _, low = _run_live(cfg, spots, "low")
    _, high = _run_live(cfg, spots, "high")
    assert all(f.normal for f in low)
    assert not {f.name: f for f in high}["uninsulated_wall"].normal


def test_warm_anomaly_still_detected_warm(cfg):
    ana, findings = _run_live(cfg, [(148, 40, 20, 32, 2.0)], "high")
    f = {x.name: x for x in findings}["uninsulated_wall"]
    assert not f.normal and f.direction == "warm"


def test_live_cost_uses_scene_t_in_and_default_t_out_nonzero(cfg):
    ana, findings = _run_live(cfg, [(148, 40, 20, 32, -1.5)], "high")
    pipe = _pipe_live(cfg, ana)
    t_in, t_out, scan = pipe._scan_temps(None, None)
    assert t_in == pytest.approx(ROOM_C, abs=0.1)  # measured scene median, not the 21 °C default
    assert t_out == 0.0 and scan["scan_mode"] == "live_interior" and scan["t_out_source"] == "default"
    total = attach_costs(findings, pipe._settings_for_scan(scan), t_in=t_in, t_out=t_out)
    assert total["energy_kwh"] > 0 and total["cost_eur_high"] >= 2.0
    assert total["cost_eur_high"] < 100
    f = {x.name: x for x in findings}["uninsulated_wall"]
    assert f.cost["survey_side"] == "interior" and f.cost["delta_u"] > 0.2
    assert "Estimate" in f.cost_short
    # With the old fixed 21 °C indoor default the same cold spot costs nothing (the bug).
    old = attach_costs(findings, CostSettings(), t_in=21.0, t_out=0.0)
    assert old["energy_kwh"] == 0


def test_live_t_in_t_out_overrides(cfg):
    ana, _ = _run_live(cfg, [], "high", n=3)
    pipe = _pipe_live(cfg, ana)
    pipe.cost_settings = pipe.cost_settings.updated({"t_out_c": -10})
    assert pipe._scan_temps(None, None)[1] == -10.0
    # re-posting the unchanged default t_in keeps automatic mode; a different one pins it
    pipe.cost_settings = pipe.cost_settings.updated({"t_in_c": 21})
    assert pipe._scan_temps(None, None)[2]["t_in_source"] == "scene_median"
    pipe.cost_settings = pipe.cost_settings.updated({"t_in_c": 22.5})
    t_in, _, scan = pipe._scan_temps(None, None)
    assert t_in == 22.5 and scan["t_in_source"] == "manual"
    # simulator meta keeps the simulator temperatures
    pipe.cost_settings = CostSettings()
    assert pipe._scan_temps(36.0, 18.0)[:2] == (36.0, 18.0)


def test_simulator_behaviour_unchanged_at_normal():
    cfg = {"meters_per_pixel": 0.002, "reference_roi": "insulated_wall", "sensitivity": "normal"}
    sim = HouseSimConfig(seed=7, heat_delay_s=0, heat_ramp_s=1)
    ana = HouseAnalyzer(cfg)
    for t in np.linspace(16, 24, 14):
        _, temps, meta = simulate_scene(float(t), sim)
        f = ana.analyze(temps, meta["boxes"], ambient_c=meta["outdoor_c"], indoor_c=meta["indoor_c"], now=float(t))
    by = {x.name: x for x in f}
    assert by["insulated_wall"].normal
    assert by["uninsulated_wall"].defect == "insulation_gap" and by["uninsulated_wall"].direction == "warm"
    assert by["window"].defect == "air_leak" and not by["window"].live


def _pipe_live(cfg, ana):
    args = parse_args(["--rois-only"])
    pipe = Pipeline.__new__(Pipeline)
    pipe.args = args
    pipe.analyzer = ana
    pipe.cost_settings = CostSettings.from_config(cfg)
    pipe.lock = threading.Lock()
    return pipe


def _api_pipe():
    from snug.bridge import AlarmBridge
    from snug.cost import KitSettings

    args = parse_args(["--simulate", "--rois-only"])
    pipe = Pipeline.__new__(Pipeline)
    pipe.args = args
    pipe.cfg = {}
    pipe.detector = ElementDetector(config=None, mode="rois")
    pipe.analyzer = HouseAnalyzer({"reference_roi": "insulated_wall", "sensitivity": "high"})
    pipe.cost_settings = CostSettings()
    pipe.kit_settings = KitSettings()
    pipe.cost_total = {}
    pipe.bridge = AlarmBridge(enabled=False)
    pipe.lock = threading.Lock()
    pipe.visual = np.zeros((192, 256, 3), np.uint8)
    pipe.overlay = pipe.visual.copy()
    pipe.temps = np.zeros((192, 256), np.float32)
    pipe.findings = []
    pipe.meta = {}
    pipe.jpeg = b""
    pipe.fps = 0.0
    pipe.latencies = {}
    pipe.started = time.time()
    pipe.running = True
    pipe.frame_id = 0
    sim = HouseSimConfig(heat_delay_s=0, heat_ramp_s=1, seed=1)
    visual, temps, meta = simulate_scene(20.0, sim)
    pipe.process_frame(visual, temps, meta)
    return pipe


def test_api_exposes_and_sets_sensitivity():
    client = TestClient(create_app(_api_pipe()))
    st = client.get("/api/state").json()
    assert st["sensitivity"] == "high" and st["sensitivity_options"] == ["low", "normal", "high"]
    assert st["sensitivity_presets"]["normal"]["normal_dt"] == 1.0
    assert "scan_mode" in st["meta"] and "scene_c" in st["meta"]
    r = client.post("/api/cost/settings", json={"sensitivity": "low"})
    assert r.status_code == 200 and r.json()["sensitivity"] == "low"
    assert client.get("/api/state").json()["sensitivity"] == "low"
    r = client.post("/api/cost/settings", json={"sensitivity": "normal", "locality": "helsinki"})
    assert r.json()["settings"]["locality"] == "helsinki" and r.json()["sensitivity"] == "normal"
    bad = client.post("/api/cost/settings", json={"sensitivity": "bogus"})
    assert bad.status_code == 400 and bad.json()["options"] == ["low", "normal", "high"]
    # cost-only patches leave sensitivity untouched
    client.post("/api/cost/settings", json={"heating_type": "electric"})
    assert client.get("/api/state").json()["sensitivity"] == "normal"


def test_cli_sensitivity_flag():
    assert parse_args(["--sensitivity", "low"]).sensitivity == "low"
    assert parse_args([]).sensitivity is None


def _live_full_pipe(cfg):
    from snug.bridge import AlarmBridge
    from snug.cost import KitSettings

    pipe = Pipeline.__new__(Pipeline)
    pipe.args = parse_args(["--rois-only"])
    pipe.cfg = cfg
    pipe.detector = ElementDetector(config=cfg, mode="rois")
    pipe.analyzer = HouseAnalyzer({**cfg, "sensitivity": "high"})
    pipe.cost_settings = CostSettings.from_config(cfg)
    pipe.kit_settings = KitSettings()
    pipe.cost_total = {}
    pipe.bridge = AlarmBridge(enabled=False)
    pipe.lock = threading.Lock()
    pipe.findings, pipe.meta, pipe.latencies = [], {}, {}
    pipe.frame_id = 0
    return pipe


def test_live_shows_only_the_coldest_area_as_the_leak(cfg):
    """Interior view: a warm person/radiator must not win over a cold spot; one flag only."""
    pipe = _live_full_pipe(cfg)
    rng = np.random.default_rng(3)
    for i in range(12):
        t = _frame(rng, [(30, 120, 30, 40, +6.0), (190, 20, 18, 14, -2.5)])
        pipe.process_frame(np.zeros((192, 256, 3), np.uint8), t, {"elapsed_s": i * 0.12})
    assert len(pipe.findings) == 1
    f = pipe.findings[0]
    assert f.direction == "cold"
    cx, cy = 190 + 9, 20 + 7
    assert f.x <= cx <= f.x + f.w and f.y <= cy <= f.y + f.h
    assert pipe.cost_total["cost_eur_high"] > 0
    assert f.w <= 40 and f.h <= 40  # tight box around the cold spot, not a whole wall
