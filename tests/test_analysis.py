import numpy as np

from snug.analysis import HouseAnalyzer
from snug.simulate import HouseSimConfig, simulate_scene


def _regions(meta):
    return meta["boxes"]


def test_end_to_end_finds_defects_after_warmup():
    cfg = HouseSimConfig(seed=7, heat_delay_s=0, heat_ramp_s=1)
    _, temps, meta = simulate_scene(20.0, cfg)
    ana = HouseAnalyzer({"meters_per_pixel": 0.002, "reference_roi": "insulated_wall"})
    # Push several frames so fluctuation can be estimated for the door leak.
    findings = None
    for t in np.linspace(16, 22, 10):
        _, temps, meta = simulate_scene(float(t), cfg)
        findings = ana.analyze(temps, _regions(meta), ambient_c=meta["outdoor_c"], indoor_c=meta["indoor_c"], now=float(t))
    by_name = {f.name: f for f in findings}
    assert by_name["insulated_wall"].normal or by_name["insulated_wall"].severity in {"ok", "low"}
    # Uninsulated wall with a blob should not look like a healthy insulated wall.
    assert by_name["uninsulated_wall"].delta_c > by_name["insulated_wall"].delta_c
    assert by_name["uninsulated_wall"].defect in {"insulation_gap", "thermal_bridge"}
    assert by_name["door"].defect in {"air_leak", "thermal_bridge", "insulation_gap"}
    # At least one non-normal high-priority finding exists
    assert any(not f.normal for f in findings)


def test_before_after_verification():
    ana = HouseAnalyzer({"reference_roi": "insulated_wall"})
    cfg_bad = HouseSimConfig(seed=4, heat_delay_s=0, heat_ramp_s=1)
    cfg_good = HouseSimConfig(seed=4, heat_delay_s=0, heat_ramp_s=1, insulation_gap=False, thermal_bridge=False, air_leak=False)
    _, temps_b, meta_b = simulate_scene(20.0, cfg_bad)
    before = ana.analyze(temps_b, _regions(meta_b), ambient_c=meta_b["outdoor_c"], indoor_c=meta_b["indoor_c"], now=20.0)
    ana.mark_before(before)
    _, temps_a, meta_a = simulate_scene(20.0, cfg_good)
    after = ana.analyze(temps_a, _regions(meta_a), ambient_c=meta_a["outdoor_c"], indoor_c=meta_a["indoor_c"], now=21.0)
    report = ana.compare_after(after)
    assert "passed" in report
    assert "regions" in report
