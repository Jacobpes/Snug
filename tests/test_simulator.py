import numpy as np

from snug.simulate import CLASS_ID, HouseSimConfig, indoor_temperature, semantic_masks, simulate_scene


def test_indoor_ramp():
    cfg = HouseSimConfig(outdoor_c=18, indoor_max_c=36, heat_delay_s=5, heat_ramp_s=10)
    assert indoor_temperature(0, cfg) == 18
    assert indoor_temperature(5, cfg) == 18
    mid = indoor_temperature(10, cfg)
    assert 18 < mid < 36
    assert abs(indoor_temperature(20, cfg) - 36) < 0.05


def test_scene_shapes_and_labels():
    visual, temps, meta = simulate_scene(20.0, HouseSimConfig(seed=1))
    assert visual.shape == (192, 256, 3)
    assert temps.shape == (192, 256)
    assert meta["labels"].shape == (192, 256)
    labels = meta["labels"]
    for name in ("wall", "window", "door", "roof", "corner"):
        assert (labels == CLASS_ID[name]).any(), name
    assert meta["indoor_c"] > meta["outdoor_c"]


def test_defects_make_regions_warmer():
    cfg_on = HouseSimConfig(seed=2, heat_delay_s=0, heat_ramp_s=1, insulation_gap=True, thermal_bridge=True, air_leak=True)
    cfg_off = HouseSimConfig(seed=2, heat_delay_s=0, heat_ramp_s=1, insulation_gap=False, thermal_bridge=False, air_leak=False)
    _, hot, _ = simulate_scene(20.0, cfg_on)
    _, cold, _ = simulate_scene(20.0, cfg_off)
    assert float(hot.max()) > float(cold.max()) + 1.0


def test_insulated_wall_cooler_than_uninsulated():
    cfg = HouseSimConfig(seed=3, heat_delay_s=0, heat_ramp_s=1, insulation_gap=True)
    _, temps, meta = simulate_scene(20.0, cfg)
    boxes = {b["name"]: b for b in meta["boxes"]}

    def mean_of(name):
        b = boxes[name]
        return float(temps[b["y"] : b["y"] + b["h"], b["x"] : b["x"] + b["w"]].mean())

    assert mean_of("uninsulated_wall") > mean_of("insulated_wall") + 1.0
