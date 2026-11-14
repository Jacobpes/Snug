from pathlib import Path

import numpy as np

from snug.detect import ElementDetector, rois_from_config
from snug.nn import TinyFomo, temps_to_input
from snug.simulate import HouseSimConfig, simulate_scene

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "detector.npz"


def test_rois_fallback_without_model():
    det = ElementDetector(weights=Path("/no/such.npz"), config=None, mode="hybrid")
    det.model = None
    det.onnx_session = None
    _, temps, _ = simulate_scene(15.0, HouseSimConfig(seed=1))
    boxes = det.detect(temps)
    names = {b["name"] for b in boxes}
    assert "window" in names and "door" in names and "insulated_wall" in names


def test_config_rois_mode():
    cfg = {"rois": [{"name": "window", "element": "window", "x": 10, "y": 10, "w": 20, "h": 20}]}
    det = ElementDetector(weights=Path("/no/such.npz"), config=cfg, mode="rois")
    temps = np.zeros((192, 256), dtype=np.float32)
    boxes = det.detect(temps)
    assert boxes == [
        {"name": "window", "element": "window", "x": 10, "y": 10, "w": 20, "h": 20, "source": "config"}
    ]


def test_trained_model_runs_and_sees_house_classes():
    if not WEIGHTS.is_file():
        return
    det = ElementDetector(weights=WEIGHTS, mode="auto")
    assert det.available
    _, temps, meta = simulate_scene(18.0, HouseSimConfig(seed=5, heat_delay_s=0, heat_ramp_s=1, jitter=False))
    boxes = det.detect(temps, ambient_c=meta["outdoor_c"])
    elements = {b["element"] for b in boxes}
    # The synthetic-trained model is coarse; require it to fire on at least two real classes.
    assert len(elements) >= 2
    grid, _ = det.infer_grid(temps, ambient_c=meta["outdoor_c"])
    assert grid.shape == (12, 16)
    assert grid.max() >= 1


def test_numpy_and_input_shape():
    if not WEIGHTS.is_file():
        return
    model = TinyFomo.load(WEIGHTS)
    x = temps_to_input(np.zeros((192, 256), dtype=np.float32), 18.0)
    logits = model.forward(x)
    assert logits.shape[0] == 6
    assert logits.shape[1:] == (12, 16)
