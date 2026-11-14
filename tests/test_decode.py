import numpy as np

from snug.capture import SENSOR_H, SENSOR_W, celsius_to_raw, encode_tc001_frame, raw_to_celsius, split_frame
from snug.colormap import temps_to_ironbow


def test_raw_roundtrip_known_temperature():
    # 25 °C → raw = (25 + 273.15) * 64
    temps = np.full((SENSOR_H, SENSOR_W), 25.0, dtype=np.float32)
    raw = celsius_to_raw(temps)
    back = raw_to_celsius(raw)
    assert back.shape == (SENSOR_H, SENSOR_W)
    assert np.allclose(back, 25.0, atol=1 / 64 + 1e-5)


def test_split_frame_visual_and_thermal():
    temps = np.linspace(18, 32, SENSOR_H * SENSOR_W, dtype=np.float32).reshape(SENSOR_H, SENSOR_W)
    visual = temps_to_ironbow(temps)
    frame = encode_tc001_frame(visual, temps)
    assert frame.shape == (384, 256, 2)
    vis, rec = split_frame(frame)
    assert vis.shape == (192, 256, 3)
    assert rec.shape == (192, 256)
    assert np.allclose(rec, temps, atol=1 / 64 + 0.05)


def test_uint8_overflow_guard():
    # High temperatures must not wrap if we forgot the uint16 cast.
    temps = np.full((4, 4), 80.0, dtype=np.float32)
    raw = celsius_to_raw(temps)
    # Simulate the bug path: multiplying as uint8 would overflow.
    lo = raw[:, :, 0].astype(np.uint16)
    hi = raw[:, :, 1].astype(np.uint16)
    rec = (lo + hi * np.uint16(256)).astype(np.float32) / 64.0 - 273.15
    assert np.allclose(rec, 80.0, atol=1 / 64 + 1e-4)
