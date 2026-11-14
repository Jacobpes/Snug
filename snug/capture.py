"""Capture frames from a Topdon TC001, or yield simulated model-house frames.

The TC001 enumerates as a UVC camera. Radiometric data is in the bottom half
of a 256x384 YUYV frame. Conversion follows leswright1977/PyThermalCamera
(tc001v4.2.py), which credits LeoDJ on the EEVblog InfiRay P2 Pro thread:

    temp_C = (lo + hi * 256) / 64 - 273.15

where lo is byte 0 and hi is byte 1 of each thermal pixel. OpenCV must not
convert the frame to RGB, or the temperature values are destroyed
(set CAP_PROP_CONVERT_RGB to 0, not True/False).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from snug.simulate import HouseSimConfig, simulate_scene

SENSOR_W = 256
SENSOR_H = 192
FRAME_W = 256
FRAME_H = 384
TARGET_FPS = 25

# USB IDs reported by limlug/topdon-thermal-rs; confirm with lsusb on the device.
TC001_VID = 0x0BDA
TC001_PID = 0x5830


def raw_to_celsius(thdata: np.ndarray) -> np.ndarray:
    """Convert the 192x256x2 thermal half-frame to a Celsius matrix.

    Cast to uint16 before multiplying, otherwise uint8 overflows
    (see PyThermalCamera issue 21).
    """
    lo = thdata[:, :, 0].astype(np.uint16)
    hi = thdata[:, :, 1].astype(np.uint16)
    raw = lo + hi * np.uint16(256)
    return raw.astype(np.float32) / 64.0 - 273.15


def celsius_to_raw(temps: np.ndarray) -> np.ndarray:
    """Inverse of raw_to_celsius, for tests and synthetic UVC frames."""
    raw = np.clip(np.rint((temps.astype(np.float32) + 273.15) * 64.0), 0, 65535).astype(np.uint16)
    lo = (raw & 0xFF).astype(np.uint8)
    hi = (raw >> 8).astype(np.uint8)
    return np.dstack([lo, hi])


def encode_tc001_frame(visual_bgr: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """Pack a 256x384x2 YUYV-style frame (visual top, radiometric bottom)."""
    visual = visual_bgr
    if visual.shape[0] != SENSOR_H or visual.shape[1] != SENSOR_W:
        visual = cv2.resize(visual, (SENSOR_W, SENSOR_H), interpolation=cv2.INTER_LINEAR)
    yuyv = cv2.cvtColor(visual, cv2.COLOR_BGR2YUV)
    # Approximate YUYV packing: Y0,U,Y1,V per two pixels.
    y = yuyv[:, :, 0]
    u = yuyv[:, :, 1]
    v = yuyv[:, :, 2]
    top = np.empty((SENSOR_H, SENSOR_W, 2), dtype=np.uint8)
    top[:, 0::2, 0] = y[:, 0::2]
    top[:, 0::2, 1] = u[:, 0::2]
    top[:, 1::2, 0] = y[:, 1::2]
    top[:, 1::2, 1] = v[:, 1::2]
    bottom = celsius_to_raw(temps)
    return np.concatenate([top, bottom], axis=0)


def _sys_usb_ids(device_path: str) -> tuple[int, int] | None:
    name = Path(device_path).name
    sysdev = Path("/sys/class/video4linux") / name / "device"
    try:
        dev = sysdev.resolve()
    except OSError:
        return None
    for parent in [dev, *dev.parents]:
        vid_p, pid_p = parent / "idVendor", parent / "idProduct"
        if vid_p.is_file() and pid_p.is_file():
            try:
                return int(vid_p.read_text().strip(), 16), int(pid_p.read_text().strip(), 16)
            except ValueError:
                return None
    return None


def _supports_radiometric(dev: str) -> bool:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width == FRAME_W and height == FRAME_H


def find_tc001_device(preferred: int | None = None) -> str:
    """Auto-detect a TC001-like V4L2 device, or use --device if given."""
    if preferred is not None:
        path = f"/dev/video{preferred}"
        if not Path(path).exists():
            raise FileNotFoundError(f"Video device not found: {path}")
        return path

    candidates: list[Path] = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
    usb_hits: list[str] = []
    size_hits: list[str] = []
    for path in candidates:
        dev = str(path)
        ids = _sys_usb_ids(dev)
        if ids == (TC001_VID, TC001_PID):
            usb_hits.append(dev)
        if _supports_radiometric(dev):
            size_hits.append(dev)
    if usb_hits:
        return usb_hits[0]
    if size_hits:
        return size_hits[0]
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(
        "No /dev/video* device found. Plug in the camera, check permissions, "
        "or use --simulate."
    )


def open_tc001(device: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device, cv2.CAP_V4L)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {device}")
    # 0, not True/False: some OpenCV builds (notably on Raspberry Pi / ARM) mishandle the bool.
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def split_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a raw 256x384x2 (or packed) TC001 frame into visual BGR + Celsius."""
    if frame.ndim == 2:
        frame = frame.reshape(FRAME_H, FRAME_W, 2)
    if frame.shape[0] != FRAME_H or frame.shape[1] != FRAME_W:
        raise RuntimeError(
            f"Unexpected frame shape {frame.shape}; expected {FRAME_H}x{FRAME_W}. "
            "Is CAP_PROP_CONVERT_RGB disabled, and is this a TC001 in 256x384 mode?"
        )
    visual_raw, thermal_raw = np.array_split(frame, 2, axis=0)
    visual = cv2.cvtColor(visual_raw, cv2.COLOR_YUV2BGR_YUYV)
    temps = raw_to_celsius(thermal_raw)
    return visual, temps


def iter_frames(
    *,
    simulate: bool = False,
    device: int | None = None,
    fps: float = TARGET_FPS,
    sim_config: HouseSimConfig | None = None,
    paced: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
    """Yield (visual_bgr, temps_c, meta) forever.

    meta includes elapsed_s and, in simulate mode, ground-truth masks/boxes.
    """
    if simulate:
        cfg = sim_config or HouseSimConfig()
        started = time.monotonic()
        period = 1.0 / max(fps, 1e-3)
        while True:
            tick = time.monotonic()
            elapsed = tick - started
            visual, temps, meta = simulate_scene(elapsed, cfg)
            meta = dict(meta)
            meta["elapsed_s"] = elapsed
            meta["source"] = "simulate"
            yield visual, temps, meta
            if paced:
                sleep = period - (time.monotonic() - tick)
                if sleep > 0:
                    time.sleep(sleep)
        return

    if sys.platform == "darwin":
        from snug.capture_mac import iter_mac_frames
        yield from iter_mac_frames(device)
        return

    path = find_tc001_device(device)
    cap = open_tc001(path)
    started = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            visual, temps = split_frame(frame)
            yield visual, temps, {"elapsed_s": time.monotonic() - started, "source": path}
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read TC001 frames or run the Snug house simulator.")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--frames", type=int, default=15)
    parser.add_argument("--heat-delay", type=float, default=6.0)
    args = parser.parse_args(argv)
    cfg = HouseSimConfig(heat_delay_s=args.heat_delay)
    gen = iter_frames(simulate=args.simulate, device=args.device, sim_config=cfg)
    for i, (visual, temps, meta) in enumerate(gen, start=1):
        print(
            f"frame {i}: visual={visual.shape} temps={temps.shape} "
            f"min={temps.min():.2f} max={temps.max():.2f} mean={temps.mean():.2f} °C "
            f"source={meta.get('source')}"
        )
        if i >= args.frames:
            break
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
