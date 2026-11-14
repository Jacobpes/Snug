"""macOS capture for the Topdon TC001.

macOS has no /dev/video*. We try two routes to get the raw 256x384 YUYV frame
(visual top half, radiometric bottom half):

1. OpenCV AVFoundation with CAP_PROP_CONVERT_RGB=0. Works only if the build
   hands back the raw buffer (shape 384x256x2 or a flat byte buffer).
2. ffmpeg (from imageio-ffmpeg, or ffmpeg on PATH) reading avfoundation with
   -pixel_format yuyv422 -video_size 256x384, piped as rawvideo. This is the
   reliable route.

The terminal app running this needs camera permission
(System Settings > Privacy & Security > Camera).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from typing import Iterator

import cv2
import numpy as np

FRAME_W, FRAME_H = 256, 384
FRAME_BYTES = FRAME_W * FRAME_H * 2


def _ffmpeg_bin() -> str | None:
    env = os.environ.get("SNUG_FFMPEG")
    if env:
        return env
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if found:
        try:
            subprocess.run([found, "-version"], capture_output=True, check=True, timeout=10)
            return found
        except Exception:
            return None
    return None


def list_avfoundation_devices(ffmpeg: str) -> list[tuple[int, str]]:
    out = subprocess.run(
        [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True, timeout=20,
    ).stderr
    devices: list[tuple[int, str]] = []
    in_video = False
    for line in out.splitlines():
        if "video devices" in line.lower():
            in_video = True
            continue
        if "audio devices" in line.lower():
            break
        m = re.search(r"\[(\d+)\]\s+(.*)$", line)
        if in_video and m:
            devices.append((int(m.group(1)), m.group(2).strip()))
    return devices


def pick_tc001_index(devices: list[tuple[int, str]]) -> int | None:
    for idx, name in devices:
        n = name.lower()
        if "uvc camera" in n or "usb" in n or "topdon" in n or "thermal" in n or "tc001" in n:
            return idx
    for idx, name in devices:
        n = name.lower()
        if "facetime" not in n and "iphone" not in n and "capture screen" not in n and "desk view" not in n:
            return idx
    return None


def _iter_ffmpeg(ffmpeg: str, index: int) -> Iterator[np.ndarray]:
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "avfoundation", "-pixel_format", "yuyv422",
        "-video_size", f"{FRAME_W}x{FRAME_H}", "-framerate", "25",
        "-i", f"{index}:none",
        "-f", "rawvideo", "-pix_fmt", "yuyv422", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=FRAME_BYTES * 2)
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(FRAME_BYTES)
            if len(buf) < FRAME_BYTES:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                raise RuntimeError(f"ffmpeg stopped: {err.strip() or 'no data'}")
            yield np.frombuffer(buf, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)
    finally:
        proc.kill()


def _try_opencv_raw(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    for _ in range(5):
        ok, f = cap.read()
    if ok and f is not None:
        if f.shape[:2] == (FRAME_H, FRAME_W) and (f.ndim == 3 and f.shape[2] == 2):
            return cap
        if f.size == FRAME_BYTES:
            return cap
    cap.release()
    return None


def _find_index(ffmpeg: str) -> int | None:
    for attempt in range(10):
        devs = list_avfoundation_devices(ffmpeg)
        index = pick_tc001_index(devs)
        print(f"[snug] avfoundation video devices: {devs} -> using {index}")
        if index is not None:
            return index
        time.sleep(1.0)
    return None


def iter_mac_frames(device: int | None) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
    from snug.capture import split_frame

    ffmpeg = _ffmpeg_bin()
    started = time.monotonic()

    if os.environ.get("SNUG_MAC_OPENCV"):
        index = device if device is not None else 0
        cap = _try_opencv_raw(index)
        if cap is not None:
            print("[snug] using OpenCV raw capture")
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        time.sleep(0.05)
                        continue
                    visual, temps = split_frame(frame.reshape(FRAME_H, FRAME_W, 2))
                    yield visual, temps, {"elapsed_s": time.monotonic() - started, "source": f"avfoundation:{index}"}
            finally:
                cap.release()
            return

    if not ffmpeg:
        raise RuntimeError("No working ffmpeg found. Run: pip install imageio-ffmpeg")

    failures = 0
    while True:
        index = device if device is not None else _find_index(ffmpeg)
        if index is None:
            raise RuntimeError("TC001 not found among cameras. Replug the USB-C cable and restart.")
        print(f"[snug] using ffmpeg avfoundation device {index}")
        try:
            for frame in _iter_ffmpeg(ffmpeg, index):
                failures = 0
                visual, temps = split_frame(frame)
                yield visual, temps, {"elapsed_s": time.monotonic() - started, "source": f"avfoundation:{index}"}
        except RuntimeError as e:
            failures += 1
            print(f"[snug] capture error ({failures}): {str(e)[-300:]}")
            if failures >= 10:
                raise
            time.sleep(1.5)
