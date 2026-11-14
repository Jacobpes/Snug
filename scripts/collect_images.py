#!/usr/bin/env python3
"""Capture TC001 (or simulator) frames for Roboflow / Edge Impulse labelling.

Saves:
  data/captures/<stamp>/thermal_XXXX.png   ironbow preview (upload these)
  data/captures/<stamp>/temps_XXXX.npy     radiometric Celsius matrix
  data/captures/<stamp>/manifest.csv       paths + optional auto-tags from the sim

For the physical house, walk around the facade and capture 180–250 frames
(see DATASETS.md / README). Then:

  Roboflow:  create a project, upload the PNGs, draw boxes
             (window, door, wall, roof, corner), export YOLOv8 or COCO,
             or upload to Edge Impulse.

  Edge Impulse: edge-impulse-uploader --category training thermal_*.png
                then label bounding boxes in the EI studio and train FOMO.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from snug.capture import iter_frames  # noqa: E402
from snug.colormap import temps_to_ironbow  # noqa: E402
from snug.simulate import HouseSimConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--interval", type=float, default=0.4, help="Seconds between saved frames.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or (ROOT / "data" / "captures" / stamp)
    out.mkdir(parents=True, exist_ok=True)
    gen = iter_frames(simulate=args.simulate, device=args.device, fps=5, sim_config=HouseSimConfig(jitter=args.simulate))
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "png", "npy", "min_c", "max_c", "auto_boxes"])
        last = 0.0
        saved = 0
        for visual, temps, meta in gen:
            now = time.monotonic()
            if now - last < args.interval:
                continue
            last = now
            saved += 1
            png_path = out / f"thermal_{saved:04d}.png"
            npy_path = out / f"temps_{saved:04d}.npy"
            preview = temps_to_ironbow(temps)
            cv2.imwrite(str(png_path), preview)
            np.save(npy_path, temps)
            boxes = ""
            if "boxes" in meta:
                boxes = ";".join(f"{b['name']}:{b['x']},{b['y']},{b['w']},{b['h']}" for b in meta["boxes"])
            writer.writerow([saved, png_path.name, npy_path.name, f"{temps.min():.2f}", f"{temps.max():.2f}", boxes])
            print(f"saved {png_path.name}  {temps.min():.1f}…{temps.max():.1f} °C")
            if saved >= args.count:
                break
    print(f"Wrote {saved} frames + {manifest}")
    print("Upload PNGs to Roboflow or Edge Impulse. Keep the .npy files for radiometric rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
