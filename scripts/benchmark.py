#!/usr/bin/env python3
"""Benchmark the Snug pipeline on simulated 256x192 frames.

Reports per-stage latency, FPS, and peak RSS. Used both natively and inside
an aarch64 emulation environment.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from snug.analysis import HouseAnalyzer  # noqa: E402
from snug.capture import raw_to_celsius, celsius_to_raw  # noqa: E402
from snug.colormap import temps_to_ironbow  # noqa: E402
from snug.detect import ElementDetector, load_config  # noqa: E402
from snug.overlay import draw_findings  # noqa: E402
from snug.simulate import HouseSimConfig, simulate_scene  # noqa: E402


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--mode", default="hybrid")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = load_config(ROOT / "config.example.json")
    det = ElementDetector(config=cfg, mode=args.mode)
    ana = HouseAnalyzer(cfg)
    sim = HouseSimConfig(heat_delay_s=0, heat_ramp_s=1, seed=0)

    # Pre-generate frames so generation cost is excluded from the pipeline.
    frames = []
    for i in range(args.warmup + args.frames):
        visual, temps, meta = simulate_scene(8.0 + i * 0.04, sim)
        frames.append((visual, temps, meta))

    # Warmup (caches, first onnx/numpy call, etc.)
    for visual, temps, meta in frames[: args.warmup]:
        regions = det.detect(temps, ambient_c=meta["outdoor_c"])
        findings = ana.analyze(temps, regions, ambient_c=meta["outdoor_c"], indoor_c=meta["indoor_c"], now=meta.get("elapsed_s", 0))
        _ = temps_to_ironbow(temps)
        _ = draw_findings(visual, findings, temps)

    # Also time a decode of a packed thermal half
    raw = celsius_to_raw(frames[0][1])

    decode, detect_t, analyze_t, color_t, total = [], [], [], [], []
    t_run = time.perf_counter()
    for visual, temps, meta in frames[args.warmup :]:
        t0 = time.perf_counter()
        _ = raw_to_celsius(raw)
        t1 = time.perf_counter()
        regions = det.detect(temps, ambient_c=meta["outdoor_c"])
        t2 = time.perf_counter()
        findings = ana.analyze(
            temps, regions, ambient_c=meta["outdoor_c"], indoor_c=meta["indoor_c"], now=t2
        )
        t3 = time.perf_counter()
        color = temps_to_ironbow(temps)
        _ = draw_findings(color, findings, temps)
        t4 = time.perf_counter()
        decode.append((t1 - t0) * 1000)
        detect_t.append((t2 - t1) * 1000)
        analyze_t.append((t3 - t2) * 1000)
        color_t.append((t4 - t3) * 1000)
        total.append((t4 - t0) * 1000)
    elapsed = time.perf_counter() - t_run
    fps = args.frames / elapsed

    def stat(xs):
        a = np.asarray(xs, dtype=np.float64)
        return {"mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}

    report = {
        "frames": args.frames,
        "mode": args.mode,
        "model_loaded": det.available,
        "pipeline_fps": float(fps),
        "decode": stat(decode),
        "detect": stat(detect_t),
        "analyze": stat(analyze_t),
        "color_overlay": stat(color_t),
        "total": stat(total),
        "peak_rss_mb": float(rss_mb()),
        "nproc_hint": os.cpu_count(),
        "uname": os.uname().machine,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
