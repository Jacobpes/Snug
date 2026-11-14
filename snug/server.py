#!/usr/bin/env python3
"""Snug dashboard: local FastAPI app, no cloud calls."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snug.analysis import SENSITIVITY_OPTIONS, SEVERITY_RANK, SENSITIVITY_PRESETS, HouseAnalyzer, RegionFinding, normalize_sensitivity
from snug.bridge import AlarmBridge
from snug.capture import iter_frames
from snug.colormap import temps_to_ironbow
from snug.cost import CostSettings, KitSettings, attach_costs, catalogue
from snug.detect import ElementDetector, load_config
from snug.overlay import draw_findings
from snug.simulate import HouseSimConfig

STATIC = ROOT / "static"


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        cfg_path = Path(args.config) if args.config else ROOT / "config.example.json"
        self.cfg = load_config(cfg_path) if cfg_path.is_file() else {}
        if args.rois_only:
            self.cfg["detect_mode"] = "rois"
        sim_over = self.cfg.get("sim") or {}
        self.sim = HouseSimConfig(
            outdoor_c=float(sim_over.get("outdoor_c", 18.0)),
            indoor_max_c=float(sim_over.get("indoor_max_c", 36.0)),
            heat_delay_s=float(args.heat_delay or sim_over.get("heat_delay_s", 6.0)),
            heat_ramp_s=float(sim_over.get("heat_ramp_s", 14.0)),
            insulation_gap=bool((self.cfg.get("defects") or {}).get("insulation_gap", True)) and not args.no_defects,
            thermal_bridge=bool((self.cfg.get("defects") or {}).get("thermal_bridge", True)) and not args.no_defects,
            air_leak=bool((self.cfg.get("defects") or {}).get("air_leak", True)) and not args.no_defects,
        )
        self.detector = ElementDetector(config=self.cfg, mode=self.cfg.get("detect_mode", "hybrid"))
        # Sensitivity: CLI > config. The model-house simulator defaults to "normal" (its
        # deltas are huge and "high" would flag the roof gradient); live cameras use config ("high").
        sens = getattr(args, "sensitivity", None)
        if sens is None:
            sens = "normal" if args.simulate else self.cfg.get("sensitivity", "high")
        self.cfg["sensitivity"] = normalize_sensitivity(sens)
        self.analyzer = HouseAnalyzer(self.cfg)
        self.cost_settings = CostSettings.from_config(self.cfg)
        self.kit_settings = KitSettings.from_config(self.cfg)
        if getattr(args, "locality", None):
            self.cost_settings = self.cost_settings.updated({"locality": args.locality})
        if getattr(args, "heating_type", None):
            self.cost_settings = self.cost_settings.updated({"heating_type": args.heating_type})
        if getattr(args, "t_in", None) is not None:
            self.cost_settings = self.cost_settings.updated({"t_in_c": args.t_in, "t_in_source": "manual"})
        if getattr(args, "t_out", None) is not None:
            self.cost_settings = self.cost_settings.updated({"t_out_c": args.t_out})
        if getattr(args, "price", None) is not None:
            self.cost_settings = self.cost_settings.updated({"price_eur_per_kwh": args.price})
        self.cost_total: dict[str, Any] = {}
        self.bridge = AlarmBridge(enabled=args.bridge)
        self.args = args
        self.lock = threading.Lock()
        self.visual = np.zeros((192, 256, 3), dtype=np.uint8)
        self.overlay = self.visual.copy()
        self.temps = np.zeros((192, 256), dtype=np.float32)
        self.findings: list[RegionFinding] = []
        self.max_findings: int = 1  # show only the most serious defect
        self.meta: dict[str, Any] = {}
        self.jpeg = b""
        self.fps = 0.0
        self.latencies: dict[str, float] = {}
        self.started = time.time()
        self.running = True
        self.frame_id = 0

    def process_frame(self, visual: np.ndarray, temps: np.ndarray, meta: dict[str, Any]) -> None:
        t0 = time.perf_counter()
        ambient = meta.get("outdoor_c")
        indoor = meta.get("indoor_c")
        t1 = time.perf_counter()
        regions = self.detector.detect(temps, ambient_c=ambient)
        t2 = time.perf_counter()
        findings = self.analyzer.analyze(
            temps, regions, ambient_c=ambient, indoor_c=indoor, now=meta.get("elapsed_s") or time.monotonic()
        )
        t_in, t_out, scan = self._scan_temps(indoor, ambient)
        settings = self._settings_for_scan(scan)
        if scan.get("scan_mode") == "live_interior":
            # Seen from inside, the coldest area is where heat escapes: always consider it.
            cold = self.analyzer.coldest_area_finding(temps, indoor_c=indoor)
            if cold is not None:
                findings = findings + [cold]
        attach_costs(findings, settings, t_in=t_in, t_out=t_out)
        findings = self._pick_most_serious(findings, prefer_cold=scan.get("scan_mode") == "live_interior")
        cost_total = attach_costs(findings, settings, t_in=t_in, t_out=t_out)
        cost_total.update(scan)
        t3 = time.perf_counter()
        color = temps_to_ironbow(temps)
        overlay = draw_findings(color, findings, temps, total=cost_total)
        ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        jpeg = buf.tobytes() if ok else b""
        t4 = time.perf_counter()
        with self.lock:
            self.visual = color
            self.overlay = overlay
            self.temps = temps
            self.findings = findings
            self.cost_total = cost_total
            self.meta = meta
            self.jpeg = jpeg
            self.latencies = {
                "detect_ms": (t2 - t1) * 1000.0,
                "analyze_ms": (t3 - t2) * 1000.0,
                "encode_ms": (t4 - t3) * 1000.0,
                "total_ms": (t4 - t0) * 1000.0,
            }
            self.frame_id += 1
        self.bridge.set_from_findings(findings)

    @staticmethod
    def _seriousness(f: RegionFinding) -> float:
        cost = (f.cost or {}).get("cost_eur_high") or 0.0
        return SEVERITY_RANK.get(f.severity, 0) * 1e6 + float(cost) * 100.0 + float(f.priority)

    def _pick_most_serious(self, findings: list[RegionFinding], prefer_cold: bool = False) -> list[RegionFinding]:
        """Show only the single most serious defect (severity first, then €/yr, then priority).

        Small hysteresis: keep the previously shown region while it is still at least 80 % as
        serious as the best one, so the flag does not jump between spots every frame.
        """
        defects = [f for f in findings if not f.normal]
        if prefer_cold and any(f.direction == "cold" for f in defects):
            # Interior scan: cold spots are the leaks; warm spots are heat sources (people, radiators).
            defects = [f for f in defects if f.direction == "cold"]
            tight = [f for f in defects if f.name == "coldest_area"]
            if tight:
                # The tight blob around the coldest pixels outlines the leak better than a whole wall ROI.
                defects = tight
        limit = int(getattr(self, "max_findings", 1))
        if not defects or limit <= 0:
            return []
        defects.sort(key=self._seriousness, reverse=True)
        best = defects[0]
        prev = getattr(self, "_shown_name", None)
        for f in defects:
            if f.name == prev and f.severity == best.severity and self._seriousness(f) >= 0.8 * self._seriousness(best):
                best = f
                break
        self._shown_name = best.name
        rest = [f for f in defects if f is not best]
        return ([best] + rest)[:limit]

    def _settings_for_scan(self, scan: dict[str, Any]) -> CostSettings:
        """A live scan is by definition an interior survey (cold spots), so pin the side."""
        cs = self.cost_settings
        if scan.get("scan_mode") == "live_interior" and (cs.survey_side or "auto").lower() == "auto":
            return cs.updated({"survey_side": "interior"})
        return cs

    def _scan_temps(self, indoor: float | None, outdoor: float | None) -> tuple[float, float, dict[str, Any]]:
        """Temperatures used to estimate U, plus where they came from.

        Simulator / model house (meta has indoor_c): measured sim values unless the user overrode t_out.
        Live camera (no meta): interior survey. T_in is the measured scene temperature (robust frame
        median) unless the user pinned t_in; T_out is the user's t_out or the Finnish heating-season
        default (cost.default_t_out_c). The frame border is never used as outdoor temperature.
        """
        cs = self.cost_settings
        if indoor is not None:
            t_in = float(indoor)
            if cs.t_out_c is not None:
                t_out, t_src = float(cs.t_out_c), "setting"
            elif outdoor is not None:
                t_out, t_src = float(outdoor), "simulator"
            else:
                t_out, t_src = t_in - 20.0, "placeholder"
            return t_in, t_out, {"scan_mode": "simulator", "t_in_source": "simulator", "t_out_source": t_src}
        scene = self.analyzer.scene_c
        if cs.t_in_source == "manual" or scene is None:
            t_in, in_src = float(cs.t_in_c), "manual" if cs.t_in_source == "manual" else "setting_default"
        else:
            t_in, in_src = float(scene), "scene_median"
        if cs.t_out_c is not None:
            t_out, out_src = float(cs.t_out_c), "setting"
        else:
            t_out, out_src = float(cs.default_t_out_c), "default"
        return t_in, t_out, {"scan_mode": "live_interior", "t_in_source": in_src, "t_out_source": out_src}

    def apply_sensitivity(self, value: Any) -> str:
        with self.lock:
            return self.analyzer.set_sensitivity(value)

    def apply_cost_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        patch = dict(patch or {})
        if patch.get("sensitivity") not in (None, ""):
            self.apply_sensitivity(patch["sensitivity"])
        self.cost_settings = self.cost_settings.updated(patch)
        with self.lock:
            findings = list(self.findings)
            indoor = self.meta.get("indoor_c")
            outdoor = self.meta.get("outdoor_c")
        if findings:
            t_in, t_out, scan = self._scan_temps(indoor, outdoor)
            total = attach_costs(findings, self._settings_for_scan(scan), t_in=t_in, t_out=t_out)
            total.update(scan)
            with self.lock:
                self.findings = findings
                self.cost_total = total
        return self.cost_settings.to_public()

    def loop(self) -> None:
        gen = iter_frames(
            simulate=self.args.simulate,
            device=self.args.device,
            fps=self.args.fps,
            sim_config=self.sim,
            paced=True,
        )
        last = time.perf_counter()
        ema = None
        for visual, temps, meta in gen:
            if not self.running:
                break
            self.process_frame(visual, temps, meta)
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            ema = inst if ema is None else (0.8 * ema + 0.2 * inst)
            self.fps = float(ema)
            last = now

    def state(self) -> dict[str, Any]:
        with self.lock:
            findings = [f.to_dict() for f in self.findings]
            six = [f.six_answers() | {"name": f.name} for f in self.findings]
            meta = {
                "indoor_c": self.meta.get("indoor_c"),
                "outdoor_c": self.meta.get("outdoor_c"),
                "source": self.meta.get("source"),
                "elapsed_s": self.meta.get("elapsed_s"),
                "scene_c": None if self.analyzer.scene_c is None else round(float(self.analyzer.scene_c), 2),
                "scan_mode": "live_interior" if self.analyzer.live else "simulator",
            }
            sensitivity = self.analyzer.sensitivity
            verify = self.analyzer.last_verification
            has_before = self.analyzer.before is not None
            frame_id = self.frame_id
            fps = self.fps
            lat = dict(self.latencies)
            cost_total = dict(self.cost_total)
            cost_settings = self.cost_settings.to_public()
            kit = self.kit_settings.to_public(cost_total)
        return {
            "brand": "Snug",
            "frame_id": frame_id,
            "fps": round(fps, 2),
            "latencies_ms": {k: round(v, 2) for k, v in lat.items()},
            "detector": {"mode": self.detector.mode, "model": self.detector.available},
            "bridge": {"enabled": self.bridge.enabled, "active": self.bridge.active},
            "meta": meta,
            "findings": findings,
            "ranked": findings,
            "answers": six,
            "cost": cost_total,
            "cost_settings": cost_settings,
            "sensitivity": sensitivity,
            "sensitivity_options": list(SENSITIVITY_OPTIONS),
            "sensitivity_presets": {k: dict(v) for k, v in SENSITIVITY_PRESETS.items()},
            "cost_catalogue": catalogue(),
            "kit": kit,
            "verify": verify,
            "has_before": has_before,
        }


def create_app(pipe: Pipeline) -> FastAPI:
    app = FastAPI(title="Snug", docs_url=None, redoc_url=None)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "brand": "Snug", "frames": pipe.frame_id, "fps": pipe.fps}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return pipe.state()

    @app.get("/api/snapshot.jpg")
    def snapshot() -> Response:
        with pipe.lock:
            data = pipe.jpeg
        return Response(data, media_type="image/jpeg")

    @app.get("/api/stream.mjpeg")
    def stream() -> StreamingResponse:
        def gen():
            last = -1
            while pipe.running:
                with pipe.lock:
                    fid = pipe.frame_id
                    data = pipe.jpeg
                if fid != last and data:
                    last = fid
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                else:
                    time.sleep(0.02)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/verify/before")
    def verify_before() -> dict[str, Any]:
        with pipe.lock:
            snap = pipe.analyzer.mark_before(pipe.findings)
        return {"ok": True, "regions": snap}

    @app.post("/api/verify/after")
    def verify_after() -> JSONResponse:
        try:
            with pipe.lock:
                report = pipe.analyzer.compare_after(pipe.findings)
        except RuntimeError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, **report})

    @app.get("/api/verify")
    def verify_get() -> dict[str, Any]:
        return {"has_before": pipe.analyzer.before is not None, "report": pipe.analyzer.last_verification}

    @app.get("/api/cost")
    def cost_get() -> dict[str, Any]:
        return {
            "settings": pipe.cost_settings.to_public(),
            "total": pipe.cost_total,
            "catalogue": catalogue(),
            "kit": pipe.kit_settings.to_public(pipe.cost_total),
        }

    @app.post("/api/cost/settings")
    def cost_settings(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        payload = payload or {}
        try:
            settings = pipe.apply_cost_settings(payload)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc), "options": list(SENSITIVITY_OPTIONS)}, status_code=400)
        return {
            "ok": True,
            "settings": settings,
            "total": pipe.cost_total,
            "sensitivity": pipe.analyzer.sensitivity,
            "sensitivity_options": list(SENSITIVITY_OPTIONS),
        }

    if STATIC.is_dir():
        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC / "index.html")

        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Snug local thermal dashboard")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--config", type=str, default=str(ROOT / "config.example.json"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--heat-delay", type=float, default=None)
    p.add_argument("--no-defects", action="store_true", help="Simulate a fully sound house (for before/after).")
    p.add_argument("--rois-only", action="store_true")
    p.add_argument("--bridge", action="store_true", help="Enable optional Arduino App Lab Bridge alarm.")
    p.add_argument("--locality", type=str, default=None, help="Finnish climate locality (e.g. vaasa, pietarsaari, helsinki).")
    p.add_argument("--heating-type", type=str, default=None, help="electric, heat_pump, ground_source, district, oil.")
    p.add_argument("--t-in", type=float, default=None, help="Pin the indoor temperature (°C). Default for a live camera: measured scene temperature.")
    p.add_argument("--t-out", type=float, default=None, help="Outdoor temperature (°C). Default: simulator value, or cost.default_t_out_c (0 °C) for a live camera.")
    p.add_argument(
        "--sensitivity",
        choices=list(SENSITIVITY_OPTIONS),
        default=None,
        help="Detection sensitivity: low (strict demo-house thresholds), normal, high (live camera default).",
    )
    p.add_argument("--price", type=float, default=None, help="Override energy price in €/kWh.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pipe = Pipeline(args)
    thread = threading.Thread(target=pipe.loop, name="snug-capture", daemon=True)
    thread.start()
    # Wait for first frame so /api/snapshot is not empty
    for _ in range(100):
        if pipe.frame_id:
            break
        time.sleep(0.05)
    app = create_app(pipe)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        pipe.running = False


if __name__ == "__main__":
    main()
