from fastapi.testclient import TestClient

from snug.analysis import HouseAnalyzer
from snug.detect import ElementDetector
from snug.server import Pipeline, create_app, parse_args
from snug.simulate import HouseSimConfig, simulate_scene


def _pipe():
    args = parse_args(["--simulate", "--rois-only"])
    # Build a pipeline but skip the background thread; inject one frame.
    pipe = Pipeline.__new__(Pipeline)
    pipe.args = args
    pipe.cfg = {}
    pipe.sim = HouseSimConfig(heat_delay_s=0, heat_ramp_s=1, seed=1)
    pipe.detector = ElementDetector(config=None, mode="rois")
    pipe.analyzer = HouseAnalyzer({"reference_roi": "insulated_wall"})
    from snug.cost import CostSettings, KitSettings
    pipe.cost_settings = CostSettings()
    pipe.kit_settings = KitSettings()
    pipe.cost_total = {}
    from snug.bridge import AlarmBridge
    import threading, time
    import numpy as np

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
    visual, temps, meta = simulate_scene(20.0, pipe.sim)
    pipe.process_frame(visual, temps, meta)
    return pipe


def test_health_state_snapshot_and_verify():
    pipe = _pipe()
    app = create_app(pipe)
    client = TestClient(app)
    h = client.get("/api/health")
    assert h.status_code == 200 and h.json()["ok"] is True
    st = client.get("/api/state").json()
    assert st["brand"] == "Snug"
    assert st["findings"]
    assert "element" in st["answers"][0]
    assert "cost" in st
    assert "cost_settings" in st
    assert "kit" in st
    prices = [item["price_eur"] for item in st["kit"]["items"]]
    assert prices[0] == 225.0
    assert abs(prices[1] - 59.90) < 1e-9
    assert prices[2] == 0
    assert "payback" in st["kit"]
    assert "cost" in st["answers"][0] or st["findings"][0].get("cost_short") is not None
    cost_api = client.get("/api/cost")
    assert cost_api.status_code == 200
    body = cost_api.json()
    assert "vaasa" in {l["id"] for l in body["catalogue"]["localities"]}
    posted = client.post("/api/cost/settings", json={"locality": "pietarsaari", "heating_type": "electric"})
    assert posted.status_code == 200 and posted.json()["ok"] is True
    assert posted.json()["settings"]["locality"] == "pietarsaari"
    snap = client.get("/api/snapshot.jpg")
    assert snap.status_code == 200
    assert snap.headers["content-type"].startswith("image/jpeg")
    assert len(snap.content) > 100
    before = client.post("/api/verify/before")
    assert before.status_code == 200 and before.json()["ok"] is True
    after = client.post("/api/verify/after")
    assert after.status_code == 200
    assert "passed" in after.json()


def test_index_served():
    pipe = _pipe()
    app = create_app(pipe)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert b"SNUG" in r.content
