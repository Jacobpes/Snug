#!/usr/bin/env python3
"""Train the tiny FOMO detector on synthetic model-house frames and export weights.

The resulting model is trained on *simulated* TC001-sized frames only. Fine-tune
on real TC001 images of the physical house before trusting it on stage
(see scripts/collect_images.py and README).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from snug.nn import CLASS_NAMES, GRID_H, GRID_W, INPUT_H, INPUT_W, CELL, temps_to_input  # noqa: E402
from snug.simulate import CLASS_ID, HouseSimConfig, simulate_scene  # noqa: E402


def labels_to_grid(label_map: np.ndarray) -> np.ndarray:
    """Majority class per 16x16 cell."""
    grid = np.zeros((GRID_H, GRID_W), dtype=np.int64)
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            cell = label_map[gy * CELL : (gy + 1) * CELL, gx * CELL : (gx + 1) * CELL]
            # majority, but prefer non-background if it covers >= 25% of the cell
            counts = np.bincount(cell.ravel(), minlength=len(CLASS_NAMES))
            if counts[1:].sum() >= (CELL * CELL) * 0.25:
                counts[0] = 0
            grid[gy, gx] = int(np.argmax(counts))
    return grid


def make_batch(n: int, rng: np.random.Generator, trained_warmup: bool = True):
    xs, ys = [], []
    for _ in range(n):
        elapsed = float(rng.uniform(4.0, 28.0 if trained_warmup else 40.0))
        cfg = HouseSimConfig(
            outdoor_c=float(rng.uniform(14.0, 22.0)),
            indoor_max_c=float(rng.uniform(30.0, 40.0)),
            heat_delay_s=float(rng.uniform(2.0, 8.0)),
            heat_ramp_s=float(rng.uniform(8.0, 18.0)),
            noise_c=float(rng.uniform(0.10, 0.28)),
            insulation_gap=bool(rng.random() > 0.2),
            thermal_bridge=bool(rng.random() > 0.2),
            air_leak=bool(rng.random() > 0.2),
            seed=int(rng.integers(0, 10_000_000)),
            jitter=True,
        )
        _visual, temps, meta = simulate_scene(elapsed, cfg, rng=rng)
        xs.append(temps_to_input(temps, meta["outdoor_c"]))
        ys.append(labels_to_grid(meta["labels"]))
    return np.stack(xs), np.stack(ys)


def export_onnx(model, path: Path) -> None:
    import torch

    dummy = torch.zeros(1, 1, INPUT_H, INPUT_W)
    # Torch 2.15 defaults to a dynamo exporter that needs onnxscript.
    # The classic exporter is enough for this tiny static graph.
    try:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["temps"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["temps"],
            output_names=["logits"],
            opset_version=17,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batches-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    args = parser.parse_args(argv)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyFomoTorch(nn.Module):
        def __init__(self, n_classes: int = len(CLASS_NAMES)):
            super().__init__()
            self.c1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
            self.c2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
            self.c3 = nn.Conv2d(16, 24, 3, stride=2, padding=1)
            self.c4 = nn.Conv2d(24, 32, 3, stride=2, padding=1)
            self.head = nn.Conv2d(32, n_classes, 1)

        def forward(self, x):
            x = F.relu(self.c1(x))
            x = F.relu(self.c2(x))
            x = F.relu(self.c3(x))
            x = F.relu(self.c4(x))
            return self.head(x)

    device = torch.device("cpu")
    model = TinyFomoTorch().to(device)
    # class weights: background is common
    weight = torch.tensor([0.2, 1.0, 2.0, 2.0, 1.4, 1.6], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(42)

    print(f"Training TinyFOMO on synthetic {INPUT_W}x{INPUT_H} frames…")
    for epoch in range(1, args.epochs + 1):
        losses = []
        accs = []
        for _ in range(args.batches_per_epoch):
            xb, yb = make_batch(args.batch_size, rng)
            xt = torch.from_numpy(xb)
            yt = torch.from_numpy(yb)
            logits = model(xt)
            loss = F.cross_entropy(logits, yt, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pred = logits.argmax(1)
            accs.append(float((pred == yt).float().mean()))
            losses.append(float(loss.item()))
        print(f"epoch {epoch:02d}  loss={np.mean(losses):.4f}  acc={np.mean(accs):.3f}")

    # quick held-out check
    xb, yb = make_batch(64, np.random.default_rng(123))
    with torch.no_grad():
        pred = model(torch.from_numpy(xb)).argmax(1).numpy()
    acc = float((pred == yb).mean())
    # per-class recall excluding bg
    recalls = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = yb == i
        recalls[name] = float((pred[mask] == i).mean()) if mask.any() else None
    print(f"held-out grid acc={acc:.3f}  recalls={recalls}")

    args.out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    # Map torch names to the numpy inference keys
    npz = {
        "c1_w": state["c1.weight"],
        "c1_b": state["c1.bias"],
        "c2_w": state["c2.weight"],
        "c2_b": state["c2.bias"],
        "c3_w": state["c3.weight"],
        "c3_b": state["c3.bias"],
        "c4_w": state["c4.weight"],
        "c4_b": state["c4.bias"],
        "head_w": state["head.weight"],
        "head_b": state["head.bias"],
    }
    npz_path = args.out / "detector.npz"
    np.savez(npz_path, **npz)
    meta = {
        "trained_on": "synthetic_model_house",
        "input": [1, INPUT_H, INPUT_W],
        "grid": [GRID_H, GRID_W],
        "classes": list(CLASS_NAMES),
        "heldout_grid_acc": acc,
        "recalls": recalls,
        "note": "Must be fine-tuned on real TC001 images of the physical demo house.",
    }
    (args.out / "detector_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    onnx_path = args.out / "detector.onnx"
    try:
        export_onnx(model, onnx_path)
        print(f"wrote {npz_path} and {onnx_path}")
    except Exception as exc:
        print(f"wrote {npz_path}; ONNX export failed ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
