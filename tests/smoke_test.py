"""Runs the full pipeline on a tiny synthetic dataset laid out like VisA.

Uses random (not ImageNet) weights, so the AUROC numbers are meaningless;
it only proves every stage runs end to end.

    python tests/smoke_test.py
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeanom.pipeline import main  # noqa: E402


def make_fake_visa(root: Path, cat: str = "fakepcb", n_train=12, n_norm=4, n_anom=4):
    rng = np.random.default_rng(0)
    rows = []
    for split, label, n in [("train", "normal", n_train), ("test", "normal", n_norm), ("test", "anomaly", n_anom)]:
        for i in range(n):
            img = (rng.normal(120, 10, (300, 300, 3))).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(img)
            rel_img = f"{cat}/Data/Images/{label.title()}/{split}_{i:03d}.JPG"
            mask_rel = ""
            if label == "anomaly":
                x, y = rng.integers(30, 220, 2)
                ImageDraw.Draw(im).rectangle([x, y, x + 50, y + 50], fill=(250, 20, 20))
                m = Image.new("L", (300, 300), 0)
                ImageDraw.Draw(m).rectangle([x, y, x + 50, y + 50], fill=255)
                mask_rel = f"{cat}/Data/Masks/Anomaly/{split}_{i:03d}.png"
                (root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
                m.save(root / mask_rel)
            (root / rel_img).parent.mkdir(parents=True, exist_ok=True)
            im.save(root / rel_img)
            rows.append([cat, split, label, rel_img, mask_rel])
    (root / "split_csv").mkdir(parents=True, exist_ok=True)
    with open(root / "split_csv" / "1cls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "split", "label", "image", "mask"])
        w.writerows(rows)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_fake_visa(d / "VisA")
        main(["--visa-root", str(d / "VisA"), "--category", "fakepcb", "--out", str(d / "results"),
              "--no-pretrained", "--backbone", "resnet18", "--max-train", "12",
              "--coreset-ratio", "0.1", "--calib", "4", "--warmup", "1", "--runs", "3"])
        out = d / "results" / "fakepcb"
        for f in ["results.json", "results.md", "latency.png", "examples.png",
                  "patchcore_fp32.onnx", "patchcore_int8.onnx"]:
            assert (out / f).exists(), f"missing {f}"
        print((out / "results.md").read_text())
        print("SMOKE TEST PASSED")
