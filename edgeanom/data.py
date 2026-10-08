"""VisA dataset loading (Amazon, CC BY 4.0).

Expects the extracted VisA_20220922.tar: one folder per category plus
split_csv/1cls.csv with columns object,split,label,image,mask.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 224


@dataclass
class Sample:
    image: Path
    mask: Path | None
    label: int  # 0 = normal, 1 = anomaly


def find_split_csv(root: Path) -> Path:
    direct = root / "split_csv" / "1cls.csv"
    if direct.exists():
        return direct
    hits = list(root.rglob("1cls.csv"))
    if not hits:
        raise FileNotFoundError(f"split_csv/1cls.csv not found under {root}")
    return hits[0]


def load_split(root: str | Path, category: str) -> tuple[list[Sample], list[Sample]]:
    root = Path(root)
    csv_path = find_split_csv(root)
    base = csv_path.parent.parent
    train, test = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["object"] != category:
                continue
            label = 0 if row["label"].strip().lower() == "normal" else 1
            mask = base / row["mask"] if row.get("mask") else None
            s = Sample(base / row["image"], mask, label)
            (train if row["split"].strip().lower() == "train" else test).append(s)
    if not train or not test:
        raise ValueError(f"No samples for category '{category}' in {csv_path}")
    return train, test


def load_image(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    """RGB image -> float32 CHW in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


def load_mask(path: Path | None, size: int = IMG_SIZE) -> np.ndarray:
    if path is None or not Path(path).exists():
        return np.zeros((size, size), dtype=np.uint8)
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 0).astype(np.uint8)
