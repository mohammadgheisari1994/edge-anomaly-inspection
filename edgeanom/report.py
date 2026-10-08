"""Results table, latency chart and example heatmaps for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _fmt(v, f="{:.1f}"):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def write_report(out: Path, meta, rows, test, images, masks, scores, maps):
    base = rows[0]["mean_ms"]
    lines = [f"# Results - VisA `{meta['category']}`", "",
             f"Machine: {meta['machine']} | threads: {meta['threads']} | batch 1 | input {meta['input']} | "
             f"backbone {meta['backbone']} | memory bank {meta['memory_bank']} patches", "",
             "| Backend | Mean latency (ms) | p95 (ms) | FPS | Speed-up | Image AUROC | Pixel AUROC | Model size (MB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['backend']} | {_fmt(r['mean_ms'])} | {_fmt(r['p95_ms'])} | {_fmt(r['fps'])} | "
                     f"{_fmt(base / r['mean_ms'], '{:.2f}x')} | {_fmt(r['image_auroc'], '{:.3f}')} | "
                     f"{_fmt(r['pixel_auroc'], '{:.3f}')} | {_fmt(r['model_mb'])} |")
    lines += ["", "Latency = model only (image in, patch scores out), mean of "
              "repeated single-image runs after warm-up. Speed-up is relative to the first row.", ""]
    (out / "results.md").write_text("\n".join(lines))

    # latency chart
    names = [r["backend"] for r in rows]
    ms = [r["mean_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.2))
    ax.barh(names[::-1], ms[::-1], color="#2f6f9f")
    for i, v in enumerate(ms[::-1]):
        ax.text(v, i, f" {v:.0f} ms", va="center", fontsize=9)
    ax.set_xlabel("Latency per image (ms, lower is better)")
    ax.set_title(f"PatchCore on VisA {meta['category']} - {meta['machine']}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=150)
    plt.close(fig)

    # example heatmaps: anomalies spread across the score ranking (not just the
    # easiest ones) + a typical (median-score) normal part
    order = np.argsort(scores)[::-1]
    anom = [i for i in order if test[i].label == 1]
    norm = [i for i in order if test[i].label == 0]
    picks = [anom[int(q * (len(anom) - 1))] for q in (0.0, 0.2, 0.4, 0.6)] if len(anom) >= 4 else anom
    picks = list(dict.fromkeys(picks)) + ([norm[len(norm) // 2]] if norm else [])
    # colour scale: median normal-part value = blue, top 0.5% of all values = red
    normal_vals = np.stack([maps[i] for i in norm]) if norm else np.stack(maps)
    vmin = float(np.percentile(normal_vals, 50))
    vmax = float(np.percentile(np.stack(maps), 99.5))
    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for c, i in enumerate(picks):
        img = images[i].transpose(1, 2, 0)
        axes[0, c].imshow(img)
        if masks[i].any():
            axes[0, c].contour(masks[i], levels=[0.5], colors="lime", linewidths=1)
        axes[0, c].set_title("defect" if test[i].label else "normal", fontsize=10)
        axes[1, c].imshow(img)
        axes[1, c].imshow(maps[i], cmap="jet", alpha=0.45, vmin=vmin, vmax=vmax)
        axes[1, c].set_title(f"score {scores[i]:.2f}", fontsize=10)
        for a in axes[:, c]:
            a.axis("off")
    fig.suptitle("Top: input (green = ground-truth defect)   Bottom: predicted anomaly map", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=130)
    plt.close(fig)
