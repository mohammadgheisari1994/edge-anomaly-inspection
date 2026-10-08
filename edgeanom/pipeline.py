"""End-to-end run: fit -> export ONNX -> INT8 quantize -> evaluate + benchmark -> report.

    python -m edgeanom.pipeline --visa-root data/VisA --category pcb1
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from .data import IMG_SIZE, load_image, load_mask, load_split
from .model import FeatureExtractor, PatchCore, greedy_coreset


def machine_name() -> str:
    if platform.system() == "Darwin":
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
            if chip:
                return f"{chip} (macOS {platform.mac_ver()[0]})"
        except OSError:
            pass
    return f"{platform.system()} {platform.machine()}"


# ---------------------------------------------------------------- fitting
def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def fit(train, args, device: str) -> PatchCore:
    extractor = FeatureExtractor(args.backbone, pretrained=not args.no_pretrained).to(device)
    samples = train if len(train) <= args.max_train else random.Random(0).sample(train, args.max_train)
    feats, row = None, 0  # preallocated: avoids a second full copy from torch.cat
    for i in range(0, len(samples), args.batch):
        x = torch.from_numpy(np.stack([load_image(s.image) for s in samples[i:i + args.batch]]))
        f = extractor(x.to(device)).cpu()
        if feats is None:
            per_img = f.shape[1]
            feats = torch.empty(len(samples) * per_img, f.shape[-1])
        f = f.reshape(-1, f.shape[-1])
        feats[row:row + f.shape[0]] = f
        row += f.shape[0]
    n_keep = max(1, int(feats.shape[0] * args.coreset_ratio))
    t0 = time.time()
    idx = greedy_coreset(feats, n_keep)
    print(f"  memory bank: {len(samples)} images, {feats.shape[0]} patches -> "
          f"{n_keep} coreset ({time.time() - t0:.0f}s)")
    grid = int(round(per_img ** 0.5))
    return PatchCore(extractor.cpu(), feats[idx], grid=grid).eval()


# ----------------------------------------------------------------- export
def export_onnx(model: PatchCore, path: Path) -> None:
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    kwargs = dict(input_names=["image"], output_names=["patch_scores"],
                  dynamic_axes={"image": {0: "batch"}, "patch_scores": {0: "batch"}},
                  opset_version=17, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, dummy, str(path), **kwargs)


def quantize_int8(fp32: Path, out: Path, calib, n_calib: int) -> None:
    """Static INT8 (QDQ) on the backbone convolutions only.

    The nearest-neighbour distance stays in FP32 to protect accuracy:
    anomaly scores are small differences between distances.
    """
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib[:n_calib])

        def get_next(self):
            s = next(self.it, None)
            return None if s is None else {"image": load_image(s.image)[None]}

    prepped = out.with_suffix(".prep.onnx")
    quant_pre_process(str(fp32), str(prepped), skip_symbolic_shape=True)
    quantize_static(str(prepped), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=["Conv"], per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    prepped.unlink(missing_ok=True)


# --------------------------------------------------------------- backends
def make_backends(model: PatchCore, fp32: Path, int8: Path, threads: int, device: str):
    import onnxruntime as ort

    torch.set_num_threads(threads)
    backends = {}

    def torch_fn(dev):
        # own copy per device: .to() is in-place on modules, so a shared
        # extractor would be moved to the last device and break the others
        m = copy.deepcopy(model).to(dev).eval()

        @torch.no_grad()
        def run(x):
            out = m(torch.from_numpy(x).to(dev))
            if dev == "mps":
                torch.mps.synchronize()
            return out.cpu().numpy()
        return run

    backends["PyTorch eager - CPU (FP32)"] = torch_fn("cpu")
    if device != "cpu":
        backends[f"PyTorch eager - {device.upper()} (FP32)"] = torch_fn(device)

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    def ort_fn(path, providers):
        state = {}

        def run(x):  # session created lazily so a provider error is caught per backend
            if "s" not in state:
                state["s"] = ort.InferenceSession(str(path), so, providers=providers)
            return state["s"].run(None, {"image": x})[0]
        return run

    backends["ONNX Runtime - CPU (FP32)"] = ort_fn(fp32, ["CPUExecutionProvider"])
    backends["ONNX Runtime - CPU (INT8)"] = ort_fn(int8, ["CPUExecutionProvider"])
    avail = ort.get_available_providers()
    if "CoreMLExecutionProvider" in avail:
        backends["ONNX Runtime - CoreML (FP32)"] = ort_fn(
            fp32, ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    if "CUDAExecutionProvider" in avail:
        backends["ONNX Runtime - CUDA (FP32)"] = ort_fn(
            fp32, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    return backends


# ------------------------------------------------------ evaluate + timing
def to_anomaly_map(patch: np.ndarray) -> np.ndarray:
    """[1,h,w] patch scores -> smoothed [224,224] map (bilinear + gaussian, as in PatchCore)."""
    t = torch.from_numpy(patch[None])
    up = torch.nn.functional.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                                         align_corners=False)[0, 0].numpy()
    return gaussian_filter(up, sigma=4)


def evaluate(run, test, images, masks):
    scores, maps = [], []
    for x in images:
        amap = to_anomaly_map(run(x[None])[0])
        maps.append(amap)
        scores.append(float(amap.max()))
    labels = np.array([s.label for s in test])
    img_auroc = roc_auc_score(labels, scores)
    pix_auroc = roc_auc_score(np.stack(masks).ravel(), np.stack(maps).ravel()) \
        if np.stack(masks).any() else float("nan")
    return img_auroc, pix_auroc, scores, maps


def benchmark(run, x, warmup: int, runs: int):
    for _ in range(warmup):
        run(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run(x)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.median(t)),
            "p95_ms": float(np.percentile(t, 95)), "fps": float(1000 / t.mean())}


# ------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--visa-root", required=True)
    p.add_argument("--category", default="pcb1")
    p.add_argument("--out", default="results")
    p.add_argument("--backbone", default="wide_resnet50_2")
    p.add_argument("--max-train", type=int, default=1000)
    p.add_argument("--coreset-ratio", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--calib", type=int, default=32)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    args = p.parse_args(argv)

    torch.manual_seed(0)
    out = Path(args.out) / args.category
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    train, test = load_split(args.visa_root, args.category)
    print(f"[{args.category}] train={len(train)} test={len(test)} "
          f"({sum(s.label for s in test)} anomalous) device={device}")

    print("1/4 fitting memory bank")
    model = fit(train, args, device)

    print("2/4 exporting ONNX (FP32) + INT8")
    fp32, int8 = out / "patchcore_fp32.onnx", out / "patchcore_int8.onnx"
    export_onnx(model, fp32)
    quantize_int8(fp32, int8, train, args.calib)

    print("3/4 evaluating + benchmarking")
    images = [load_image(s.image) for s in test]
    masks = [load_mask(s.mask) for s in test]
    x1 = images[0][None]
    rows, best_maps = [], None
    for name, run in make_backends(model, fp32, int8, args.threads, device).items():
        try:
            img_auc, pix_auc, scores, maps = evaluate(run, test, images, masks)
            speed = benchmark(run, x1, args.warmup, args.runs)
        except Exception as e:  # one backend failing must not lose the others
            print(f"  {name:32s} SKIPPED ({type(e).__name__}: {str(e)[:120]})")
            continue
        size = (int8 if "INT8" in name else fp32).stat().st_size / 1e6 if "ONNX" in name else None
        rows.append({"backend": name, "image_auroc": img_auc, "pixel_auroc": pix_auc,
                     "model_mb": size, **speed})
        print(f"  {name:32s} {speed['mean_ms']:7.1f} ms  {speed['fps']:6.1f} FPS  "
              f"img-AUROC {img_auc:.3f}  px-AUROC {pix_auc:.3f}")
        if best_maps is None:
            best_maps = (scores, maps)

    meta = {"category": args.category, "backbone": args.backbone,
            "train_images_used": min(len(train), args.max_train),
            "coreset_ratio": args.coreset_ratio, "memory_bank": int(model.bank_t.shape[1]),
            "threads": args.threads, "batch_size": 1, "input": f"{IMG_SIZE}x{IMG_SIZE}",
            "machine": machine_name(),
            "torch": torch.__version__, "pretrained": not args.no_pretrained}
    import onnxruntime
    meta["onnxruntime"] = onnxruntime.__version__
    (out / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    print("4/4 writing report")
    from .report import write_report
    write_report(out, meta, rows, test, images, masks, *best_maps)
    print(f"done -> {out / 'results.md'}")


if __name__ == "__main__":
    main()
