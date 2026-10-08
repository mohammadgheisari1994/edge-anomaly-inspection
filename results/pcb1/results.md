# Results - VisA `pcb1`

Machine: Apple M4 (macOS 27.0.1) | threads: 5 | batch 1 | input 224x224 | backbone wide_resnet50_2 | memory bank 7087 patches

| Backend | Mean latency (ms) | p95 (ms) | FPS | Speed-up | Image AUROC | Pixel AUROC | Model size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch eager - CPU (FP32) | 34.5 | 35.3 | 29.0 | 1.00x | 0.936 | 0.995 | - |
| PyTorch eager - MPS (FP32) | 23.8 | 24.1 | 42.0 | 1.45x | 0.936 | 0.995 | - |
| ONNX Runtime - CPU (FP32) | 42.3 | 43.2 | 23.7 | 0.82x | 0.936 | 0.995 | 134.7 |
| ONNX Runtime - CPU (INT8) | 35.2 | 36.0 | 28.4 | 0.98x | 0.929 | 0.994 | 60.6 |
| ONNX Runtime - CoreML (FP32) | 28.5 | 36.0 | 35.1 | 1.21x | 0.936 | 0.995 | 134.7 |

Latency = model only (image in, patch scores out), mean of repeated single-image runs after warm-up. Speed-up is relative to the first row.
