#!/usr/bin/env bash
# TensorRT path for NVIDIA Jetson (JetPack ships trtexec).
# NOT YET RUN ON JETSON HARDWARE - numbers in the main README come from the Mac run.
#
#   ./jetson/build_trt.sh results/pcb1/patchcore_fp32.onnx
#
# Builds an FP16 engine and prints trtexec's latency/throughput summary.
set -euo pipefail
ONNX="${1:?path to patchcore_fp32.onnx}"
ENGINE="${ONNX%.onnx}_fp16.engine"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

"$TRTEXEC" --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --shapes=image:1x3x224x224 \
  --warmUp=500 --iterations=200 --avgRuns=50

echo "Engine: $ENGINE"
