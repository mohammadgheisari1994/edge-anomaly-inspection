#!/usr/bin/env bash
# One command: environment -> dataset -> fit -> ONNX/INT8 -> benchmark -> report.
#   ./run_all.sh            # default category pcb1
#   ./run_all.sh pcb1 pcb2  # several categories
# Full log is written to run.log.
set -euo pipefail
cd "$(dirname "$0")"
exec > >(tee -a run.log) 2>&1
echo "=== run started $(date) ==="

# newest Python 3.10+ available (macOS system python can be too old for current torch)
PY=""
for v in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$v" >/dev/null 2>&1 && "$v" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    PY="$v"; break
  fi
done
[ -z "$PY" ] && { echo "Need Python 3.10+ (brew install python@3.12)"; exit 1; }
echo "Using $($PY --version)"

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
# python.org builds on macOS ship without CA certificates; use certifi's bundle
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"

./scripts/download_visa.sh data/VisA

CATS=("$@")
[ ${#CATS[@]} -eq 0 ] && CATS=(pcb1)
for c in "${CATS[@]}"; do
  python -m edgeanom.pipeline --visa-root data/VisA --category "$c" --out results
done
python scripts/update_readme.py "results/${CATS[0]}"
echo
echo "Results: results/<category>/results.md, latency.png, examples.png"
echo "=== run finished $(date) ==="
