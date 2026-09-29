#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=scripts:${PYTHONPATH:-}
source /venv/main/bin/activate

echo "=== unit tests ==="
python scripts/test_units.py

echo "=== extract activations ==="
python scripts/extract_activations.py --dataset all --batch-size 8 --device cuda --resume

echo "=== layer probes ==="
python scripts/probe_layers.py --device cuda

echo "=== nullspace ==="
python scripts/nullspace.py --device cuda

echo "=== steering ==="
python scripts/steer.py --device cuda

echo "=== summarize ==="
python scripts/summarize.py

echo "=== done ==="
ls -la artifacts/figures artifacts/results
