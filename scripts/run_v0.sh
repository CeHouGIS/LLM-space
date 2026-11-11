#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/code/urban-visual-gene/streetspatial_v0
DATA_ROOT=/host/mnt/nas/urban-visual-gene/streetspatial_v0
PYTHON_BIN=${STREETSPATIAL_PYTHON:-/root/code/urban-visual-gene/experiments/dinov3_generation_revision/.venv/bin/python}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m src.check_views \
  --manifest "$DATA_ROOT/data/splits/pilot200_final/transformations.jsonl" \
  --output "$DATA_ROOT/data/splits/pilot200_final/review_sheets" --no-sheets
"$PYTHON_BIN" -u -m src.extract_features \
  --manifest "$DATA_ROOT/data/splits/pilot200_final/transformations.jsonl" \
  --output "$DATA_ROOT/outputs/v0/pilot200_features"
"$PYTHON_BIN" -m src.audit_run
"$PYTHON_BIN" -u -m src.analyze \
  --features "$DATA_ROOT/outputs/v0/pilot200_features" \
  --output "$DATA_ROOT/outputs/v0/pilot200_analysis"
