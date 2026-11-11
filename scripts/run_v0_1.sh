#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/root/code/urban-visual-gene/streetspatial_v0
PYTHON_BIN=${STREETSPATIAL_PYTHON:-/root/code/urban-visual-gene/experiments/dinov3_generation_revision/.venv/bin/python}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -u -m src.v01 --output "$PROJECT_ROOT/outputs/v0_1"
"$PYTHON_BIN" -u -m src.v01_composition --output "$PROJECT_ROOT/outputs/v0_1" --limit 2
"$PYTHON_BIN" -m src.v01_figures --output "$PROJECT_ROOT/outputs/v0_1"
