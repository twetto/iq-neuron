#!/usr/bin/env bash
# Regenerate the IMU de-rotation PC-circuit visualization for ONE EuRoC frame.
#
#   1. dump that frame's de-rotated 3-DoF problem (omega + obs) via the Rust
#      `euroc_egomotion_imu` example (cargo run, builds if needed)
#   2. run tests/plot_pc_derot_viz.py to render
#        tests/pc_derot_raster.png    (membrane heatmap + spike raster + 6-DoF m(t))
#        tests/pc_derot_weights.png   (the closed-loop synaptic matrix W[post,pre])
#
# Usage:
#   ./run_pc_derot_viz.sh [dataset_path] [frame]
# Examples:
#   ./run_pc_derot_viz.sh
#   ./run_pc_derot_viz.sh /home/twetto/Downloads/V1_01_easy 150
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (script lives in utils/)

DATASET="${1:-/home/twetto/Downloads/V1_01_easy}"
FRAME="${2:-150}"
NUM_FRAMES=$((FRAME + 10))                       # process just past the target frame
CSV="$ROOT/rust/derot_frame_${FRAME}.csv"
PYTHON="$ROOT/venv/bin/python"

[ -d "$DATASET/mav0" ] || { echo "error: no EuRoC dataset at $DATASET (expected mav0/)" >&2; exit 1; }
[ -x "$PYTHON" ] || PYTHON="python3"             # fall back to system python

echo "==> dumping frame $FRAME of $DATASET -> $CSV"
( cd "$ROOT/rust" && \
  DUMP_DEROT_PATH="$CSV" DUMP_DEROT_FRAME="$FRAME" \
  cargo run -p iqif-vio --example euroc_egomotion_imu --release -- \
    "$DATASET" "$NUM_FRAMES" /tmp/egomotion_imu_dump.csv )

echo "==> rendering visualization"
"$PYTHON" "$ROOT/tests/plot_pc_derot_viz.py" "$CSV"

echo "==> done:"
echo "    $ROOT/tests/pc_derot_raster.png"
echo "    $ROOT/tests/pc_derot_weights.png"
