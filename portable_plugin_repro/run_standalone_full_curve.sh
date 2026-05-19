#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_mac_portable.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-$MINORTEST_DIR/benchmark/LongBench/repro_portable_plugin_full_curve}"
mkdir -p "$OUT_DIR"

cd "$MINORTEST_DIR"
"$PYTHON_BIN" benchmark/LongBench/bench_mac_vs_flashinfer_direct.py \
  --contexts "65536,98304,126976" \
  --batch "1" \
  --hit-rates "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.875,0.9,0.95,0.96875,1.0" \
  --bench-mode synthetic_head \
  --warmup 12 \
  --iters 60 \
  --flashinfer-baseline-timing plan_run_wall \
  --partial-fp32 \
  --csv "$OUT_DIR/curve.csv"
