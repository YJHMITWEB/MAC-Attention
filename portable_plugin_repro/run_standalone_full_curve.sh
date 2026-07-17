#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_mac_portable.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-$MAC_ATTENTION_REPO_ROOT/results/repro_cuda_graph_hit_curves}"
mkdir -p "$OUT_DIR"

cd "$MAC_ATTENTION_REPO_ROOT"
for LAYOUT in "gqa8 4" "gqa4 8"; do
  read -r NAME HKV <<<"$LAYOUT"
  "$PYTHON_BIN" benchmark/bench_mac_vs_flashinfer_direct.py \
    --contexts "32768,65536,98304,126976" \
    --batch "1,16,64" \
    --hit-rates "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.875,0.9,0.95,0.96875,1.0" \
    --bench-mode synthetic_head \
    --cuda-graph \
    --hq 32 \
    --hkv "$HKV" \
    --warmup 12 \
    --iters 60 \
    --flashinfer-baseline-timing plan_run_wall \
    --partial-fp32 \
    --csv "$OUT_DIR/$NAME.csv"
done
