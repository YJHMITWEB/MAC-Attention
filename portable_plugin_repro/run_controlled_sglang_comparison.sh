#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_mac_portable.sh"

OUT_DIR="${OUT_DIR:-$MAC_RESULTS_ROOT/repro_portable_plugin_controlled_latency_c1}"
mkdir -p "$OUT_DIR"

cd "$MAC_ATTENTION_REPO_ROOT"
python "$MAC_BENCH_ROOT/test_longbench_decode_latency.py" \
  --mode both \
  --contexts "${CONTEXTS:-65536,98304,126976}" \
  --sample-indices "${SAMPLE_INDICES:-125,469,413}" \
  --concurrency-levels "${CONCURRENCY_LEVELS:-1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --ignore-eos \
  --portable-mac-plugin true \
  --port "${BENCH_PORT:-21931}" \
  --out-dir "$OUT_DIR"
