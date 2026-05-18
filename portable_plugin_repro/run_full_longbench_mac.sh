#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_mac_portable.sh"

cd "$LONG_BENCH_ROOT"
if [[ ! -x ./autorun_batch_node0.sh ]]; then
  echo "LONG_BENCH_ROOT must contain executable autorun_batch_node0.sh: $LONG_BENCH_ROOT" >&2
  exit 1
fi
./autorun_batch_node0.sh \
  --mode mac \
  --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS:-1}" \
  --port "${BENCH_PORT:-18543}" \
  --model-name "${MODEL_NAME:-Llama-3.1-8B-Instruct}"
