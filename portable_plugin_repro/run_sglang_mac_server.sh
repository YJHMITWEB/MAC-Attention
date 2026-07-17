#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_mac_portable.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-18543}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH must point to the model served by SGLang." >&2
  exit 1
fi

CUDA_GRAPH_ARGS=()
if [[ "${MAC_DISABLE_CUDA_GRAPH:-0}" != "0" ]]; then
  CUDA_GRAPH_ARGS+=(--disable-cuda-graph)
fi

exec "$PYTHON_BIN" -m mac_attention.integrations.sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --attention-backend flashinfer \
  --trust-remote-code \
  "${CUDA_GRAPH_ARGS[@]}" \
  --disable-radix-cache \
  --page-size 1 \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --port "$PORT" \
  "$@"
