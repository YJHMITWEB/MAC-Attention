#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./autorun_batch_node0.sh [MAX_CONCURRENT_REQUESTS]
  ./autorun_batch_node0.sh --max-concurrent-requests N
  ./autorun_batch_node0.sh --max-concurrent-requests N --port PORT
  ./autorun_batch_node0.sh --max-concurrent-requests N --model-name Llama-3.1-8B-Instruct
  ./autorun_batch_node0.sh --baseline
  ./autorun_batch_node0.sh --mode baseline

MAX_CONCURRENT_REQUESTS controls both:
  - SGLang --max-running-requests
  - the number of in-flight LongBench API requests

Default mode is "mac". Use --baseline or --mode baseline for the
no-MAC FlashInfer baseline. CUDA graph is controlled by
LONGBENCH_DISABLE_CUDA_GRAPH and is enabled by default in this reference copy.

This runs the full LongBench set by default. Set MAX_SAMPLES or pass
--max-samples for a smoke test.

You may also set these environment variables:
  MAX_CONCURRENT_REQUESTS, BENCH_PORT, MODEL_NAME, CUDA_VISIBLE_DEVICES,
  CHUNKED_PREFILL_SIZE, MAC_THRESHOLD, MAC_LOOKBACK_TOKENS_LEFT,
  MAC_GEN_MIN_LIMIT, MAC_SEMANTIC_POS_AHEAD, MAC_PROFILE,
  MAC_PROFILE_PATH, MAC_PERSISTENT_MIXED_EARLY_MISS_DIRECT,
  MAC_PERSISTENT_HIT_TAIL_GROUP, MAX_SAMPLES, RUN_MODE,
  PORTABLE_MAC_PLUGIN, SGLANG_ROOT, MAC_ATTENTION_ROOT, LONG_BENCH_ROOT,
  LONGBENCH_DISABLE_CUDA_GRAPH, LONGBENCH_CUDA_GRAPH_BS,
  LONGBENCH_DISABLE_PIECEWISE_CUDA_GRAPH.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC_TUNING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-1}"
BENCH_PORT="${BENCH_PORT:-18543}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
SERVER_TP="${SERVER_TP:-1}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_INDEX="${SAMPLE_INDEX:--1}"
KILL_EXISTING="${KILL_EXISTING:-1}"
RUN_MODE="${RUN_MODE:-mac}"
PORTABLE_MAC_PLUGIN="${PORTABLE_MAC_PLUGIN:-1}"
MAC_THRESHOLD="${MAC_THRESHOLD:-0.45}"
MAC_LOOKBACK_TOKENS_LEFT="${MAC_LOOKBACK_TOKENS_LEFT:-512}"
MAC_LOOKBACK_TOKENS_RIGHT="${MAC_LOOKBACK_TOKENS_RIGHT:-0}"
MAC_GEN_MIN_LIMIT="${MAC_GEN_MIN_LIMIT:-2048}"
MAC_SEMANTIC_POS_AHEAD="${MAC_SEMANTIC_POS_AHEAD:-256}"
MAC_PROFILE="${MAC_PROFILE:-0}"
MAC_PROFILE_PATH="${MAC_PROFILE_PATH:-}"
MAC_PERSISTENT_MAX_CONTEXT="${MAC_PERSISTENT_MAX_CONTEXT:-131072}"
MAC_PERSISTENT_FAST_WINDOW="${MAC_PERSISTENT_FAST_WINDOW:-8192}"
MAC_PERSISTENT_TILE_TOKENS="${MAC_PERSISTENT_TILE_TOKENS:-32}"
MAC_PERSISTENT_STAGE_TOKENS="${MAC_PERSISTENT_STAGE_TOKENS:-64}"
MAC_PERSISTENT_MATCH_TILE_SLOTS="${MAC_PERSISTENT_MATCH_TILE_SLOTS:-32}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-concurrent-requests|--max-running-requests|-m)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      MAX_CONCURRENT_REQUESTS="$2"
      shift 2
      ;;
    --port|-p)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      BENCH_PORT="$2"
      shift 2
      ;;
    --model-name)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      MODEL_NAME="$2"
      shift 2
      ;;
    --server-tp)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      SERVER_TP="$2"
      shift 2
      ;;
    --max-samples)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --sample-index)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      SAMPLE_INDEX="$2"
      shift 2
      ;;
    --mac-profile)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      MAC_PROFILE="$2"
      shift 2
      ;;
    --mac-profile-path)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      MAC_PROFILE_PATH="$2"
      shift 2
      ;;
    --baseline)
      RUN_MODE=baseline
      shift
      ;;
    --mac)
      RUN_MODE=mac
      shift
      ;;
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      RUN_MODE="$2"
      shift 2
      ;;
    --no-kill)
      KILL_EXISTING=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_CONCURRENT_REQUESTS="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

if ! [[ "$MAX_CONCURRENT_REQUESTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT_REQUESTS must be a positive integer, got: $MAX_CONCURRENT_REQUESTS" >&2
  exit 2
fi

if ! [[ "$BENCH_PORT" =~ ^[1-9][0-9]*$ ]]; then
  echo "BENCH_PORT must be a positive integer, got: $BENCH_PORT" >&2
  exit 2
fi

if ! [[ "$SERVER_TP" =~ ^[1-9][0-9]*$ ]]; then
  echo "SERVER_TP must be a positive integer, got: $SERVER_TP" >&2
  exit 2
fi

if ! [[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]]; then
  echo "MAX_SAMPLES must be a non-negative integer, got: $MAX_SAMPLES" >&2
  exit 2
fi

if ! [[ "$MAC_PROFILE" =~ ^[0-9]+$ ]]; then
  echo "MAC_PROFILE must be a non-negative integer, got: $MAC_PROFILE" >&2
  exit 2
fi

if [[ "$RUN_MODE" != "mac" && "$RUN_MODE" != "baseline" ]]; then
  echo "RUN_MODE must be either 'mac' or 'baseline', got: $RUN_MODE" >&2
  exit 2
fi

module purge
module load cuda/13.0
if ! module load "${GCC_MODULE:-gcc/13.2.0}" >/dev/null 2>&1; then
  if ! module load gcc/13.2 >/dev/null 2>&1; then
    module load gcc/12.2
  fi
fi
if [[ -x /opt/rh/gcc-toolset-13/root/usr/bin/gcc && -x /opt/rh/gcc-toolset-13/root/usr/bin/g++ ]]; then
  export PATH="/opt/rh/gcc-toolset-13/root/usr/bin:$PATH"
elif [[ -x /opt/rh/gcc-toolset-12/root/usr/bin/gcc && -x /opt/rh/gcc-toolset-12/root/usr/bin/g++ ]]; then
  export PATH="/opt/rh/gcc-toolset-12/root/usr/bin:$PATH"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-macAttention_latest}"

export MINORTEST_ROOT="${MINORTEST_ROOT:-$(cd "$MAC_TUNING_ROOT/.." && pwd)}"
if [[ -z "${SGLANG_ROOT:-}" ]]; then
  if [[ -d "$MINORTEST_ROOT/OFFICIAL_GITHUB/sglang" ]]; then
    export SGLANG_ROOT="$MINORTEST_ROOT/OFFICIAL_GITHUB/sglang"
  elif [[ -d "$MINORTEST_ROOT/opensource_1/sglang_official_clean_bbe9c7eeb" ]]; then
    export SGLANG_ROOT="$MINORTEST_ROOT/opensource_1/sglang_official_clean_bbe9c7eeb"
  else
    export SGLANG_ROOT="$MINORTEST_ROOT/opensource_1/sglang"
  fi
fi
if [[ -z "${MAC_ATTENTION_ROOT:-}" ]]; then
  export MAC_ATTENTION_ROOT="$MAC_TUNING_ROOT/attention"
fi
export LONG_BENCH_ROOT="${LONG_BENCH_ROOT:-$MINORTEST_ROOT/LongBench}"
export MAC_WORKSPACE_BASE="$MAC_ATTENTION_ROOT"
export TVM_FFI_GPU_BACKEND=cuda
export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDAHOSTCXX="$CXX"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$MAC_ATTENTION_ROOT/.torch_extensions}"
export PYTHONPATH="$MAC_ATTENTION_ROOT/src:$SGLANG_ROOT/python:${PYTHONPATH:-}"
export MAC_ATTENTION_SGLANG_PROFILE=0
export MAC_ATTENTION_SGLANG_PROFILE_SYNC=0
export MAC_USE_FUSED_KV_ROPE="${MAC_USE_FUSED_KV_ROPE:-1}"
export MAC_USE_FUSED_Q_PRESERVE_ROPE="${MAC_USE_FUSED_Q_PRESERVE_ROPE:-1}"
export MAC_PERSISTENT_MIXED_GROUP_FALLBACK="${MAC_PERSISTENT_MIXED_GROUP_FALLBACK:-1}"
export MAC_PERSISTENT_MIXED_EARLY_MISS_DIRECT="${MAC_PERSISTENT_MIXED_EARLY_MISS_DIRECT:-1}"
export MAC_PERSISTENT_HIT_TAIL_GROUP="${MAC_PERSISTENT_HIT_TAIL_GROUP:-1}"
export MAC_PERSISTENT_ALL_HIT_DIRECT="${MAC_PERSISTENT_ALL_HIT_DIRECT:-0}"
export LONGBENCH_DISABLE_CUDA_GRAPH="${LONGBENCH_DISABLE_CUDA_GRAPH:-0}"
export LONGBENCH_CUDA_GRAPH_BS="${LONGBENCH_CUDA_GRAPH_BS:-1}"
export LONGBENCH_DISABLE_PIECEWISE_CUDA_GRAPH="${LONGBENCH_DISABLE_PIECEWISE_CUDA_GRAPH:-1}"
export LONGBENCH_PIECEWISE_CUDA_GRAPH_TOKENS="${LONGBENCH_PIECEWISE_CUDA_GRAPH_TOKENS:-8192}"
export LONGBENCH_SGLANG_EXTRA_ARGS="${LONGBENCH_SGLANG_EXTRA_ARGS:---skip-server-warmup --decode-log-interval 16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "$RUN_MODE" == "baseline" ]]; then
  unset MAC_ATTENTION_ENABLE
  unset MAC_ATTENTION_PORTABLE_PLUGIN
  unset SGLANG_LAUNCH_MODULE
  unset MAC_ATTENTION_SGLANG_CONFIG
  unset MAC_PERSISTENT_COOP
  unset MAC_PERSISTENT_MAX_CONTEXT
  unset MAC_PERSISTENT_FAST_WINDOW
  unset MAC_PERSISTENT_TILE_TOKENS
  unset MAC_PERSISTENT_STAGE_TOKENS
  unset MAC_PERSISTENT_MATCH_TILE_SLOTS
  unset MAC_PERSISTENT_DEBUG
  unset MAC_PERSISTENT_PARTIAL_FP32
  unset MAC_PERSISTENT_CACHE_LAYOUT
  unset MAC_PERSISTENT_CANDIDATE_MODE
  unset MAC_PERSISTENT_FAST_MATH
  unset MAC_PERSISTENT_MIXED_GROUP_FALLBACK
  unset MAC_PERSISTENT_MIXED_EARLY_MISS_DIRECT
  unset MAC_PERSISTENT_HIT_TAIL_GROUP
  unset MAC_PERSISTENT_ALL_HIT_DIRECT
else
  export MAC_ATTENTION_ENABLE=1
  export MAC_THRESHOLD
  export MAC_LOOKBACK_TOKENS_LEFT
  export MAC_LOOKBACK_TOKENS_RIGHT
  export MAC_GEN_MIN_LIMIT
  export MAC_SEMANTIC_POS_AHEAD
  export MAC_PROFILE
  export MAC_PROFILE_PATH
  export MAC_DISABLE_CUDA_GRAPH="${MAC_DISABLE_CUDA_GRAPH:-0}"
  export MAC_FORCE_PAGED_PREFILL="${MAC_FORCE_PAGED_PREFILL:-1}"
  export MAC_PERSISTENT_COOP="${MAC_PERSISTENT_COOP:-1}"
  export MAC_PERSISTENT_MAX_CONTEXT
  export MAC_PERSISTENT_FAST_WINDOW
  export MAC_PERSISTENT_TILE_TOKENS
  export MAC_PERSISTENT_STAGE_TOKENS
  export MAC_PERSISTENT_MATCH_TILE_SLOTS
  export MAC_PERSISTENT_DEBUG="${MAC_PERSISTENT_DEBUG:-0}"
  export MAC_PERSISTENT_PARTIAL_FP32="${MAC_PERSISTENT_PARTIAL_FP32:-1}"
  export MAC_PERSISTENT_CACHE_LAYOUT="${MAC_PERSISTENT_CACHE_LAYOUT:-slot_major}"
  export MAC_PERSISTENT_CANDIDATE_MODE="${MAC_PERSISTENT_CANDIDATE_MODE:-last_M}"
  export MAC_PERSISTENT_FAST_MATH="${MAC_PERSISTENT_FAST_MATH:-1}"
  if [[ "$PORTABLE_MAC_PLUGIN" == "1" ]]; then
    export MAC_ATTENTION_PORTABLE_PLUGIN=1
    export MAC_ATTENTION_SGLANG_STRICT="${MAC_ATTENTION_SGLANG_STRICT:-1}"
    export SGLANG_LAUNCH_MODULE="${SGLANG_LAUNCH_MODULE:-mac_attention.integrations.sglang.launch_server}"
  fi
fi

cd "$LONG_BENCH_ROOT"

mkdir -p "$TORCH_EXTENSIONS_DIR"
mkdir -p "$MINORTEST_ROOT/benchmark/LongBench/results"

if [[ "$KILL_EXISTING" == "1" ]]; then
  echo "Cleaning stale SGLang processes and port $BENCH_PORT before launch."
  if command -v lsof >/dev/null 2>&1; then
    PORT_PIDS="$(lsof -tiTCP:"$BENCH_PORT" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$PORT_PIDS" ]]; then
      kill -9 $PORT_PIDS >/dev/null 2>&1 || true
    fi
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${BENCH_PORT}/tcp" >/dev/null 2>&1 || true
  fi
  pkill -u "$USER" -f "sglang\\.launch_server.*--port ${BENCH_PORT}" >/dev/null 2>&1 || true
  pkill -u "$USER" -f "python3 -m sglang\\.launch_server.*--port ${BENCH_PORT}" >/dev/null 2>&1 || true
  pkill -u "$USER" -f "mac_attention\\.integrations\\.sglang\\.launch_server.*--port ${BENCH_PORT}" >/dev/null 2>&1 || true
  pkill -u "$USER" -f "python3 -m mac_attention\\.integrations\\.sglang\\.launch_server.*--port ${BENCH_PORT}" >/dev/null 2>&1 || true
  pkill -u "$USER" -f "pred_my_lb2.py.*--port ${BENCH_PORT}" >/dev/null 2>&1 || true
  sleep 2
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
if [[ "$RUN_MODE" == "baseline" ]]; then
  OUT="$LONG_BENCH_ROOT/results/baseline_full_${RUN_TS}"
  MAC_PROFILE_PATH=""
else
  OUT="$LONG_BENCH_ROOT/results/mac_latest_full_${RUN_TS}"
  if [[ -z "$MAC_PROFILE_PATH" ]]; then
    MAC_PROFILE_PATH="$MINORTEST_ROOT/benchmark/LongBench/mac_latest_profile_${RUN_TS}"
  fi
fi
mkdir -p "$OUT"
if [[ -n "$MAC_PROFILE_PATH" ]]; then
  mkdir -p "$MAC_PROFILE_PATH"
fi
echo "Writing LongBench outputs to: $OUT"
echo "Run mode: $RUN_MODE"
echo "Max concurrent requests: $MAX_CONCURRENT_REQUESTS"
echo "SGLang API port: $BENCH_PORT"
echo "Model: $MODEL_NAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SGLang source: $SGLANG_ROOT"
echo "MAC-Attention source: $MAC_ATTENTION_ROOT"
if [[ "$RUN_MODE" == "mac" ]]; then
  echo "Portable MAC plugin: $PORTABLE_MAC_PLUGIN"
  echo "SGLang launch module: ${SGLANG_LAUNCH_MODULE:-sglang.launch_server}"
  echo "MAC profile: $MAC_PROFILE"
  echo "MAC profile path: $MAC_PROFILE_PATH"
fi

PRED_ARGS=(
  --port "$BENCH_PORT"
  --server-tp "$SERVER_TP"
  --max-running-requests "$MAX_CONCURRENT_REQUESTS"
  --request-concurrency "$MAX_CONCURRENT_REQUESTS"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --model-name "$MODEL_NAME"
  --max-samples "$MAX_SAMPLES"
  --output-timestamp "$RUN_TS"
  --results-dir "$OUT"
)

if [[ "$SAMPLE_INDEX" != "-1" ]]; then
  PRED_ARGS+=(--sample-index "$SAMPLE_INDEX")
fi

if [[ "$RUN_MODE" == "mac" && "$PORTABLE_MAC_PLUGIN" != "1" ]]; then
  PRED_ARGS+=(
    --enable-mac true
    --mac-threshold "$MAC_THRESHOLD"
    --mac-lookback-tokens-left "$MAC_LOOKBACK_TOKENS_LEFT"
    --mac-lookback-tokens-right "$MAC_LOOKBACK_TOKENS_RIGHT"
    --mac-gen-min-limit "$MAC_GEN_MIN_LIMIT"
    --mac-semantic-pos-ahead "$MAC_SEMANTIC_POS_AHEAD"
    --mac-profile "$MAC_PROFILE"
    --mac-profile-path "$MAC_PROFILE_PATH"
    --mac-disable-cuda-graph true
    --mac-force-paged-prefill true
    --mac-persistent-coop true
    --mac-persistent-max-context "$MAC_PERSISTENT_MAX_CONTEXT"
    --mac-persistent-fast-window "$MAC_PERSISTENT_FAST_WINDOW"
    --mac-persistent-tile-tokens "$MAC_PERSISTENT_TILE_TOKENS"
    --mac-persistent-stage-tokens "$MAC_PERSISTENT_STAGE_TOKENS"
    --mac-persistent-match-tile-slots "$MAC_PERSISTENT_MATCH_TILE_SLOTS"
    --mac-persistent-debug 0
    --mac-persistent-partial-fp32 true
    --mac-persistent-cache-layout slot_major
    --mac-persistent-candidate-mode last_M
    --mac-persistent-fast-math true
  )
fi

python pred_my_lb2.py "${PRED_ARGS[@]}" 2>&1 | tee "$OUT/run.log"
