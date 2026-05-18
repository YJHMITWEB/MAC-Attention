# Runbook

This runbook assumes the repository is checked out as a standalone public-style
MAC-Attention tree. Paths below are placeholders and should be set by the user
or CI environment.

## Environment

```bash
export MAC_ATTENTION_REPO_ROOT=<path-to-mac-attention>
export MAC_ATTENTION_ROOT="$MAC_ATTENTION_REPO_ROOT/attention"
export SGLANG_ROOT=<path-to-official-sglang>
export LONG_BENCH_ROOT=<path-to-LongBench>
export MODEL_PATH=<path-to-Llama-3.1-8B-Instruct-or-compatible-model>
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="$MAC_ATTENTION_ROOT/src:$SGLANG_ROOT/python:${PYTHONPATH:-}"
export MAC_WORKSPACE_BASE="$MAC_ATTENTION_ROOT"
export TORCH_EXTENSIONS_DIR="$MAC_ATTENTION_ROOT/.torch_extensions"
export TVM_FFI_GPU_BACKEND=cuda
```

On CUDA 13 systems, use a compiler/CUDA environment equivalent to:

```bash
module purge >/dev/null 2>&1 || true
module load cuda/13.0

export GCC_ROOT=${GCC_ROOT:-<path-to-gcc-13-toolchain>}
export CUDA_ROOT=${CUDA_ROOT:-<path-to-cuda-13>}
export CC="$GCC_ROOT/bin/gcc"
export CXX="$GCC_ROOT/bin/g++"
export CUDAHOSTCXX="$GCC_ROOT/bin/g++"
export NVCC_PREPEND_FLAGS="-ccbin=$GCC_ROOT/bin/g++"
export PATH="$GCC_ROOT/bin:$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$GCC_ROOT/lib64:$GCC_ROOT/lib/gcc/x86_64-redhat-linux/13:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"
```

If FlashInfer headers are not auto-discovered:

```bash
export MAC_FLASHINFER_INCLUDE_DIR="$(python - <<'PY'
from pathlib import Path
import flashinfer
print(Path(flashinfer.__file__).resolve().parent / "data" / "include")
PY
)"
```

## Default MAC Settings

The reproduction wrappers set these defaults:

```bash
export MAC_ATTENTION_ENABLE=1
export MAC_ATTENTION_PORTABLE_PLUGIN=1
export MAC_ATTENTION_SGLANG_STRICT=1
export MAC_THRESHOLD=0.45
export MAC_LOOKBACK_TOKENS_LEFT=512
export MAC_LOOKBACK_TOKENS_RIGHT=0
export MAC_GEN_MIN_LIMIT=2048
export MAC_SEMANTIC_POS_AHEAD=256
export MAC_DISABLE_CUDA_GRAPH=1
export MAC_FORCE_PAGED_PREFILL=1
export MAC_PERSISTENT_COOP=1
export MAC_PERSISTENT_MAX_CONTEXT=131072
export MAC_PERSISTENT_FAST_WINDOW=8192
export MAC_PERSISTENT_TILE_TOKENS=32
export MAC_PERSISTENT_STAGE_TOKENS=64
export MAC_PERSISTENT_MATCH_TILE_SLOTS=32
export MAC_PERSISTENT_PARTIAL_FP32=1
export MAC_PERSISTENT_CACHE_LAYOUT=slot_major
export MAC_PERSISTENT_CANDIDATE_MODE=last_M
export MAC_PERSISTENT_FAST_MATH=1
export MAC_PERSISTENT_PARALLEL_Z2_SCHEDULE=1
export MAC_PERSISTENT_MIXED_MISSPACK_Z2=1
export MAC_FUSE_HIT_TAIL_IN_MERGE=0
export MAC_PERSISTENT_MIXED_GROUP_FALLBACK=1
export MAC_PERSISTENT_MIXED_EARLY_MISS_DIRECT=1
export MAC_PERSISTENT_HIT_TAIL_GROUP=1
export MAC_PERSISTENT_ALL_HIT_DIRECT=0
export MAC_USE_FUSED_KV_ROPE=1
export MAC_USE_FUSED_Q_PRESERVE_ROPE=1
```

## Correctness

```bash
cd "$MAC_ATTENTION_REPO_ROOT"
./portable_plugin_repro/run_correctness.sh
```

Expected outcome: all persistent decode, Q-preserve, and plugin config tests
pass.

## Hook Smoke Test

```bash
cd "$MAC_ATTENTION_REPO_ROOT"
./portable_plugin_repro/run_hook_smoke.sh
```

Expected outcome: `enable_mac True`, `server_args_hooked True`, and a positive
number of installed SGLang hooks.

## Standalone Full Hit Curve

```bash
cd "$MAC_ATTENTION_REPO_ROOT"
OUT_DIR=results/standalone_full_curve \
./portable_plugin_repro/run_standalone_full_curve.sh
```

The default hit grid is:

```text
0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.875,0.9,0.95,0.96875,1.0
```

For `batch=1,Hq=32,bench-mode=synthetic_head`, do not report `hit=0.99`: it
rounds to zero miss heads and is identical to all-hit. `hit=0.96875` is the
first real one-miss-head point.

## Controlled SGLang Production Latency

```bash
cd "$MAC_ATTENTION_REPO_ROOT"
OUT_DIR=results/controlled_sglang \
CONCURRENCY_LEVELS=1,2,4 \
MAX_NEW_TOKENS=64 \
./portable_plugin_repro/run_controlled_sglang_comparison.sh
```

This launches baseline FlashInfer and portable MAC servers on selected
LongBench samples and reports decode-log steady throughput.

## Full LongBench

MAC:

```bash
cd "$MAC_ATTENTION_REPO_ROOT"
CUDA_VISIBLE_DEVICES=0 \
MAC_PERSISTENT_PARALLEL_Z2_SCHEDULE=1 \
MAC_PERSISTENT_MIXED_MISSPACK_Z2=1 \
MAC_FUSE_HIT_TAIL_IN_MERGE=0 \
MAX_CONCURRENT_REQUESTS=1 \
BENCH_PORT=18543 \
MODEL_NAME=Llama-3.1-8B-Instruct \
./portable_plugin_repro/run_full_longbench_mac.sh
```

FlashInfer baseline:

```bash
cd "$MAC_ATTENTION_REPO_ROOT"
CUDA_VISIBLE_DEVICES=0 \
MAX_CONCURRENT_REQUESTS=1 \
BENCH_PORT=18543 \
MODEL_NAME=Llama-3.1-8B-Instruct \
./portable_plugin_repro/run_full_longbench_baseline.sh
```

The full LongBench scripts require an external `LONG_BENCH_ROOT` containing the
benchmark runner and data.
