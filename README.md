# <img src="assets/icon.png" height="48" style="vertical-align: -13px;"> MAC-Attention

🎓**Accepted at MLSys 2026**🎓

📄 **Paper:** [arXiv:2604.00235](https://arxiv.org/abs/2604.00235)

## Update 2026-05-19: Portable SGLang Plugin

MAC-Attention includes a portable SGLang plugin path. The goal is to use an
official SGLang checkout with minimal integration work: install this package,
configure MAC through environment variables, and launch through the wrapper
module. SGLang source files do not need to be patched.

The public result shown below focuses on the fused MAC-Attention CUDA decode
kernel with CUDA graph disabled. In practical long-context workloads, the hit
ratio is usually above 95-99%; the figure highlights that high-hit region
because it is where users typically experience MAC-Attention on real datasets.

```bash
export SGLANG_ROOT=<path-to-official-sglang>
export MAC_ATTENTION_REPO_ROOT=<path-to-this-repo>
export MODEL_PATH=<path-to-model>

python -m pip install -e "$SGLANG_ROOT/python"
python -m pip install -e "$MAC_ATTENTION_REPO_ROOT"

export PYTHONPATH="$MAC_ATTENTION_REPO_ROOT/attention/src:$SGLANG_ROOT/python:${PYTHONPATH:-}"
export MAC_ATTENTION_ENABLE=1
export MAC_ATTENTION_PORTABLE_PLUGIN=1
export MAC_ATTENTION_SGLANG_STRICT=1
export MAC_DISABLE_CUDA_GRAPH=1

"$MAC_ATTENTION_REPO_ROOT/portable_plugin_repro/run_sglang_mac_server.sh"
```

For standalone MAC-vs-FlashInfer curves:

```bash
OUT_DIR=<path-to-output-dir> \
  "$MAC_ATTENTION_REPO_ROOT/portable_plugin_repro/run_standalone_full_curve.sh"
```

The source CSVs are committed under `results/`, and the figures below can be
regenerated with:

```bash
python -m pip install matplotlib seaborn pandas
python plot_portable_plugin_results.py
```

![Fused MAC-Attention CUDA kernel hit curve](assets/portable_update_20260519/no_cuda_graph_hit_curve.png)

**MAC-Attention** reduces long-context decode attention work by reusing cached
attention states for semantically similar tokens. The current implementation
packages:

- a fused persistent BF16 decode CUDA kernel with in-kernel matching, load
  scheduling, partial attention, merge, and cache update;
- a portable SGLang integration that installs hooks at runtime instead of
  editing SGLang source files;
- standalone correctness and hit-curve benchmark scripts for comparing the
  fused MAC-Attention kernel against FlashInfer.

## 🚀 Quick Start

Prerequisites:
- An NVIDIA GPU with BF16 support. The current validation target is H100-class
  hardware.
- A CUDA-enabled PyTorch environment. The current SGLang validation environment
  uses CUDA 13.0.
- `flashinfer-python` available in the environment for FlashInfer baselines and
  SGLang's FlashInfer attention backend.
- An official SGLang checkout.

Clone both repositories and install them in editable mode:

```bash
git clone https://github.com/sgl-project/sglang.git
git clone https://github.com/YJHMITWEB/MAC-Attention.git

export SGLANG_ROOT="$PWD/sglang"
export MAC_ATTENTION_REPO_ROOT="$PWD/MAC-Attention"

python -m pip install -e "$SGLANG_ROOT/python"
python -m pip install -e "$MAC_ATTENTION_REPO_ROOT"
python -m pip install flashinfer-python
```

Load the portable MAC-Attention defaults:

```bash
source "$MAC_ATTENTION_REPO_ROOT/portable_plugin_repro/env_mac_portable.sh"
python -c "import torch, mac_attention; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Launch SGLang with MAC-Attention enabled:

```bash
export MODEL_PATH=<path-to-model>
export CUDA_VISIBLE_DEVICES=0
export MAC_DISABLE_CUDA_GRAPH=1

"$MAC_ATTENTION_REPO_ROOT/portable_plugin_repro/run_sglang_mac_server.sh" \
  --host 0.0.0.0
```

The wrapper launches `mac_attention.integrations.sglang.launch_server`, which
loads official SGLang and installs MAC-Attention hooks in process. For a plain
FlashInfer baseline, launch official SGLang directly with the same model and
serving arguments.

## 🗂️ Project Layout

```text
MAC-Attention/
├── README.md
├── pyproject.toml
├── attention/
│   ├── src/mac_attention/
│   │   └── integrations/sglang/
│   │       ├── bridge.py                # JIT loader for CUDA extensions
│   │       ├── config.py                # Env and CLI config for SGLang hooks
│   │       ├── hook_installer.py        # Runtime hook entry point
│   │       ├── launch_server.py         # SGLang launch wrapper
│   │       ├── plugin.py                # SGLang plugin entry point
│   │       ├── flashinfer_hooks.py      # FlashInfer decode hook integration
│   │       ├── llama_hooks.py           # Model-side query/cache hooks
│   │       ├── cuda_graph_hooks.py      # CUDA graph compatibility hooks
│   │       ├── schedule_hooks.py        # Decode scheduling hooks
│   │       ├── profiling.py             # Lightweight MAC profiling helpers
│   │       └── csrc/
│   │           ├── mac_decode_persistent.cu
│   │           ├── mac_decode_rope_preserve.cu
│   │           ├── mac_merge_downdate_cache.cu
│   │           └── mac_prefill_update_cache.cu
│   ├── tests/
│   │   ├── test_mac_persistent_decode.py
│   │   ├── test_sglang_plugin_config.py
│   │   └── test_sglang_q_preserve.py
│   └── tools/
│       ├── bench_mac_persistent_decode.py
│       ├── check_mac_persistent_decode.py
│       └── profile_mac_persistent_decode.py
├── benchmark/
│   └── bench_mac_vs_flashinfer_direct.py
├── portable_plugin_repro/
│   ├── env_mac_portable.sh
│   ├── run_correctness.sh
│   ├── run_hook_check.sh
│   ├── run_sglang_mac_server.sh
│   └── run_standalone_full_curve.sh
├── results/
│   └── no_cuda_graph_hit_curve/comparison.csv
├── assets/portable_update_20260519/
│   ├── no_cuda_graph_hit_curve.png
│   └── no_cuda_graph_hit_curve.svg
└── plot_portable_plugin_results.py
```

## 🛠️ Build / JIT Compilation

The CUDA kernels are built on demand through `torch.utils.cpp_extension`. The
first correctness run, benchmark run, or SGLang decode that reaches the MAC path
will compile the extension from `attention/src/mac_attention/integrations/sglang/csrc/`.

The portable environment script keeps build products inside the repository by
default:

```bash
source portable_plugin_repro/env_mac_portable.sh
echo "$TORCH_EXTENSIONS_DIR"
```

Useful build knobs:

```bash
export MAC_WORKSPACE_BASE="$MAC_ATTENTION_REPO_ROOT/attention"
export TORCH_EXTENSIONS_DIR="$MAC_ATTENTION_REPO_ROOT/attention/.torch_extensions"
export TVM_FFI_GPU_BACKEND=cuda
```

Force a clean rebuild:

```bash
rm -rf "$MAC_ATTENTION_REPO_ROOT/attention/.torch_extensions"
```

## 🧪 Sanity Checks

Verify the SGLang hook installation without launching a server:

```bash
"$MAC_ATTENTION_REPO_ROOT/portable_plugin_repro/run_hook_check.sh"
```

Run the CUDA kernel and plugin tests:

```bash
"$MAC_ATTENTION_REPO_ROOT/portable_plugin_repro/run_correctness.sh"
```

Run the fused decode kernel checker directly:

```bash
cd "$MAC_ATTENTION_REPO_ROOT"
PYTHONPATH=attention/src python attention/tools/check_mac_persistent_decode.py
```

## 📊 Benchmarks

The public benchmark set is the no-CUDA-graph fused-kernel hit
curve against FlashInfer. It sweeps one request at context lengths 64K, 72K,
96K, and 127K, and hit ratios from 0 to 1.0:

```bash
cd "$MAC_ATTENTION_REPO_ROOT"
export MAC_DISABLE_CUDA_GRAPH=1
OUT_DIR="$MAC_ATTENTION_REPO_ROOT/results/repro_no_cuda_graph_hit_curve" \
  "$MAC_ATTENTION_REPO_ROOT/portable_plugin_repro/run_standalone_full_curve.sh"
```

The wrapper runs:

```bash
python benchmark/bench_mac_vs_flashinfer_direct.py \
  --contexts "65536,73728,98304,126976" \
  --batch "1" \
  --hit-rates "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.875,0.9,0.95,0.96875,1.0" \
  --bench-mode synthetic_head \
  --warmup 12 \
  --iters 60 \
  --flashinfer-baseline-timing plan_run_wall \
  --partial-fp32 \
  --csv "$OUT_DIR/curve.csv"
```

Result files:
- `results/no_cuda_graph_hit_curve/comparison.csv`: committed source data for
  the README figure.
- `assets/portable_update_20260519/no_cuda_graph_hit_curve.png`: rendered
  figure.
- `assets/portable_update_20260519/no_cuda_graph_hit_curve.svg`: vector figure.

Regenerate the figure:

```bash
python -m pip install matplotlib seaborn pandas
python plot_portable_plugin_results.py
```

Run the MAC-only persistent decode microbenchmark:

```bash
cd "$MAC_ATTENTION_REPO_ROOT"
PYTHONPATH=attention/src python attention/tools/bench_mac_persistent_decode.py \
  --kv-len 65536 73728 98304 126976 \
  --batch 1 \
  --bench-mode synthetic_head \
  --bench-hit-rate 0.95 \
  --warmup 12 \
  --iters 60 \
  --partial-fp32 \
  --csv results/mac_only_persistent_decode.csv
```

## Runtime Knobs

All production SGLang integration options are passed through environment
variables. The defaults below are set by `portable_plugin_repro/env_mac_portable.sh`.

```bash
export MAC_ATTENTION_ENABLE=1
export MAC_ATTENTION_PORTABLE_PLUGIN=1
export MAC_ATTENTION_SGLANG_STRICT=1
export MAC_DISABLE_CUDA_GRAPH=1

export MAC_THRESHOLD=0.45
export MAC_LOOKBACK_TOKENS_LEFT=512
export MAC_LOOKBACK_TOKENS_RIGHT=0
export MAC_GEN_MIN_LIMIT=2048
export MAC_SEMANTIC_POS_AHEAD=256

export MAC_PERSISTENT_PARALLEL_Z2_SCHEDULE=1
export MAC_PERSISTENT_MIXED_MISSPACK_Z2=1
export MAC_FUSE_HIT_TAIL_IN_MERGE=0
export MAC_PERSISTENT_PARTIAL_FP32=1
export MAC_USE_FUSED_KV_ROPE=1
export MAC_USE_FUSED_Q_PRESERVE_ROPE=1
```

Key meanings:
- `MAC_ATTENTION_ENABLE`: enables MAC-Attention in the SGLang hook path.
- `MAC_ATTENTION_PORTABLE_PLUGIN`: selects the portable runtime integration.
- `MAC_ATTENTION_SGLANG_STRICT`: fails fast if expected SGLang hook points are
  missing.
- `MAC_DISABLE_CUDA_GRAPH`: disables SGLang CUDA graph capture for the public
  result path. Set to `0` only when validating CUDA graph behavior separately.
- `MAC_THRESHOLD`: semantic query-match threshold.
- `MAC_LOOKBACK_TOKENS_LEFT`: ring-cache rows used for matching.
- `MAC_GEN_MIN_LIMIT`: minimum generated/context length before MAC decode is
  attempted.
- `MAC_SEMANTIC_POS_AHEAD`: rectification band used by fused decode cache
  updates.
- `MAC_PERSISTENT_PARALLEL_Z2_SCHEDULE` and
  `MAC_PERSISTENT_MIXED_MISSPACK_Z2`: current load-scheduling path used by the
  committed benchmark.
- `MAC_USE_FUSED_KV_ROPE` and `MAC_USE_FUSED_Q_PRESERVE_ROPE`: fused RoPE
  helpers used by the SGLang integration.

## SGLang Integration Flow

`portable_plugin_repro/run_sglang_mac_server.sh` launches:

```bash
python -m mac_attention.integrations.sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --attention-backend flashinfer \
  --trust-remote-code \
  --disable-radix-cache \
  --page-size 1 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" \
  --port "${PORT:-18543}"
```

At startup, `launch_server.py` imports official SGLang, installs the
MAC-Attention hooks, and then delegates to SGLang's normal server launch. During
decode, the hooks preserve model query state, maintain MAC ring caches, call the
fused persistent decode kernel for supported BF16 paged-KV requests, and use the
standard SGLang/FlashInfer path when MAC is disabled or a request is outside the
supported MAC configuration.
