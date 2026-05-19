# <img src="assets/icon.png" height="48" style="vertical-align: -13px;"> MAC-Attention

🎓**Accepted at MLSys 2026**🎓

📄 **Paper:** [arXiv:2604.00235](https://arxiv.org/abs/2604.00235)

**MAC-Attention** is a high-performance attention mechanism that reduces
decoding overhead by **reusing attention computation across semantically
similar tokens**.

This branch contains the portable SGLang plugin implementation:

- a self-contained cooperative MAC persistent decode CUDA kernel,
- a Python package that installs SGLang hooks without modifying SGLang source,
- standalone FlashInfer-versus-MAC benchmark drivers,
- lightweight usage examples for standalone benchmarking and SGLang launch.

Important note on runtime critical path:

- **Match + attention** are on the decode critical path.
- **Prefill cache update** and **rectification/cache update** are designed to
  run asynchronously where supported and are not the target bottleneck.

![](assets/workflow.png)

## Portable SGLang Plugin

The plugin is designed for an official SGLang checkout. Users should not patch
SGLang files. Instead, put this package and SGLang on `PYTHONPATH`, set MAC
options through environment variables, and launch through the wrapper module.

Install SGLang and MAC-Attention in editable mode:

```bash
export SGLANG_ROOT=<path-to-official-sglang>
export MAC_ATTENTION_REPO_ROOT=<path-to-this-repo>

python -m pip install -e "$SGLANG_ROOT/python"
python -m pip install -e "$MAC_ATTENTION_REPO_ROOT/attention"
```

Launch SGLang with MAC-Attention enabled:

```bash
export SGLANG_ROOT=<path-to-official-sglang>
export MAC_ATTENTION_REPO_ROOT=<path-to-this-repo>
export MODEL_PATH=<path-to-model>

export PYTHONPATH="$MAC_ATTENTION_REPO_ROOT/attention/src:$SGLANG_ROOT/python:${PYTHONPATH:-}"
export MAC_ATTENTION_ENABLE=1
export MAC_ATTENTION_PORTABLE_PLUGIN=1
export MAC_ATTENTION_SGLANG_STRICT=1
export MAC_DISABLE_CUDA_GRAPH=1

export MAC_PERSISTENT_PARALLEL_Z2_SCHEDULE=1
export MAC_PERSISTENT_MIXED_MISSPACK_Z2=1
export MAC_FUSE_HIT_TAIL_IN_MERGE=0
export MAC_USE_FUSED_KV_ROPE=1
export MAC_USE_FUSED_Q_PRESERVE_ROPE=1

python -m mac_attention.integrations.sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --attention-backend flashinfer \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-radix-cache \
  --page-size 1 \
  --chunked-prefill-size 8192 \
  --port 18543
```

The wrapper installs hooks before delegating to `sglang.launch_server`. All MAC
runtime parameters are controlled by environment variables.

CUDA graph is disabled by default in the portable plugin. Users can opt in by
setting `MAC_DISABLE_CUDA_GRAPH=0` and launching SGLang with CUDA graph enabled.
The no-CUDA-graph path is the default because it is the best-validated portable
baseline and reproduces the non-graph results in `results/`.

To run the official SGLang FlashInfer baseline from the same checkout, unset
MAC-specific plugin variables and launch SGLang normally:

```bash
export SGLANG_PLUGINS=__none__
unset MAC_ATTENTION_ENABLE
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --attention-backend flashinfer \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-radix-cache \
  --page-size 1 \
  --chunked-prefill-size 8192 \
  --port 18544
```

## Hard Rules

- Keep the MAC math identical to the paper: fixed recent-query ring,
  pre-RoPE squared-L2 matching, threshold/rectification semantics, and no
  sorting or reranking in the match window.
- Keep miss and partial-miss work inside the single cooperative MAC decode
  kernel and device helpers.
- Do not add host-side all-miss routing, separate production kernels,
  production FlashInfer calls, or compile-time escape hatches.
- Do not add an ad hoc all-miss or `hit=0` special case. `hit=0` is the first
  point on the same MAC-Attention hit-ratio curve.
- The only allowed ad hoc fast path is all-hit, where no load balancing is
  needed.
- The hit curve should transition smoothly. Higher hit ratios must not select
  structurally more expensive paths.

## Project Layout

```text
MAC-Attention/
├── attention/
│   ├── src/mac_attention/
│   │   └── integrations/sglang/
│   │       ├── csrc/                         # CUDA kernels
│   │       ├── bridge.py                     # extension loader
│   │       ├── config.py                     # env/config mapping
│   │       ├── hook_installer.py             # in-memory SGLang hooks
│   │       └── launch_server.py              # portable launch wrapper
│   ├── tests/                                # correctness and hook tests
│   └── tools/                                # direct kernel tools
├── benchmark/LongBench/
│   ├── bench_mac_vs_flashinfer_direct.py     # standalone kernel curve
│   └── test_longbench_decode_latency.py      # controlled SGLang latency
├── portable_plugin_repro/                    # reproduction shell wrappers
├── results/                                  # curated reproducibility results
├── assets/                                   # paper/public README assets
└── pyproject.toml
```

## Standalone Benchmark Example

Run a standalone FlashInfer-versus-MAC synthetic head curve. Write generated
outputs outside the repository tree:

```bash
export MAC_ATTENTION_REPO_ROOT=<path-to-this-repo>
export CUDA_VISIBLE_DEVICES=0
export MAC_RESULTS_DIR=<path-outside-this-repo>/mac_attention_results
mkdir -p "$MAC_RESULTS_DIR"

export MAC_PERSISTENT_PARALLEL_Z2_SCHEDULE=1
export MAC_PERSISTENT_MIXED_MISSPACK_Z2=1
export MAC_FUSE_HIT_TAIL_IN_MERGE=0
export MAC_USE_FUSED_KV_ROPE=1
export MAC_USE_FUSED_Q_PRESERVE_ROPE=1

python "$MAC_ATTENTION_REPO_ROOT/benchmark/LongBench/bench_mac_vs_flashinfer_direct.py" \
  --contexts "65536,73728,98304,126976" \
  --batch "1" \
  --hit-rates "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.875,0.9,0.95,0.96875,1.0" \
  --bench-mode synthetic_head \
  --warmup 12 \
  --iters 60 \
  --partial-fp32 \
  --flashinfer-baseline-timing plan_run_wall \
  --csv "$MAC_RESULTS_DIR/standalone_hit_curve.csv"
```

For exact-quota `batch=1,Hq=32` synthetic head curves, do not report
`hit=0.99`; it rounds to all-hit. Use `hit=0.96875` as the first real
one-miss-head point.

## Validation Examples

Run unit tests:

```bash
python -m pytest \
  "$MAC_ATTENTION_REPO_ROOT/attention/tests/test_sglang_plugin_config.py" \
  "$MAC_ATTENTION_REPO_ROOT/attention/tests/test_sglang_q_preserve.py"
```

Run CUDA correctness for the persistent decode kernel:

```bash
python "$MAC_ATTENTION_REPO_ROOT/attention/tools/check_mac_persistent_decode.py"
```

## Included Results

Curated reproducibility results are included under `results/`.

- `results/portable_mac_cudagraph_handoff_20260519/`
  - latest portable implementation handoff;
  - complete standalone no-CUDA-graph hit curve against FlashInfer;
  - CUDA-graph SGLang production hit-curve sweep;
  - CUDA-graph LongBench output/reference material.
- `results/portable_plugin_hook_gap_20260517/`
  - no-CUDA-graph portable plugin checkpoint;
  - controlled SGLang production comparison against FlashInfer and the previous
    best patched-SGLang implementation;
  - standalone full-curve comparison against the previous best.

No-CUDA-graph SGLang production comparison, fixed LongBench prompts,
concurrency 1, `max_new_tokens=64`:

| context | FlashInfer tok/s | portable MAC tok/s | MAC / FlashInfer | previous best MAC tok/s |
|---:|---:|---:|---:|---:|
| 64K | 68.40 | 64.50 | 0.943x | 64.54 |
| 96K | 59.51 | 63.12 | 1.061x | 63.51 |
| 127K | 52.91 | 62.49 | 1.181x | 63.01 |

Standalone no-CUDA-graph full hit curve:

- source: `results/portable_mac_cudagraph_handoff_20260519/standalone_no_cudagraph_official_vs_best_20260517/comparison_current_vs_previous_best.csv`
- contexts: `64K`, `72K`, `96K`, `127K`
- hit ratios: `0`, `0.1`, `0.2`, `0.3`, `0.4`, `0.5`, `0.6`, `0.7`,
  `0.8`, `0.875`, `0.9`, `0.95`, `0.96875`, `1`
- FlashInfer timing: plan plus run wall time

## Development Notes

- Keep new scratch benchmark outputs outside this repository or in ignored
  result folders. Only curated reproducibility bundles should be committed.
- Keep production hooks portable: no SGLang source edits should be required.
- Keep performance comparisons against FlashInfer plan-inclusive baselines when
  measuring end-to-end decode latency.
