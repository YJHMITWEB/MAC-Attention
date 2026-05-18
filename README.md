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
- controlled SGLang and LongBench reproduction scripts.

Important note on runtime critical path:

- **Match + attention** are on the decode critical path.
- **Prefill cache update** and **rectification/cache update** are designed to
  run asynchronously where supported and are not the target bottleneck.

![](assets/workflow.png)

## Portable SGLang Plugin

The plugin is designed for an official SGLang checkout. Users should not patch
SGLang files. Instead, put this package and SGLang on `PYTHONPATH`, set MAC
options through environment variables, and launch through the wrapper module:

```bash
export MAC_ATTENTION_REPO_ROOT=<path-to-this-repo>
export MAC_ATTENTION_ROOT="$MAC_ATTENTION_REPO_ROOT/attention"
export SGLANG_ROOT=<path-to-official-sglang>
export MODEL_PATH=<path-to-model>

export PYTHONPATH="$MAC_ATTENTION_ROOT/src:$SGLANG_ROOT/python:${PYTHONPATH:-}"
export MAC_ATTENTION_ENABLE=1
export MAC_ATTENTION_PORTABLE_PLUGIN=1
export MAC_ATTENTION_SGLANG_STRICT=1

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
├── docs/                                     # design, runbook, baselines
├── assets/                                   # paper/public README assets
└── pyproject.toml
```

## Quick Validation

Set the external paths first:

```bash
export MAC_ATTENTION_REPO_ROOT=<path-to-this-repo>
export SGLANG_ROOT=<path-to-official-sglang>
export LONG_BENCH_ROOT=<path-to-LongBench>
export MODEL_PATH=<path-to-model>
export CUDA_VISIBLE_DEVICES=0
```

Then run:

```bash
./portable_plugin_repro/run_correctness.sh
./portable_plugin_repro/run_hook_smoke.sh
./portable_plugin_repro/run_standalone_full_curve.sh
./portable_plugin_repro/run_controlled_sglang_comparison.sh
```

See `docs/RUNBOOK.md` for CUDA 13 compiler setup, default MAC environment
variables, full LongBench commands, and reproduction details.

## Current Baseline

The current portable plugin reproduces the best tuned SGLang integration within
normal benchmark noise. Representative controlled SGLang decode throughput:

| concurrency | context | FlashInfer tok/s | portable MAC tok/s | MAC / FlashInfer |
|---:|---:|---:|---:|---:|
| 1 | 64K | 68.77 | 63.43 | 0.922x |
| 1 | 96K | 59.56 | 62.48 | 1.049x |
| 1 | 127K | 53.04 | 61.90 | 1.167x |
| 2 | 64K | 104.15 | 111.84 | 1.074x |
| 2 | 96K | 84.08 | 117.63 | 1.399x |
| 2 | 127K | 72.02 | 113.04 | 1.570x |
| 4 | 64K | 141.17 | 191.57 | 1.357x |
| 4 | 96K | 67.62 | 99.55 | 1.472x |
| 4 | 127K | 61.84 | 97.64 | 1.579x |

Standalone and production details are documented in
`docs/PERFORMANCE_BASELINE_20260517.md`.

## Development Notes

- Generated benchmark outputs belong under `results/` and are ignored.
- For exact-quota `batch=1,Hq=32` synthetic head curves, do not report
  `hit=0.99`; it rounds to all-hit. Use `hit=0.96875` as the first real
  one-miss-head point.
- Future optimization work should update `docs/OPTIMIZATION_LOG.md`.

