# Portable Plugin Hook Gap Checkpoint

This checkpoint captures the portable-plugin implementation that closes the
extra Python hook overhead relative to the previous patched-SGLang best-so-far
checkpoint.

## What Changed

- MAC-Attention can be enabled as a portable SGLang plugin through environment
  variables instead of SGLang CLI/source hacks.
- The Llama model and attention hooks cache MAC config on the forward batch and
  use direct hook factories for the hot decode path.
- FlashInfer backend wrappers keep the portable plugin entry points, but MAC
  decode remains self-contained in the MAC implementation and does not route to
  FlashInfer decode.
- Reproduction scripts under `portable_plugin_repro/` now work from both the
  local `opensource_1` workspace layout and this GitHub overlay layout.

## Validation Summary

Correctness:

```bash
portable_plugin_repro/run_correctness.sh
# 14 passed in 105.88s
```

Standalone kernel curve:

- Artifact: `standalone_full_curve_synthetic_head_cached_config/curve.csv`
- FlashInfer timing mode: plan plus run wall time.
- Contexts: 64K, 96K, 127K.
- Hit rates: `0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.875,0.9,0.95,0.96875,1.0`.
- Mean portable delta versus previous best: `-1.33%` across 42 matched rows.

Controlled SGLang decode, concurrency 1:

| context | FlashInfer tok/s | portable MAC tok/s | previous best MAC tok/s | portable vs previous |
|---:|---:|---:|---:|---:|
| 64K | 68.40 | 64.50 | 64.54 | -0.06% |
| 96K | 59.51 | 63.12 | 63.51 | -0.61% |
| 127K | 52.91 | 62.49 | 63.01 | -0.81% |

Partial LongBench stop-run:

- The full portable LongBench run was intentionally stopped at 311 of 503 rows.
- Accuracy on the first 311 rows matched the previous best: `29.260%`.
- Mean request latency on those rows was `16.612s`, versus `16.786s` for the
  matching previous-best prefix.

Detailed comparison:

- `comparison/portable_vs_best_so_far_direct_llama_attention.md`
- `comparison/portable_vs_best_so_far_direct_llama_attention.json`

## Reproduction

Run from this repository, or set the roots explicitly if the official SGLang
clone and LongBench data live elsewhere.

```bash
export SGLANG_ROOT=/path/to/official/sglang
export LONG_BENCH_ROOT=/path/to/LongBench
export MODEL_PATH=/path/to/Llama-3.1-8B-Instruct
export PYTHON_BIN=/path/to/python

portable_plugin_repro/run_correctness.sh

OUT_DIR=results/repro_standalone_full_curve \
  portable_plugin_repro/run_standalone_full_curve.sh

OUT_DIR=results/repro_controlled_sglang_c1 \
  portable_plugin_repro/run_controlled_sglang_comparison.sh
```

The accepted MAC environment knobs are centralized in
`portable_plugin_repro/env_mac_portable.sh`. The important defaults are:

```bash
MAC_ATTENTION_ENABLE=1
MAC_ATTENTION_PORTABLE_PLUGIN=1
MAC_ATTENTION_SGLANG_STRICT=1
MAC_PERSISTENT_PARALLEL_Z2_SCHEDULE=1
MAC_PERSISTENT_MIXED_MISSPACK_Z2=1
MAC_FUSE_HIT_TAIL_IN_MERGE=0
MAC_USE_FUSED_KV_ROPE=1
MAC_USE_FUSED_Q_PRESERVE_ROPE=1
```

## Remaining Gap

The portable plugin overhead is now essentially closed relative to the patched
SGLang best-so-far path. The remaining performance problem is still the core
kernel curve: full-KV and low-hit rows lag FlashInfer, and the jump from all-hit
to the first real miss (`hit=0.96875` for 32 query heads) is still larger than
we want.
