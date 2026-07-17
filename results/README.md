# Curated Results

This directory contains small, public result bundles for the portable
MAC-Attention implementation.

The README figures are generated from these CSVs by
`plot_portable_plugin_results.py` and written to
`assets/perf_update_20260717/`.

## `cuda_graph_hit_curves`

Standalone fused MAC-Attention CUDA-kernel hit-curve comparison against
FlashInfer, with both kernels captured into CUDA graphs and timed as graph
replay, measured on H100 80GB (HBM3).

- Files: `gqa8.csv` (`Hq=32`, `Hkv=4`, the Qwen3-30B shape) and `gqa4.csv`
  (`Hq=32`, `Hkv=8`)
- Batch sizes: `1`, `16`, `64`
- Context lengths: `32K`, `64K`, `96K`, `127K`
- Hit ratios: `0`, `0.1`, `0.2`, `0.3`, `0.4`, `0.5`, `0.6`, `0.7`,
  `0.8`, `0.875`, `0.9`, `0.95`, `0.96875`, `1.0`
- Synthetic per-head hit patterns (`--bench-mode synthetic_head`),
  `--partial-fp32`, warmup 12, iters 60.
- Reproduce with `portable_plugin_repro/run_standalone_full_curve.sh`.
- The README figure highlights the high-hit region (`95-99%+`), which is the
  common operating regime we observed in practical long-context datasets.
