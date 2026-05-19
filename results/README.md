# Curated Results

This directory contains small, public result bundles for the portable
MAC-Attention implementation.

The README figures are generated from these CSVs by
`plot_portable_plugin_results.py` and written to
`assets/portable_update_20260519/`.

## `no_cuda_graph_hit_curve`

Standalone MAC-vs-FlashInfer hit-curve comparison with CUDA graph disabled.

- Primary file: `comparison.csv`
- Context lengths: `64K`, `72K`, `96K`, `127K`
- Hit ratios: `0`, `0.1`, `0.2`, `0.3`, `0.4`, `0.5`, `0.6`, `0.7`,
  `0.8`, `0.875`, `0.9`, `0.95`, `0.96875`, `1.0`
- FlashInfer timing includes plan plus run wall time.
