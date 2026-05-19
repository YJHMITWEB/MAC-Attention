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

## `cuda_graph_standalone_ab`

Standalone CUDA-graph hit-curve data from the portable kernel tuning work. In
the README figure, FlashInfer is plotted as the parity baseline and
MAC-Attention is plotted per context length relative to that baseline.

- `tail_baseline.csv`: baseline curve from the same tuning session.
- `warp_owned_match.csv`: MAC-Attention curve used in the README figure.

CUDA graph is supported by the implementation but is not the default. Use
`MAC_DISABLE_CUDA_GRAPH=0` to opt in when launching SGLang.
