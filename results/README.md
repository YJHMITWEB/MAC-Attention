# Curated Results

This directory contains the small result bundles that are meant to travel with
the portable MAC-Attention implementation.

## `portable_plugin_hook_gap_20260517`

No-CUDA-graph portable plugin checkpoint.

- Controlled SGLang production comparison on fixed LongBench prompts.
- FlashInfer baseline and portable MAC decode-log throughput.
- Comparison against the previous patched-SGLang best implementation.
- Standalone full-curve comparison against the previous best implementation.

Key production numbers, concurrency 1:

| context | FlashInfer tok/s | portable MAC tok/s | MAC / FlashInfer |
|---:|---:|---:|---:|
| 64K | 68.40 | 64.50 | 0.943x |
| 96K | 59.51 | 63.12 | 1.061x |
| 127K | 52.91 | 62.49 | 1.181x |

## `portable_mac_cudagraph_handoff_20260519`

Latest portable implementation handoff.

- Complete standalone no-CUDA-graph hit curve against FlashInfer plan+run wall
  timing.
- CUDA-graph SGLang production hit-curve sweep.
- LongBench CUDA-graph output/reference files.
- Command logs and manifests needed to reproduce the measurements.

CUDA graph is supported by the implementation but is not the default. Use
`MAC_DISABLE_CUDA_GRAPH=0` to opt in.
