# Curated Results

This directory contains the small result bundles that are meant to travel with
the portable MAC-Attention implementation.

## `portable_mac_cudagraph_handoff_20260519`

Latest portable implementation handoff.

- Complete standalone no-CUDA-graph hit curve against FlashInfer plan+run wall
  timing.
- Focused standalone CUDA-graph optimization A/B curves.
- Command/result manifests for the public standalone artifacts.

CUDA graph is supported by the implementation but is not the default. Use
`MAC_DISABLE_CUDA_GRAPH=0` to opt in when launching SGLang.
