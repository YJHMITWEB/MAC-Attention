# Portable MAC Standalone Result Bundle - 2026-05-19

This directory is the curated public result bundle for the portable
MAC-Attention implementation.

Included artifacts:

- `standalone_no_cudagraph_official_vs_best_20260517/`
  - Portable MAC no-CUDA-graph standalone hit curve, compared against
    FlashInfer plan+run timing and the previous patched-SGLang best code.
  - Primary summary file: `comparison_current_vs_previous_best.csv`.
- `standalone_tail_baseline_ab_20260519_093943_gilgamesh/`
  - Same-session standalone baseline for the focused warp-owned match-scan A/B.
- `standalone_warp_owned_match_clean_20260519_094341_gilgamesh/`
  - Focused standalone candidate curve showing whole-curve improvement before
    SGLang integration validation.
- `MANIFEST.txt`
  - File list captured when this public bundle was assembled.

The internal prompt-runner artifacts used during development are intentionally
not included here. Public users can reproduce standalone curves with
`portable_plugin_repro/run_standalone_full_curve.sh` and can launch SGLang with
`portable_plugin_repro/run_sglang_mac_server.sh`.
