# Portable MAC CUDA-Graph Handoff Results - 2026-05-19

This directory is the curated result bundle for moving portable MAC-Attention
work to a new cluster.

Included artifacts:

- `sglang_production_cudagraph_full_hitcurve_20260519_002159/`
  - Latest completed full SGLang production CUDA-graph hit curve against the
    FlashInfer baseline.
  - Primary summary file: `comparison.csv`.
  - Original commands: `COMMANDS.txt`.
- `standalone_no_cudagraph_official_vs_best_20260517/`
  - Previous portable MAC no-CUDA-graph standalone hit curve, compared against
    FlashInfer plan+run and the previous patched-SGLang best code.
  - Primary summary file: `comparison_current_vs_previous_best.csv`.
- `standalone_tail_baseline_ab_20260519_093943_gilgamesh/`
  - Same-session standalone baseline for the focused warp-owned match-scan A/B.
- `standalone_warp_owned_match_clean_20260519_094341_gilgamesh/`
  - Focused standalone candidate curve showing whole-curve improvement before
    SGLang production validation.
- `sglang_cudagraph_warp_owned_match_focus_20260519_094950_gilgamesh/`
  - Focused SGLang CUDA-graph gate for the pre-barrier warp-owned candidate.
  - This run is intentionally included as a failure/caveat artifact: FlashInfer
    baseline completed, MAC served the first 64K hit=0.5 row, then the CUDA
    context failed before the next rows.
- `longbench_cuda_graph_20260518_191508/`
  - Latest LongBench MAC output and previous best-so-far reference output.
- `MANIFEST.txt`
  - File list captured when this bundle was assembled.

See `docs/PORTABLE_MAC_CUDAGRAPH_HANDOFF_20260519.md` for environment,
commands, status, and pending validation steps.
