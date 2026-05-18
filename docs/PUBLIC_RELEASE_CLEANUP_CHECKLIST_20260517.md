# Public Release Cleanup Checklist - 2026-05-17

## Completed In This Public Tree

- Created a standalone public-style repository layout under `OFFICIAL_GITHUB`.
- Kept MAC core code under `attention/` and removed private workspace cache
  files.
- Removed stale private path fallbacks from benchmark and reproduction scripts.
- Replaced private tuning-overlay docs with public README, runbook, design
  notes, and performance baseline.
- Kept the portable plugin as the primary SGLang integration path.
- Preserved current best defaults:
  - `MAC_PERSISTENT_PARALLEL_Z2_SCHEDULE=1`
  - `MAC_PERSISTENT_MIXED_MISSPACK_Z2=1`
  - `MAC_FUSE_HIT_TAIL_IN_MERGE=0`
  - `MAC_PERSISTENT_PARTIAL_FP32=1`

## Must Stay True Before Publication

- Official SGLang source remains unmodified.
- MAC args are passed through environment variables.
- Production decode remains a single self-contained MAC kernel path.
- No host-side `hit=0` or all-miss special case is introduced.
- No production FlashInfer call is introduced into the MAC path.
- `hit=0.99` stays out of default exact-quota hit curves.

## Validation Still Required After Any Cleanup

- `./portable_plugin_repro/run_correctness.sh`
- `./portable_plugin_repro/run_hook_smoke.sh`
- `./portable_plugin_repro/run_standalone_full_curve.sh`
- `./portable_plugin_repro/run_controlled_sglang_comparison.sh`

Compare against `docs/PERFORMANCE_BASELINE_20260517.md`.
