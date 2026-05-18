# Performance Baseline - 2026-05-17

These are the accepted reference numbers used for this public cleanup. They
come from the portable-plugin retake and the best tuned integration on the same
GPU class and software environment. FlashInfer standalone latency includes
plan plus run time.

## Standalone Portable Versus Previous Best

Full hit grid:

```text
contexts: 65536, 98304, 126976
batch: 1
hit rates: 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.875,0.9,0.95,0.96875,1.0
mode: synthetic_head
warmup: 12
iters: 60
FlashInfer timing: plan_run_wall
```

The portable plugin matched the previous best within run-to-run noise:

- mean delta versus previous best: `-0.70%` across 42 rows.
- a suspected 96K / `hit=0.3` outlier was rechecked and did not reproduce.

Selected previous-best standalone rows:

| context | hit | MAC ms | FlashInfer ms | MAC speedup |
|---:|---:|---:|---:|---:|
| 64K | 0 | 0.2961 | 0.2728 | 0.921x |
| 64K | 0.8 | 0.2519 | 0.2728 | 1.083x |
| 64K | 0.96875 | 0.2038 | 0.2728 | 1.339x |
| 64K | 1 | 0.0997 | 0.2728 | 2.736x |
| 96K | 0 | 0.4127 | 0.3424 | 0.830x |
| 96K | 0.8 | 0.2849 | 0.3424 | 1.202x |
| 96K | 0.96875 | 0.2178 | 0.3424 | 1.572x |
| 96K | 1 | 0.0987 | 0.3424 | 3.469x |
| 127K | 0 | 0.5009 | 0.4056 | 0.810x |
| 127K | 0.8 | 0.3189 | 0.4056 | 1.272x |
| 127K | 0.96875 | 0.2228 | 0.4056 | 1.820x |
| 127K | 1 | 0.0985 | 0.4056 | 4.118x |

## SGLang Production Portable Versus FlashInfer

Controlled LongBench samples, steady decode throughput:

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

Portable versus same-day hacked-SGLang MAC matched closely:

- mean delta: `+0.28%` across the 9 controlled production cells.

## Cleanup Guard Before Public Tree Split

After the first cleanup pass, correctness and performance guards were run:

- plugin/config tests: `10 passed`.
- persistent decode correctness: `14 passed`.
- hook smoke: hooks installed and `server_args_hooked=True`.
- SGLang production smoke at 64K / concurrency 1 / 32 generated tokens:
  portable MAC steady decode `62.23 tok/s`.

Standalone canary rows after cleanup remained in-family with previous best:

| context | hit | clean MAC ms | delta vs previous |
|---:|---:|---:|---:|
| 64K | 0 | 0.2953 | -0.28% |
| 64K | 0.8 | 0.2494 | -1.00% |
| 64K | 0.96875 | 0.1930 | -5.29% |
| 96K | 0 | 0.3899 | -5.52% |
| 96K | 0.8 | 0.2863 | +0.52% |
| 96K | 0.96875 | 0.2125 | -2.43% |
| 127K | 0 | 0.4854 | -3.10% |
| 127K | 0.8 | 0.3296 | +3.38% |
| 127K | 0.96875 | 0.2250 | +0.98% |

## Acceptance Rule For This Public Cleanup

The cleaned public tree should reproduce the same correctness and be within
normal benchmark noise of the current best. A regression larger than a few
percent on repeated standalone or controlled SGLang measurements should block
publication until explained.

