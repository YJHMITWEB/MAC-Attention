# Current Design

## Goal

MAC-Attention should run as a portable SGLang plugin while preserving the same
kernel behavior and performance as the best tuned integration. Official SGLang
source files should remain unmodified; MAC-Attention is injected by launching
through `mac_attention.integrations.sglang.launch_server` with environment
variables.

## SGLang Integration Path

1. `launch_server.py` installs the environment-derived MAC config.
2. `hook_installer.py` patches SGLang modules in memory before SGLang starts.
3. `flashinfer_hooks.py`, `llama_hooks.py`, and `schedule_hooks.py` connect MAC
   metadata, Q/K/V preservation, and decode execution into SGLang's production
   flow.
4. Worker child processes import `child_process.py`, which reinstalls the same
   hooks in scheduler/model processes.
5. The decode path calls the MAC extension loader in `bridge.py`, which JIT
   builds the CUDA sources under `csrc/`.

The integration deliberately avoids a source patch overlay. The official SGLang
checkout is selected with `SGLANG_ROOT`, and this package is selected with
`MAC_ATTENTION_ROOT`.

## Decode Kernel Contract

The production decode work is performed by
`mac_decode_persistent.cu`. It is a single cooperative persistent kernel for
the MAC decode operation. It owns:

- matching against the fixed recent-query window,
- hit/miss decisions,
- copied contribution handling,
- full-KV recompute work for misses and mixed rows,
- partial-state production,
- online softmax state merge,
- output writeback,
- optional asynchronous cache update helpers.

The kernel must not route all-miss rows to a host-side fallback, separate
production kernel, or FlashInfer function. `hit=0` and partial-hit rows are the
same continuous MAC path. The only special fast path allowed by design is the
all-hit path, where no load balancing is needed.

## Current Performance-Oriented Modes

- `MAC_PERSISTENT_PARALLEL_Z2_SCHEDULE=1`
  - enables the z2-parallel schedule used by the current best result.
- `MAC_PERSISTENT_MIXED_MISSPACK_Z2=1`
  - packs mixed-row miss prefix work across z2 lanes to reduce the high-hit
    cliff and improve production LongBench throughput.
- `MAC_FUSE_HIT_TAIL_IN_MERGE=0`
  - disables hit-tail fusion in merge, which was found to hurt the first real
    miss point.
- `MAC_PERSISTENT_PARTIAL_FP32=1`
  - keeps partial-state math stable and aligned with accepted correctness.

These switches are still controlled by environment variables so profiling can
bisect behavior, but the public/default path should use the settings above.

## Matching Semantics

MAC math follows the paper:

- fixed recent-query ring,
- pre-RoPE squared-L2 matching,
- threshold-based hit decision with existing rectification semantics,
- no sorting or reranking inside the match window,
- `MAC_SEMANTIC_POS_AHEAD=256` and `MAC_LOOKBACK_TOKENS_LEFT=512` in the
  accepted LongBench configuration.

Synthetic standalone hit curves are only for kernel stress testing. LongBench
quality should be checked against the accepted reference output for model-level
behavior when changing matching or cache-update code.

## Known Remaining Gap

The latest accepted path reproduces the best tuned integration, but the kernel
is still weaker than FlashInfer in full-KV-heavy cases:

- `hit=0` is slower than FlashInfer plan+run at 64K, 96K, and 127K.
- low-hit rows are still below the theoretical speed implied by reduced FLOPs.
- high-hit production is strong, but the first real miss point remains a
  visible latency step relative to all-hit.

Future work should improve the continuous full-KV/mixed state contract,
producer memory traffic, and merge cost without adding hit-ratio-specific
escape paths.

