# MAC-Attention Workspace Rules

These rules apply to all MAC-Attention optimization work in this repository.

- Keep MAC math identical to the paper: fixed recent-query ring, pre-RoPE squared-L2 matching, threshold/rectification semantics, and no sorting or reranking in the match window.
- Keep miss and partial-miss work inside the single cooperative MAC decode kernel and device helpers.
- Do not add host-side all-miss routing, separate production kernels, production FlashInfer calls, or compile-time escape hatches.
- Do not add an ad hoc all-miss or `hit=0` special case. `hit=0` is the first point on the same MAC-Attention hit-ratio curve.
- The only allowed ad hoc fast path is all-hit, where no load balancing is needed.
- The design must transition smoothly across the full hit-ratio curve. Do not hop between disconnected modes as hit ratio changes.
- A higher hit ratio must not select a structurally more expensive path. Mixed-row redesigns must use a continuous cost model and be judged on whole-curve monotonic behavior, not isolated point wins.
- For standalone `batch=1,Hq=32,bench-mode=synthetic_head` exact-quota runs, do not include `hit=0.99` in default/report hit grids. It rounds to zero miss heads and is just all-hit under a misleading label; use `hit=0.96875` as the first one-miss-head point.
- Validate accepted optimizations on both the standalone benchmark and the SGLang/LongBench path.
- Keep `docs/OPTIMIZATION_LOG.md` updated for every meaningful accepted or rejected optimization attempt.
