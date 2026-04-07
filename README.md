# <img src="assets/icon.png" height="48" style="vertical-align: -13px;"> MAC-Attention

🎓**Accepted at MLSys 2026**🎓

📄 **Paper:** [arXiv:2604.00235](https://arxiv.org/abs/2604.00235)

**MAC-Attention** is a high-performance attention mechanism that reduces decoding overhead by **reusing attention computation across semantically similar tokens**.
This repository contains the **full reference implementation**, including:
- MAC “ring match” CUDA extension (`ext/macMatch.cu`)
- MAC prefill cache-update CUDA extension (`ext/mac_prefill_update_cache.cu`)
- the `mac_attention` Python package (in `attention/`) which JIT-builds the attention + rectification-cache ops

Important note on runtime critical path:
- **Match + attention** are on the decode critical path.
- **Prefill cache update** and **rectification+cache** are designed to run **asynchronously** (e.g., on a separate CUDA stream) and are **not on the critical path**.

![](assets/workflow.png)

## 🚀 Quick Start

Prereqs:
- NVIDIA H100 GPUs
- CUDA 12.8
- PyTorch with CUDA enabled (`torch.cuda.is_available() == True`)

Install `mac_attention` (editable):

```bash
python -m pip install -e attention
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Run the end-to-end example (first run will JIT-compile CUDA extensions):

```bash
python -u e2e_mac_workflow_example.py --steps 1
```

## 🗂️ Project Layout

```
MAC-Attention/
├── README.md
├── attention/                                   # Python package: mac_attention
│   ├── pyproject.toml
│   ├── examples/
│   │   └── bench_time_grid_min.py               # Minimal benchmark for MACDecode
│   ├── tests/                                   
│   └── src/mac_attention/
│       ├── attention/
│       │   ├── mac_decode.py                    
│       │   └── mac_rectification_cache.py       
│       ├── _jit/                                
│       └── _vendor/                             # Vendored CUDA/C++ (flashinfer-based)
├── ext/
│   ├── macMatch.cu                              # Standalone ring match (+ scheduler)
│   └── mac_prefill_update_cache.cu              # Standalone prefill cache update
├── bench_mac_match.py                           # Match benchmark (baseline vs macMatch)
├── bench_mac_kernel_latency_2x2.py             # Figure-style kernel-latency benchmark
├── bench_mac_prefill_update_cache.py            # Prefill cache update benchmark
├── bench_time_grid_mac_attention_speedup.py     # Paper-style sweep: MAC attention vs FlashInfer
├── bench_time_grid_mac_match_plan_attention.py  # Time grid: plan + attention
├── bench_time_grid_mac_rectification_cache.py   # Time grid: plan + rectification+cache
├── e2e_mac_workflow_example.py                  # End-to-end workflow example
├── plot_attn_speedup.py                         # Shared-axis speedup figure from benchmark CSV
├── plot_mac_kernel_latency_2x2.py              # Paper-style 2x2 kernel-latency figure
└── results/                                     # Generated CSVs (created on first run)
```

## 🛠️ Build / JIT Compilation

This repo uses `torch.utils.cpp_extension` and builds CUDA extensions on-demand.

### 1) Standalone extensions (`ext/`)

Built via `torch.utils.cpp_extension.load(...)` when you run:
- `bench_mac_match.py` (builds `ext/macMatch.cu`, and optionally a baseline `.cu`)
- `bench_mac_kernel_latency_2x2.py` (builds `ext/macMatch.cu`)
- `bench_mac_prefill_update_cache.py` (builds `ext/mac_prefill_update_cache.cu`)
- `bench_time_grid_mac_attention_speedup.py` (builds `ext/macMatch.cu`)
- `bench_time_grid_mac_match_plan_attention.py` (builds `ext/macMatch.cu`)
- `e2e_mac_workflow_example.py` (builds `ext/macMatch.cu`)

Build directory:
```text
${MAC_WORKSPACE_BASE:-$HOME}/.cache/mac/torch_extensions
```

### 2) `mac_attention` package CUDA ops (`attention/`)

Built when calling `.plan(...)` on:
- `mac_attention.MACDecodeWithPagedKVCacheWrapper`
- `mac_attention.MACRectificationCacheWithPagedKVCacheWrapper`

This happens in:
- `bench_mac_kernel_latency_2x2.py`
- `bench_time_grid_mac_attention_speedup.py`
- `bench_time_grid_mac_match_plan_attention.py`
- `bench_time_grid_mac_rectification_cache.py`
- `e2e_mac_workflow_example.py`

Cache root:
```text
${MAC_WORKSPACE_BASE:-$HOME}/.cache/mac
```

Useful knobs:
```bash
export MAC_WORKSPACE_BASE=$HOME                           # where to put build artifacts
export MAC_JIT_VERBOSE=1                                  
```

Force a clean rebuild:
```bash
rm -rf ${MAC_WORKSPACE_BASE:-$HOME}/.cache/mac
```

## 🧪 Sanity Check

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q attention/tests
```

## 📊 Benchmarks

All scripts write CSVs to `results/` (created automatically).

### 1) Match benchmark (baseline vs macMatch)

Script: `bench_mac_match.py`

What it measures (all in microseconds):
- schedule time (`*_schedule_us`)
- kernel time (`*_kernel_us`)
- total (`*_us = schedule_us + kernel_us`)
- correctness vs baseline:
  - exact output equality (`output_correctness`)
  - winner-distance sanity check (`l2_correctness`)

Default sweep:
- `R=64`
- `M ∈ {256,512,1024,2048}`
- `H ∈ {8,32}`
- `D=128`
- `N ∈ {1,2,4,8,16,32}`

Run:
```bash
python -u bench_mac_match.py
```

Output:
- `results/bench_mac_match_results.csv`

Notes:
- `--baseline` is optional and disabled by default.

### 2) Prefill cache update benchmark

Script: `bench_mac_prefill_update_cache.py`

Measures:
- prefill cache update time (`avg_us`)

Note: In the full system this is typically launched **asynchronously** and overlapped; its latency is **not** on the decode critical path.

Run:
```bash
python -u bench_mac_prefill_update_cache.py
```

Output:
- `results/bench_mac_prefill_update_cache_results.csv`

### 3) Paper-style 2x2 kernel-latency benchmark

Script: `bench_mac_kernel_latency_2x2.py`

Measures:
- panel **(a)**: `macMatch` kernel latency vs `flashinfer` decode-kernel latency
- panel **(b)**: load-balance planner study using **attention-kernel latency only** (`MAC Perfect`, `MAC w.LB`, `MAC w.o. LB`)
- panels **(c)** and **(d)**: MAC breakdown (`Match`, `Plan`, `Attention`) vs full-attention baseline

Default setup:
- panel **(a)**:
  - lengths: `512, 1024, 2048`
  - GQA configs: `8-2`, `32-8`, `40-10`
  - batch size: `1`
- panel **(b)**:
  - context lengths: `32K, 64K, 128K`
  - `sigma ∈ {0.1, 0.2, 0.3}`
  - GQA config: `32-8`
  - batch size: `1`
  - `MAC Perfect`: uniform KV access ratio `sigma`
  - `MAC w.LB`: per-head KV access ratio sampled from `Normal(mean=sigma, std=sigma)` and clamped to `[0,1]`
  - `MAC w.o. LB`: worst-case uniform KV access ratio `min(1, sigma + sigma)`
  - the plotted bars use `*_attn_us` only; planner latency is still emitted in the CSV for reference
- panels **(c)** and **(d)**:
  - contexts: `32K, 64K, 128K`
  - skip ratios: `0.99, 0.90, 0.80`
  - GQA configs: `32-8`, `64-8`
  - batch size: `4`

Run:
```bash
python -u bench_mac_kernel_latency_2x2.py
```

Outputs:
- `results/bench_mac_kernel_latency_panel_a.csv`
- `results/bench_mac_kernel_latency_panel_b.csv`
- `results/bench_mac_kernel_latency_breakdown.csv`

Plot:
```bash
python -u plot_mac_kernel_latency_2x2.py
```

Figure:
- `results/mac_kernel_latency_2x2.pdf`
- `results/mac_kernel_latency_2x2.png`

### 4) Time grid: plan + attention

Script: `bench_time_grid_mac_match_plan_attention.py`

Measures:
- `MACDecodeWithPagedKVCacheWrapper.plan(...)` time (`standalone_plan_time_us`)
- attention time (`standalone_attn_time_us`)

Run:
```bash
python -u bench_time_grid_mac_match_plan_attention.py
```

Output:
- `results/bench_time_grid_mac_match_plan_attention_results.csv`

### 5) Paper-style attention speedup benchmark

Script: `bench_time_grid_mac_attention_speedup.py`

Measures:
- `mac_match_schedule_us`
- `mac_match_kernel_us`
- `mac_match_us`
- `mac_plan_time_us`
- `mac_attn_time_us`
- `mac_total_time_us`
- `flashinfer_baseline_time_us`

Output schema:
- `batch_size`
- `context_length`
- `KV_Access`
- `mac_match_rows_per_stage`
- `mac_match_load_warps`
- `mac_match_schedule_us`
- `mac_match_kernel_us`
- `mac_match_us`
- `mac_plan_time_us`
- `mac_attn_time_us`
- `mac_total_time_us`
- `flashinfer_baseline_time_us`

Notes:
- This benchmark requires the `flashinfer` Python package for the baseline path.
- It uses the same sweep shape as the paper benchmark: `batch_size in {1,2,4,8,16,32,64}`, `context_length in {32768,65536,131072,262144}`, `KV_Access in {0.01,0.05,0.1,0.2,0.3,0.4}`.
- The plot uses `mac_total_time_us` as the end-to-end MAC critical-path latency.
- `mac_match_schedule_us` is emitted for visibility, but the plotted critical path follows the e2e workflow example and uses match kernel + plan + attention.

Run:
```bash
python -u bench_time_grid_mac_attention_speedup.py
```

Output:
- `results/bench_time_grid_results.csv`

Plot:
```bash
python -u plot_attn_speedup.py
```

Figure:
- `results/attn_speedup_row_from_csv_sharedaxes.pdf`

### 6) Time grid: plan + rectification+cache

Script: `bench_time_grid_mac_rectification_cache.py`

Measures:
- `MACRectificationCacheWithPagedKVCacheWrapper.plan(...)` time (`standalone_plan_time_us`)
- rectification+cache time (`standalone_macRectificationCache_time_us`)

Note: Rectification+cache is typically launched **asynchronously** and overlapped; it is **not** on the decode critical path.

Run:
```bash
python -u bench_time_grid_mac_rectification_cache.py
```

Output:
- `results/bench_time_grid_mac_rectification_cache_results.csv`

### 7) (Optional) Minimal smoke benchmark

```bash
python -u attention/examples/bench_time_grid_min.py
```

## 🔄 End-to-End Workflow Example

Script: `e2e_mac_workflow_example.py`

Order of operations:
1) **mac match** (`ext/macMatch.cu`)
   - inputs: `queries`, `query_cache`
   - outputs: `hit`, `left`, `idx`
2) **mac attention** (`MACDecodeWithPagedKVCacheWrapper`)
   - inputs: `queries`, paged KV cache, ring `attn_cache`/`lse_cache`, and `attn_start_pos = left`
   - outputs: full `(o, lse)`
3) **macRectificationCache** (`MACRectificationCacheWithPagedKVCacheWrapper`)
   - inputs: `queries`, paged KV cache, ring caches, and full `(o, lse)`
   - outputs: windowed `(o, lse)` (and updates ring caches)
   - uses a fixed `window_left` (does not depend on previous kernels)
   - typically launched **asynchronously** and overlapped; **not** on the decode critical path

Run:
```bash
python -u e2e_mac_workflow_example.py --steps 2
```
