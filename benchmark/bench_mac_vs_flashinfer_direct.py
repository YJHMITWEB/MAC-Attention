#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch

import flashinfer

ROOT = Path(os.environ.get("MAC_ATTENTION_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_ATTENTION_ROOT = ROOT / "attention"
ATTENTION_ROOT = Path(
    os.environ.get("MAC_ATTENTION_ROOT", str(DEFAULT_ATTENTION_ROOT))
).resolve()
sys.path.insert(0, str(ATTENTION_ROOT))
sys.path.insert(0, str(ATTENTION_ROOT / "src"))

from mac_attention.integrations.sglang.bridge import load_mac_persistent_decode_extension
from tools.bench_mac_persistent_decode import _bench_mode_value, _make_case, _workspace


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(",", " ").split() if x.strip()]


def parse_masks(text: str) -> list[int]:
    masks: list[int] = []
    for raw in text.replace(",", " ").split():
        token = raw.strip()
        if not token:
            continue
        masks.append(int(token, 0))
    return masks


def warn_degenerate_hit_grid(args: argparse.Namespace, batch_values: list[int]) -> None:
    if getattr(args, "bench_miss_masks", ""):
        return
    bench_mode = _bench_mode_value(args.bench_mode)
    if bench_mode not in (1, 2) or args.bench_hit_rate_std != 0.0:
        return
    hit_values = parse_floats(args.hit_rates)
    unit_name = "Hkv groups" if bench_mode == 2 else "Hq heads"
    for batch in batch_values:
        unit_total = batch * (args.hkv if bench_mode == 2 else args.hq)
        if unit_total <= 0:
            continue
        first_miss_hit = 1.0 - 1.0 / float(unit_total)
        for hit_rate in hit_values:
            miss_units = _bench_target_miss_units(hit_rate, unit_total)
            if 0.0 < 1.0 - hit_rate and miss_units == 0:
                print(
                    "warning: requested hit={hit:g} rounds to all-hit "
                    "(0 miss units) for exact {mode} quota with batch={batch}, "
                    "{unit_name}={unit_total}. Omit it from reported hit curves; "
                    "use hit={first:g} for the first real miss point or increase batch.".format(
                        hit=hit_rate,
                        mode=args.bench_mode,
                        batch=batch,
                        unit_name=unit_name,
                        unit_total=unit_total,
                        first=first_miss_hit,
                    ),
                    file=sys.stderr,
                    flush=True,
                )


def _u32(x: int) -> int:
    return x & 0xFFFFFFFF


def _mix_u32(x: int) -> int:
    x = _u32(x)
    x ^= x >> 16
    x = _u32(x * 0x7FEB352D)
    x ^= x >> 15
    x = _u32(x * 0x846CA68B)
    x ^= x >> 16
    return _u32(x)


def _uniform01_u32(x: int) -> float:
    return float(_mix_u32(x) & 0x00FFFFFF) * (1.0 / 16777216.0)


def _pseudo_normal_u32(x: int) -> float:
    s = (
        _uniform01_u32(x)
        + _uniform01_u32(_u32(x ^ 0x9E3779B9))
        + _uniform01_u32(_u32(x ^ 0x85EBCA6B))
        + _uniform01_u32(_u32(x ^ 0xC2B2AE35))
    )
    return (s - 2.0) * 1.7320508075688772


def _bench_target_miss_units(hit_rate: float, unit_total: int) -> int:
    if unit_total <= 0:
        return 0
    miss_rate = min(max(1.0 - hit_rate, 0.0), 1.0)
    miss_units = int(math.floor(miss_rate * float(unit_total) + 0.5))
    return min(max(miss_units, 0), unit_total)


def _cache_end_for_pos(pos: int, semantic_pos_ahead: int) -> int:
    end = pos + 1 - semantic_pos_ahead
    return end if end > 0 else 0


def _pos_to_ring_slot(past_len: int, capacity: int, pos: int) -> int:
    if past_len < capacity:
        return pos
    base = past_len - capacity
    tail = past_len % capacity
    order_idx = pos - base
    slot = order_idx + tail
    if slot >= capacity:
        slot -= capacity
    if slot < 0:
        slot += capacity
    return slot


def _synthetic_hit_and_slot(
    args: argparse.Namespace,
    *,
    bench_mode: int,
    batch: int,
    hq: int,
    hkv: int,
    past_len: int,
    n: int,
    hq_i: int,
    hkv_i: int,
    hit_rate: float,
) -> tuple[bool, int]:
    base_key = _u32(
        int(args.bench_seed)
        ^ _u32(0 * 0x9E3779B9)
        ^ _u32(past_len * 0x85EBCA6B)
    )
    bench_head_or_group = hkv_i if bench_mode == 2 else hq_i
    key = _u32(base_key ^ _u32(n * 0xC2B2AE35) ^ _u32(bench_head_or_group * 0x27D4EB2F))
    effective_hit_rate = min(
        max(hit_rate + args.bench_hit_rate_std * _pseudo_normal_u32(_u32(key ^ 0x4CF5AD43)), 0.0),
        1.0,
    )
    if args.bench_hit_rate_std == 0.0:
        unit_total = batch * (hkv if bench_mode == 2 else hq)
        unit = n * (hkv if bench_mode == 2 else hq) + (hkv_i if bench_mode == 2 else hq_i)
        miss_units = _bench_target_miss_units(hit_rate, unit_total)
        offset = int(base_key % unit_total) if unit_total > 0 else 0
        rotated = unit + offset
        if rotated >= unit_total:
            rotated -= unit_total
        hit = past_len > 0 and args.capacity > 0 and not (rotated < miss_units)
    else:
        hit = (
            past_len > 0
            and args.capacity > 0
            and _uniform01_u32(_u32(key ^ 0x165667B1)) < effective_hit_rate
        )
    if not hit:
        return False, -1

    max_lag = min(args.capacity, past_len)
    new_end = _cache_end_for_pos(past_len, args.semantic_pos_ahead)
    lag_f = args.bench_match_lag_mean + args.bench_match_lag_std * _pseudo_normal_u32(
        _u32(key ^ 0xD3A2646C)
    )
    if args.bench_skip_ratio >= 0.0:
        skip_ratio = min(
            max(
                args.bench_skip_ratio
                + args.bench_skip_ratio_std * _pseudo_normal_u32(_u32(key ^ 0xFD7046C5)),
                0.0,
            ),
            0.999999,
        )
        target_prefix_end = int(round(skip_ratio * float(past_len)))
        if target_prefix_end > new_end - 1:
            target_prefix_end = new_end - 1
        if target_prefix_end < 0:
            target_prefix_end = 0
        lag = int(round(float(new_end - target_prefix_end)))
        if lag < 1:
            lag = 1
        if lag > past_len:
            lag = past_len
    else:
        lag = int(round(lag_f))
        if lag < 1:
            lag = 1
        if lag > max_lag:
            lag = max_lag
    if max_lag <= 0 or new_end <= 0:
        return False, -1
    match_pos = past_len - lag
    if match_pos < 0:
        match_pos = 0
    return True, _pos_to_ring_slot(past_len, args.capacity, match_pos)


def apply_synthetic_cache_pattern(
    case: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    batch: int,
    kv_len: int,
    hit_rate: float,
    bench_mode: int,
) -> str:
    query_cache = case["query_cache"]
    q = case["q"]
    group = int(case["group"])
    past_len = kv_len - 1
    forced_mask = int(getattr(args, "bench_force_miss_mask", -1))
    if forced_mask >= 0:
        # Start from all miss for the forced mask. In benchmark-off mode the
        # kernel will use the real match scan, so untouched lanes must stay far
        # from q instead of inheriting the all-hit cache pattern.
        query_cache.normal_()
        query_cache.add_(10)
        lane_mask = forced_mask & ((1 << min(group, 30)) - 1)
        with torch.no_grad():
            for n in range(batch):
                for hkv_i in range(args.hkv):
                    for lane_i in range(group):
                        if (lane_mask >> lane_i) & 1:
                            continue
                        hq_i = hkv_i * group + lane_i
                        # Any valid slot with identical pre-RoPE q creates the same MAC hit
                        # semantics as the exact-quota synthetic path. Keep it deterministic
                        # so per-mask rows are directly comparable across contexts.
                        slot = _pos_to_ring_slot(past_len, args.capacity, max(past_len - 1, 0))
                        if slot >= 0:
                            query_cache[n, slot, hq_i, :].copy_(q[n, hq_i, :])
        return f"forced_miss_mask=0x{lane_mask:x}"

    if bench_mode == 0:
        return "hit_pattern"
    if args.synthetic_cache_pattern == "all_hit":
        case["query_cache"].copy_(case["q"][:, None, :, :].expand_as(case["query_cache"]))
        return "all_hit"
    if args.synthetic_cache_pattern == "all_miss":
        return "all_miss"

    with torch.no_grad():
        for n in range(batch):
            for hkv_i in range(args.hkv):
                for lane_i in range(group):
                    hq_i = hkv_i * group + lane_i
                    hit, slot = _synthetic_hit_and_slot(
                        args,
                        bench_mode=bench_mode,
                        batch=batch,
                        hq=args.hq,
                        hkv=args.hkv,
                        past_len=past_len,
                        n=n,
                        hq_i=hq_i,
                        hkv_i=hkv_i,
                        hit_rate=hit_rate,
                    )
                    if hit and slot >= 0:
                        query_cache[n, slot, hq_i, :].copy_(q[n, hq_i, :])
    return "faithful"


def bench_flashinfer(
    case: dict[str, torch.Tensor],
    *,
    hq: int,
    hkv: int,
    dim: int,
    warmup: int,
    iters: int,
    use_graph: bool = False,
) -> dict[str, float]:
    q = case["q"]
    k = case["k"]
    v = case["v"]
    batch = int(q.shape[0])
    kv_len = int(k.shape[0] // batch)
    dtype = q.dtype
    page_size = 1
    k_cache = k.reshape(batch * kv_len, page_size, hkv, dim)
    v_cache = v.reshape(batch * kv_len, page_size, hkv, dim)
    indptr = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * kv_len
    indices = torch.arange(batch * kv_len, device=q.device, dtype=torch.int32)
    last_page_len = torch.ones(batch, device=q.device, dtype=torch.int32)
    workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
    out = torch.empty_like(q)
    lse = torch.empty(batch, hq, device=q.device, dtype=torch.float32)
    wrapper_kwargs = {"kv_layout": "NHD", "use_cuda_graph": use_graph}
    if use_graph:
        # CUDA-graph mode requires persistent plan buffers.
        wrapper_kwargs.update(
            paged_kv_indptr_buffer=indptr,
            paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=last_page_len,
        )
    try:
        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            backend="auto",
            **wrapper_kwargs,
        )
    except TypeError:
        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            **wrapper_kwargs,
        )
    def plan() -> None:
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            hq,
            hkv,
            dim,
            page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            sm_scale=1.0 / math.sqrt(dim),
        )

    def launch() -> None:
        wrapper.run(q, (k_cache, v_cache), out=out, lse=lse, return_lse=True)

    plan()
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    if use_graph:
        # Plan cost is paid once at capture time; steady-state serving replays
        # the captured graph, so time graph replay for every reported metric.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start_wall = time.perf_counter()
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        replay_wall_ms = 1000.0 * (time.perf_counter() - start_wall) / max(iters, 1)
        replay_event_ms = float(start.elapsed_time(end) / max(iters, 1))
        return {
            "flashinfer_run_event_ms": replay_event_ms,
            "flashinfer_plan_run_event_ms": replay_event_ms,
            "flashinfer_plan_run_wall_ms": replay_wall_ms,
        }

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    torch.cuda.synchronize()
    run_event_ms = float(start.elapsed_time(end) / max(iters, 1))

    for _ in range(warmup):
        plan()
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start.record()
    for _ in range(iters):
        plan()
        launch()
    end.record()
    torch.cuda.synchronize()
    plan_run_wall_ms = 1000.0 * (time.perf_counter() - start_wall) / max(iters, 1)
    plan_run_event_ms = float(start.elapsed_time(end) / max(iters, 1))
    return {
        "flashinfer_run_event_ms": run_event_ms,
        "flashinfer_plan_run_event_ms": plan_run_event_ms,
        "flashinfer_plan_run_wall_ms": plan_run_wall_ms,
    }


def bench_mac(ext, args: argparse.Namespace, batch: int, kv_len: int, hit_rate: float, tile_tokens: int) -> dict[str, object]:
    forced_mask = int(getattr(args, "bench_force_miss_mask", -1))
    bench_mode = _bench_mode_value(args.bench_mode)
    case = _make_case(
        batch=batch,
        kv_len=kv_len,
        hq=args.hq,
        hkv=args.hkv,
        dim=args.dim,
        capacity=args.capacity,
        hit_pattern="all_miss" if bench_mode != 0 else "all_hit",
    )
    synthetic_cache_pattern = apply_synthetic_cache_pattern(
        case, args, batch=batch, kv_len=kv_len, hit_rate=hit_rate, bench_mode=bench_mode
    )
    workspace = _workspace(
        batch,
        args.hq,
        args.hkv,
        case["group"],
        args.dim,
        args.capacity,
        args.max_context,
        tile_tokens,
        args.match_tile_slots,
        args.semantic_pos_ahead,
        args.front_sink_tokens,
        torch.float32 if args.partial_fp32 else torch.bfloat16,
    )
    def launch() -> None:
        ext.mac_persistent_decode_bf16(
            case["q"],
            case["q"],
            case["k"],
            case["v"],
            case["req_to_token"],
            case["req_ids"],
            case["past_lens"],
            case["out_cache_loc"],
            case["query_cache"],
            case["attn_cache"],
            case["lse_cache"],
            case["out"],
            case["out_lse"],
            *workspace,
            args.max_context,
            tile_tokens,
            args.match_tile_slots,
            args.semantic_pos_ahead,
            args.front_sink_tokens,
            0,
            args.lookback_right,
            0,
            args.threshold,
            1.0 / math.sqrt(args.dim),
            args.debug,
            bench_mode,
            hit_rate,
            args.bench_hit_rate_std,
            args.bench_skip_ratio,
            args.bench_skip_ratio_std,
            args.bench_match_lag_mean,
            args.bench_match_lag_std,
            args.bench_seed,
            forced_mask,
            0,
            case["query_cache"],
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        )

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    if args.cuda_graph:
        # Steady-state serving replays a captured graph; time replay directly.
        # The cooperative persistent kernel is graph-capturable.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        launch_once = graph.replay
    else:
        launch_once = launch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start.record()
    for _ in range(args.iters):
        launch_once()
    end.record()
    torch.cuda.synchronize()
    wall_ms = 1000.0 * (time.perf_counter() - start_wall) / max(args.iters, 1)
    event_ms = float(start.elapsed_time(end) / max(args.iters, 1))
    counts = [int(x) for x in workspace[22].detach().cpu().tolist()]
    cycles = [int(x) for x in workspace[-1].detach().cpu().tolist()]
    metrics = _workspace_cost_metrics(workspace, args, batch)
    row: dict[str, object] = {
        "batch": batch,
        "kv_len": kv_len,
        "context_len": kv_len - 1,
        "hit_rate": hit_rate,
        "synthetic_cache_pattern": synthetic_cache_pattern,
        "tile_tokens": tile_tokens,
        "mac_event_ms": event_ms,
        "mac_wall_ms": wall_ms,
        "mac_ms": wall_ms if args.flashinfer_baseline_timing == "plan_run_wall" else event_ms,
        "complete_rows": counts[0],
        "group_tail_rows": counts[1],
        "hit_tail_rows": counts[2],
        "fallback_reduce_groups": counts[3],
        "balanced_tile_tokens": counts[4],
        "non_hit_groups": counts[5],
        "full_fallback_groups": counts[6],
        "max_full_fallback_reduce_chunks": counts[7],
        "max_mixed_reduce_chunks": counts[8],
    }
    row.update(metrics)
    if args.debug >= 2 and len(cycles) >= 12:
        detail_total = max(cycles[11] - cycles[0], 0)
        tail_before_reduce = (
            len(cycles) >= 14
            and cycles[13] >= cycles[6]
            and cycles[7] >= cycles[13]
            and cycles[8] >= cycles[7]
        )
        row["phase_order"] = "tail_before_reduce" if tail_before_reduce else "reduce_before_tail"
        detail_ranges = [
            ("match_pct", 0, 1),
            ("prepare_pct", 1, 2),
            ("complete_pct", 2, 6),
        ]
        if tail_before_reduce:
            detail_ranges.extend(
                [
                    ("tail_pct_detail", 6, 13),
                    ("reduce_pct_detail", 13, 7),
                ]
            )
        else:
            detail_ranges.extend(
                [
                    ("reduce_pct_detail", 6, 7),
                    ("tail_pct_detail", 7, 8),
                ]
            )
        detail_ranges.extend(
            [
                ("fallback_group_merge_pct", 8, 9),
                ("merge_write_pct", 9, 10),
                ("cache_subtract_pct", 10, 11),
            ]
        )
        for name, i, j in detail_ranges:
            row[name] = (100.0 * max(cycles[j] - cycles[i], 0) / detail_total) if detail_total else 0.0
        if not row["fallback_reduce_groups"]:
            row["reduce_pct_detail"] = 0.0
        if len(cycles) >= 13:
            reduce_sync_base = cycles[13] if tail_before_reduce else cycles[6]
            row["reduce_sync_wait_pct"] = (
                100.0 * max(cycles[12] - reduce_sync_base, 0) / detail_total
            ) if detail_total and row["fallback_reduce_groups"] else 0.0
            row["reduce_compute_pct"] = (
                100.0 * max(cycles[7] - cycles[12], 0) / detail_total
            ) if detail_total and row["fallback_reduce_groups"] else 0.0
    return row


def _workspace_cost_metrics(
    workspace: list[torch.Tensor], args: argparse.Namespace, batch: int
) -> dict[str, object]:
    head_hit = workspace[3][:batch].detach().to(torch.bool).cpu()
    head_rect_start = workspace[7][:batch].detach().cpu()
    head_new_end = workspace[8][:batch].detach().cpu()
    group_rect_begin = workspace[9][:batch].detach().cpu()
    group_rect_end = workspace[10][:batch].detach().cpu()
    group_rect_tiles = workspace[11][:batch].detach().cpu()
    group_mode = workspace[15][:batch].detach().cpu()
    complete_task_go = workspace[16]
    complete_task_tile = workspace[17]
    task_counts = workspace[22].detach().cpu()
    partial_rect_o = workspace[23]

    hkv = int(args.hkv)
    hq = int(args.hq)
    group = max(1, hq // max(1, hkv))
    dim = int(args.dim)
    partial_state_bytes = dim * int(partial_rect_o.element_size()) + 4

    head_hit_by_group = head_hit.reshape(batch, hkv, group)
    head_tokens = (head_new_end - head_rect_start).clamp_min(0).reshape(batch, hkv, group)
    rect_tokens = (group_rect_end - group_rect_begin).clamp_min(0)
    group_split = group_rect_tiles < 0
    scheduled_head_slots = torch.where(
        group_split.unsqueeze(-1),
        head_tokens,
        rect_tokens.unsqueeze(-1).expand_as(head_tokens),
    )
    miss_by_group = ~head_hit_by_group
    miss_count = miss_by_group.sum(dim=-1)
    miss_mask = torch.zeros((batch, hkv), dtype=torch.int64)
    for lane in range(min(group, 16)):
        miss_mask += miss_by_group[..., lane].to(torch.int64) << lane

    hist_bins = torch.bincount(miss_count.reshape(-1), minlength=group + 1)
    mask_hist_bins = torch.bincount(miss_mask.reshape(-1), minlength=1 << min(group, 4))
    mixed = group_mode == 2
    full = group_mode == 1

    counts = [int(x) for x in task_counts.tolist()]
    complete_rows = counts[0] if counts else 0
    group_total = batch * hkv
    if complete_rows > 0:
        task_go = complete_task_go[:complete_rows].detach().cpu()
        task_tile = complete_task_tile[:complete_rows].detach().cpu()
        task_base_go = task_go.clone()
        encoded_mixed_z2 = (task_go >= group_total) & (task_go < group_total * 2)
        task_base_go[encoded_mixed_z2] -= group_total
        task_is_group = task_go >= 0
        task_valid_group = task_is_group & (task_base_go >= 0) & (task_base_go < group_total)
        group_mode_flat = group_mode.reshape(-1)
        miss_count_flat = miss_count.reshape(-1)
        miss_mask_flat = miss_mask.reshape(-1)
        task_mode = torch.full_like(task_go, -1)
        task_miss_count = torch.zeros_like(task_go)
        task_miss_mask = torch.zeros_like(task_go)
        if task_valid_group.any():
            task_mode[task_valid_group] = group_mode_flat[task_base_go[task_valid_group]]
            task_miss_count[task_valid_group] = miss_count_flat[
                task_base_go[task_valid_group]
            ].to(task_miss_count.dtype)
            task_miss_mask[task_valid_group] = miss_mask_flat[
                task_base_go[task_valid_group]
            ].to(task_miss_mask.dtype)
        complete_group_rows = int((task_go >= 0).sum().item())
        complete_head_rows = int((task_go < 0).sum().item())
        complete_mixed_z2_rows = int(encoded_mixed_z2.sum().item())
        complete_full_group_rows = int((task_valid_group & (task_mode == 1)).sum().item())
        complete_mixed_group_rows = int(
            (task_valid_group & (task_mode == 2) & (~encoded_mixed_z2)).sum().item()
        )
        complete_mac_hit_group_rows = int((task_valid_group & (task_mode == 0)).sum().item())
        materialized_partial_rect_states = complete_head_rows + complete_group_rows * group
        active_partial_rect_states = complete_head_rows
        if task_valid_group.any():
            rect_begin_flat = group_rect_begin.reshape(-1)
            rect_end_flat = group_rect_end.reshape(-1)
            head_rect_start_by_group = head_rect_start.reshape(batch, hkv, group)
            head_new_end_by_group = head_new_end.reshape(batch, hkv, group)
            group_task_tile_counts = torch.zeros(group_total, dtype=torch.int64)
            for go_i in task_base_go[task_valid_group].tolist():
                if 0 <= int(go_i) < group_total:
                    group_task_tile_counts[int(go_i)] += 1
            for go_i in torch.nonzero(group_task_tile_counts, as_tuple=False).flatten().tolist():
                group_task_tile_counts[go_i] = max(
                    int(group_task_tile_counts[go_i].item()),
                    int(task_tile[task_valid_group & (task_base_go == go_i)].max().item()) + 1,
                )
            for idx in torch.nonzero(task_valid_group, as_tuple=False).flatten().tolist():
                go_i = int(task_base_go[idx].item())
                mode_i = int(task_mode[idx].item())
                if mode_i in (0, 1):
                    active_partial_rect_states += group
                    continue
                tile_count = max(int(group_task_tile_counts[go_i].item()), 1)
                rect_tokens_i = max(int(rect_end_flat[go_i].item() - rect_begin_flat[go_i].item()), 0)
                tile_tokens_i = max((rect_tokens_i + tile_count - 1) // tile_count, 1)
                begin_i = int(rect_begin_flat[go_i].item()) + int(task_tile[idx].item()) * tile_tokens_i
                end_i = min(begin_i + tile_tokens_i, int(rect_end_flat[go_i].item()))
                n_i = go_i // hkv
                hkv_i = go_i % hkv
                active_heads = 0
                for lane in range(group):
                    head_begin_i = int(head_rect_start_by_group[n_i, hkv_i, lane].item())
                    head_end_i = int(head_new_end_by_group[n_i, hkv_i, lane].item())
                    if end_i > head_begin_i and begin_i < head_end_i:
                        active_heads += 1
                active_partial_rect_states += active_heads
        inactive_partial_rect_states = max(
            materialized_partial_rect_states - active_partial_rect_states, 0
        )
        partial_rect_states = int(
            materialized_partial_rect_states
        )
        group_task_mask_hist_bins = torch.bincount(
            task_miss_mask[task_valid_group].reshape(-1),
            minlength=1 << min(group, 4),
        )
    else:
        complete_group_rows = 0
        complete_head_rows = 0
        complete_mixed_z2_rows = 0
        complete_full_group_rows = 0
        complete_mixed_group_rows = 0
        complete_mac_hit_group_rows = 0
        materialized_partial_rect_states = 0
        active_partial_rect_states = 0
        inactive_partial_rect_states = 0
        partial_rect_states = 0
        group_task_mask_hist_bins = torch.zeros(1 << min(group, 4), dtype=torch.int64)

    rect_state_reads = int(group_rect_tiles.clamp_min(0).sum().item() * group)
    active_phase4_rect_state_reads = 0
    rect_tiles_flat = group_rect_tiles.reshape(-1)
    rect_begin_flat = group_rect_begin.reshape(-1)
    rect_end_flat = group_rect_end.reshape(-1)
    group_mode_flat_for_reads = group_mode.reshape(-1)
    head_rect_start_by_group = head_rect_start.reshape(batch, hkv, group)
    head_new_end_by_group = head_new_end.reshape(batch, hkv, group)
    for go_i in range(group_total):
        tiles_i = int(max(int(rect_tiles_flat[go_i].item()), 0))
        if tiles_i <= 0:
            continue
        mode_i = int(group_mode_flat_for_reads[go_i].item())
        if mode_i in (0, 1):
            active_phase4_rect_state_reads += tiles_i * group
            continue
        rect_tokens_i = max(int(rect_end_flat[go_i].item() - rect_begin_flat[go_i].item()), 0)
        tile_tokens_i = max((rect_tokens_i + tiles_i - 1) // tiles_i, 1)
        n_i = go_i // hkv
        hkv_i = go_i % hkv
        for tile_i in range(tiles_i):
            begin_i = int(rect_begin_flat[go_i].item()) + tile_i * tile_tokens_i
            end_i = min(begin_i + tile_tokens_i, int(rect_end_flat[go_i].item()))
            for lane in range(group):
                head_begin_i = int(head_rect_start_by_group[n_i, hkv_i, lane].item())
                head_end_i = int(head_new_end_by_group[n_i, hkv_i, lane].item())
                if end_i > head_begin_i and begin_i < head_end_i:
                    active_phase4_rect_state_reads += 1
    inactive_phase4_rect_state_reads = max(rect_state_reads - active_phase4_rect_state_reads, 0)
    mixed_scheduled = scheduled_head_slots[mixed].sum() if mixed.any() else torch.tensor(0)
    mixed_actual = head_tokens[mixed].sum() if mixed.any() else torch.tensor(0)
    mixed_miss_tokens = head_tokens[miss_by_group & mixed.unsqueeze(-1)].sum() if mixed.any() else torch.tensor(0)
    mixed_sched_slots = scheduled_head_slots[mixed].sum() if mixed.any() else torch.tensor(0)
    mixed_unused_slots = (
        (scheduled_head_slots[mixed] - head_tokens[mixed]).clamp_min(0).sum()
        if mixed.any()
        else torch.tensor(0)
    )

    def scalar(value: torch.Tensor) -> int:
        return int(value.item())

    metrics: dict[str, object] = {
        "miss_head_count": scalar(miss_by_group.sum()),
        "miss_count_0_groups": int(hist_bins[0].item()) if hist_bins.numel() > 0 else 0,
        "miss_count_1_groups": int(hist_bins[1].item()) if hist_bins.numel() > 1 else 0,
        "miss_count_2_groups": int(hist_bins[2].item()) if hist_bins.numel() > 2 else 0,
        "miss_count_3_groups": int(hist_bins[3].item()) if hist_bins.numel() > 3 else 0,
        "miss_count_4_groups": int(hist_bins[4].item()) if hist_bins.numel() > 4 else 0,
        "mixed_miss1_groups": scalar(((mixed) & (miss_count == 1)).sum()),
        "mixed_miss2_groups": scalar(((mixed) & (miss_count == 2)).sum()),
        "mixed_miss3_groups": scalar(((mixed) & (miss_count == 3)).sum()),
        "full_miss_groups": scalar(full.sum()),
        "scheduled_head_slot_tokens": scalar(scheduled_head_slots.sum()),
        "actual_head_tokens": scalar(head_tokens.sum()),
        "active_miss_head_tokens": scalar(head_tokens[miss_by_group].sum()),
        "mixed_scheduled_head_slot_tokens": scalar(mixed_scheduled),
        "mixed_actual_head_tokens": scalar(mixed_actual),
        "mixed_active_miss_head_tokens": scalar(mixed_miss_tokens),
        "mixed_unused_head_slot_ratio": (
            float((mixed_unused_slots.float() / mixed_sched_slots.clamp_min(1).float()).item())
            if mixed_sched_slots.numel() > 0
            else 0.0
        ),
        "complete_group_rows": complete_group_rows,
        "complete_head_rows": complete_head_rows,
        "complete_mixed_z2_rows": complete_mixed_z2_rows,
        "complete_full_group_rows": complete_full_group_rows,
        "complete_mixed_group_rows": complete_mixed_group_rows,
        "complete_mac_hit_group_rows": complete_mac_hit_group_rows,
        "estimated_materialized_partial_rect_states": materialized_partial_rect_states,
        "estimated_active_partial_rect_states": active_partial_rect_states,
        "estimated_inactive_partial_rect_states": inactive_partial_rect_states,
        "estimated_inactive_partial_rect_state_ratio": (
            float(inactive_partial_rect_states) / float(max(materialized_partial_rect_states, 1))
        ),
        "estimated_partial_rect_states": partial_rect_states,
        "estimated_partial_rect_bytes": partial_rect_states * partial_state_bytes,
        "estimated_phase4_rect_state_reads": rect_state_reads,
        "estimated_phase4_rect_read_bytes": rect_state_reads * partial_state_bytes,
        "estimated_active_phase4_rect_state_reads": active_phase4_rect_state_reads,
        "estimated_inactive_phase4_rect_state_reads": inactive_phase4_rect_state_reads,
        "estimated_inactive_phase4_rect_read_ratio": (
            float(inactive_phase4_rect_state_reads) / float(max(rect_state_reads, 1))
        ),
        "miss_mask_hist": ";".join(
            f"{idx}:{int(count.item())}" for idx, count in enumerate(mask_hist_bins) if int(count.item()) != 0
        ),
        "group_task_miss_mask_hist": ";".join(
            f"{idx}:{int(count.item())}"
            for idx, count in enumerate(group_task_mask_hist_bins)
            if int(count.item()) != 0
        ),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="65536,98304,126976")
    parser.add_argument("--batch", default="1,2,4")
    parser.add_argument("--hit-rates", default="0,0.5,0.75,0.875,1.0")
    parser.add_argument("--tile-tokens", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--semantic-pos-ahead", type=int, default=256)
    parser.add_argument("--front-sink-tokens", type=int, default=0)
    parser.add_argument("--match-tile-slots", type=int, default=32)
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--lookback-right", type=int, default=0)
    parser.add_argument("--hq", type=int, default=32)
    parser.add_argument("--hkv", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument(
        "--flashinfer-baseline-timing",
        choices=["run_event", "plan_run_event", "plan_run_wall"],
        default="plan_run_wall",
        help=(
            "Which FlashInfer latency to use for flashinfer_ms and speedup. "
            "Default includes plan+run wall-clock time."
        ),
    )
    parser.add_argument("--partial-fp32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture both MAC and FlashInfer decode in CUDA graphs and time graph replay.",
    )
    parser.add_argument(
        "--mac-fast-math",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the MAC persistent extension with --use_fast_math, matching SGLang production.",
    )
    parser.add_argument("--bench-mode", default="synthetic_group")
    parser.add_argument("--bench-hit-rate-std", type=float, default=0.0)
    parser.add_argument("--bench-skip-ratio", type=float, default=-1.0)
    parser.add_argument("--bench-skip-ratio-std", type=float, default=0.0)
    parser.add_argument("--bench-match-lag-mean", type=float, default=64.0)
    parser.add_argument("--bench-match-lag-std", type=float, default=0.0)
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument(
        "--bench-miss-masks",
        default="",
        help=(
            "Optional comma/space-separated per-GQA-group miss masks, e.g. "
            "'0x0,0x1,0x3,0x5,0x7,0xf'. When set, every group uses the same "
            "mask and hit_rate is derived from popcount(mask). This is a "
            "diagnostic mode for per-mask mixed/full path attribution."
        ),
    )
    parser.add_argument(
        "--synthetic-cache-pattern",
        choices=["faithful", "all_hit", "all_miss"],
        default="faithful",
        help="Cache contents used for synthetic bench modes; faithful mirrors the synthetic hit/miss slots.",
    )
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    batch_values = parse_ints(args.batch)
    warn_degenerate_hit_grid(args, batch_values)
    torch.manual_seed(11)
    ext = load_mac_persistent_decode_extension(False, fast_math=args.mac_fast_math)
    rows: list[dict[str, object]] = []
    forced_masks = parse_masks(args.bench_miss_masks) if args.bench_miss_masks else []
    args.bench_force_miss_mask = -1
    for batch in batch_values:
        for context_len in parse_ints(args.contexts):
            kv_len = context_len + 1
            if kv_len > args.max_context + 1:
                raise ValueError(f"context={context_len} exceeds max_context={args.max_context}")
            baseline_case = _make_case(
                batch=batch,
                kv_len=kv_len,
                hq=args.hq,
                hkv=args.hkv,
                dim=args.dim,
                capacity=args.capacity,
                hit_pattern="all_hit",
            )
            fi = bench_flashinfer(
                baseline_case,
                hq=args.hq,
                hkv=args.hkv,
                dim=args.dim,
                warmup=args.warmup,
                iters=args.iters,
                use_graph=args.cuda_graph,
            )
            fi_ms = {
                "run_event": fi["flashinfer_run_event_ms"],
                "plan_run_event": fi["flashinfer_plan_run_event_ms"],
                "plan_run_wall": fi["flashinfer_plan_run_wall_ms"],
            }[args.flashinfer_baseline_timing]
            hit_specs: list[tuple[float, int]] = []
            if forced_masks:
                group = max(1, args.hq // max(1, args.hkv))
                active_bits = (1 << min(group, 30)) - 1
                for mask in forced_masks:
                    lane_mask = mask & active_bits
                    miss_lanes = int(lane_mask.bit_count())
                    hit_specs.append(((group - miss_lanes) / float(group), lane_mask))
            else:
                hit_specs = [(hit_rate, -1) for hit_rate in parse_floats(args.hit_rates)]
            for hit_rate, forced_mask in hit_specs:
                args.bench_force_miss_mask = forced_mask
                row = bench_mac(ext, args, batch, kv_len, hit_rate, args.tile_tokens)
                row.update(fi)
                row["flashinfer_ms"] = fi_ms
                row["bench_miss_mask"] = "" if forced_mask < 0 else f"0x{forced_mask:x}"
                row["mac_speedup_vs_flashinfer"] = fi_ms / float(row["mac_ms"]) if float(row["mac_ms"]) else ""
                rows.append(row)
                print(
                    f"batch={batch} context={context_len} hit={hit_rate:g} "
                    f"miss_mask={row['bench_miss_mask']} "
                    f"flashinfer_ms={fi_ms:.4f} mac_ms={float(row['mac_ms']):.4f} "
                    f"speedup={float(row['mac_speedup_vs_flashinfer']):.3f} "
                    f"fi_run_event={fi['flashinfer_run_event_ms']:.4f} "
                    f"fi_plan_run_event={fi['flashinfer_plan_run_event_ms']:.4f} "
                    f"fi_plan_run_wall={fi['flashinfer_plan_run_wall_ms']:.4f} "
                    f"mac_event={float(row['mac_event_ms']):.4f} "
                    f"mac_wall={float(row['mac_wall_ms']):.4f}",
                    flush=True,
                )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "batch",
        "context_len",
        "kv_len",
        "hit_rate",
        "bench_miss_mask",
        "synthetic_cache_pattern",
        "flashinfer_ms",
        "flashinfer_run_event_ms",
        "flashinfer_plan_run_event_ms",
        "flashinfer_plan_run_wall_ms",
        "mac_ms",
        "mac_event_ms",
        "mac_wall_ms",
        "mac_speedup_vs_flashinfer",
        "tile_tokens",
        "complete_rows",
        "group_tail_rows",
        "hit_tail_rows",
        "fallback_reduce_groups",
        "balanced_tile_tokens",
        "non_hit_groups",
        "full_fallback_groups",
        "max_full_fallback_reduce_chunks",
        "max_mixed_reduce_chunks",
        "miss_head_count",
        "miss_count_0_groups",
        "miss_count_1_groups",
        "miss_count_2_groups",
        "miss_count_3_groups",
        "miss_count_4_groups",
        "mixed_miss1_groups",
        "mixed_miss2_groups",
        "mixed_miss3_groups",
        "full_miss_groups",
        "scheduled_head_slot_tokens",
        "actual_head_tokens",
        "active_miss_head_tokens",
        "mixed_scheduled_head_slot_tokens",
        "mixed_actual_head_tokens",
        "mixed_active_miss_head_tokens",
        "mixed_unused_head_slot_ratio",
        "complete_group_rows",
        "complete_head_rows",
        "complete_mixed_z2_rows",
        "complete_full_group_rows",
        "complete_mixed_group_rows",
        "complete_mac_hit_group_rows",
        "estimated_materialized_partial_rect_states",
        "estimated_active_partial_rect_states",
        "estimated_inactive_partial_rect_states",
        "estimated_inactive_partial_rect_state_ratio",
        "estimated_partial_rect_states",
        "estimated_partial_rect_bytes",
        "estimated_phase4_rect_state_reads",
        "estimated_phase4_rect_read_bytes",
        "estimated_active_phase4_rect_state_reads",
        "estimated_inactive_phase4_rect_state_reads",
        "estimated_inactive_phase4_rect_read_ratio",
        "miss_mask_hist",
        "group_task_miss_mask_hist",
        "phase_order",
        "match_pct",
        "prepare_pct",
        "complete_pct",
        "reduce_pct_detail",
        "reduce_sync_wait_pct",
        "reduce_compute_pct",
        "tail_pct_detail",
        "fallback_group_merge_pct",
        "merge_write_pct",
        "cache_subtract_pct",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


if __name__ == "__main__":
    main()
