#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch

try:
    import flashinfer
except ImportError as exc:
    raise RuntimeError(
        "flashinfer is required for the FlashInfer baseline in this benchmark. "
        "Install flashinfer before running bench_mac_kernel_latency_2x2.py."
    ) from exc

from mac_attention import MACDecodeWithPagedKVCacheWrapper
from bench_mac_match import bench as bench_host_us
from bench_mac_match import call_mac_ring_match, load_macMatch_extension

PANEL_A_GQA: List[Tuple[str, int, int]] = [
    ("GQA 8-2", 8, 2),
    ("GQA 32-8", 32, 8),
    ("GQA 40-10", 40, 10),
]
PANEL_B_GQA = ("GQA 32-8", 32, 8)
BREAKDOWN_GQA: List[Tuple[str, int, int]] = [
    ("GQA 32-8", 32, 8),
    ("GQA 64-8", 64, 8),
]


def measure_cuda_time_ms(func_to_measure, num_runs: int = 100, num_warmup_runs: int = 10) -> float:
    for _ in range(num_warmup_runs):
        func_to_measure()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        func_to_measure()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(num_runs)


def _parse_int_list(xs: Iterable[int]) -> List[int]:
    return [int(x) for x in xs]


def _parse_float_list(xs: Iterable[float]) -> List[float]:
    return [float(x) for x in xs]


def _attn_start_pos_val(kv_len: int, kv_access: float) -> int:
    window = int(float(kv_access) * float(kv_len - 1))
    return int(kv_len - 1 - window)


def _skip_ratio_to_access(skip_ratio: float) -> float:
    return float(1.0 - float(skip_ratio))


def _cuda_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(int(seed))
    return gen


def _randn_bf16(shape: Sequence[int], *, seed: int, device: torch.device) -> torch.Tensor:
    return torch.randn(
        tuple(shape),
        generator=_cuda_generator(seed),
        device=device,
        dtype=torch.bfloat16,
    )


def _normal_clamped_ratios(
    *,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    mean: float,
    std: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    q_per_kv = num_qo_heads // num_kv_heads
    ratios = torch.normal(
        mean=float(mean),
        std=float(std),
        size=(int(batch_size), int(num_kv_heads), int(q_per_kv)),
        generator=_cuda_generator(seed),
        device=device,
        dtype=torch.float32,
    )
    return ratios.clamp_(0.0, 1.0).reshape(int(batch_size), int(num_qo_heads))


def _ratios_to_attn_start_pos(kv_len: int, ratios: torch.Tensor) -> torch.Tensor:
    if ratios.dtype != torch.float32:
        ratios = ratios.to(torch.float32)
    window = (ratios * float(kv_len - 1)).to(torch.int32)
    return (int(kv_len - 1) - window).contiguous()


def _build_panel_a_rows(
    *,
    mac_match_ext,
    device: torch.device,
    head_dim: int,
    lengths: Sequence[int],
    num_runs: int,
    warmup_runs: int,
    match_threshold: float,
    match_offset: int,
    workspace_bytes: int,
    seed: int,
) -> List[dict]:
    rows: List[dict] = []
    max_length = max(int(x) for x in lengths)

    for gqa_index, (gqa_label, num_qo_heads, num_kv_heads) in enumerate(PANEL_A_GQA):
        baseline_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        baseline_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            baseline_workspace,
            kv_layout="NHD",
            use_tensor_cores=False,
        )
        k_cache = torch.zeros(
            (max_length, 1, num_kv_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        v_cache = torch.zeros_like(k_cache)
        last_page_len = torch.ones((1,), device=device, dtype=torch.int32)
        req_ids = torch.zeros((1,), device=device, dtype=torch.int32)

        for length_index, length in enumerate(lengths):
            local_seed = int(seed) + 1000 * gqa_index + 17 * length_index
            query_cache = _randn_bf16((1, length, num_qo_heads, head_dim), seed=local_seed, device=device)
            queries = _randn_bf16((1, num_qo_heads, head_dim), seed=local_seed + 1, device=device)
            request_length = torch.full((1,), int(length), device=device, dtype=torch.int32)
            hit = torch.empty((1, num_qo_heads), device=device, dtype=torch.bool)
            left = torch.empty((1, num_qo_heads), device=device, dtype=torch.int32)
            idx = torch.empty((1, num_qo_heads), device=device, dtype=torch.int32)

            match_plan = mac_match_ext.mac_ring_match_schedule(
                int(1),
                int(num_qo_heads),
                int(length),
                int(head_dim),
                int(0),
                float(0.95),
                bool(False),
            )

            def mac_match_run():
                return call_mac_ring_match(
                    mac_match_ext,
                    q_cache=query_cache,
                    request_length=request_length,
                    queries=queries,
                    req_ids=req_ids,
                    threshold=float(match_threshold),
                    rows_per_stage=int(match_plan["rows_per_stage"]),
                    load_warps=int(match_plan["load_warps"]),
                    my_offset=int(match_offset),
                    hit=hit,
                    left=left,
                    idx=idx,
                )

            q = _randn_bf16((1, num_qo_heads, head_dim), seed=local_seed + 2, device=device)
            indptr = torch.tensor([0, int(length)], device=device, dtype=torch.int32)
            page_indices = torch.arange(int(length), device=device, dtype=torch.int32)
            baseline_wrapper.plan(
                indptr,
                page_indices,
                last_page_len,
                int(num_qo_heads),
                int(num_kv_heads),
                int(head_dim),
                int(1),
                pos_encoding_mode="NONE",
                q_data_type=torch.bfloat16,
                data_type=torch.bfloat16,
            )

            def flashinfer_decode_run():
                return baseline_wrapper.run_return_lse(q, (k_cache, v_cache))

            mac_match_us = bench_host_us(mac_match_run, num_runs, warmup_runs)
            flashinfer_decode_us = bench_host_us(flashinfer_decode_run, num_runs, warmup_runs)

            rows.append(
                {
                    "gqa_label": gqa_label,
                    "num_qo_heads": int(num_qo_heads),
                    "num_kv_heads": int(num_kv_heads),
                    "head_dim": int(head_dim),
                    "batch_size": 1,
                    "length": int(length),
                    "mac_match_us": round(mac_match_us, 2),
                    "flashinfer_decode_us": round(flashinfer_decode_us, 2),
                }
            )
            print(
                f"[panel a] {gqa_label:>9s} len={length:4d} | "
                f"match={mac_match_us:8.2f} us flashinfer={flashinfer_decode_us:8.2f} us"
            )

    return rows


def _build_breakdown_rows(
    *,
    mac_match_ext,
    device: torch.device,
    head_dim: int,
    batch_size: int,
    context_lengths: Sequence[int],
    skip_ratios: Sequence[float],
    cache_capacity: int,
    num_runs: int,
    warmup_runs: int,
    hit_rate: float,
    workspace_bytes: int,
    match_threshold: float,
    match_offset: int,
    seed: int,
) -> List[dict]:
    rows: List[dict] = []
    max_context_length = max(int(x) for x in context_lengths)

    for gqa_index, (gqa_label, num_qo_heads, num_kv_heads) in enumerate(BREAKDOWN_GQA):
        num_blocks = int(batch_size * max_context_length)
        k_cache = torch.zeros(
            (num_blocks, 1, num_kv_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        v_cache = torch.zeros_like(k_cache)
        attn_cache = torch.zeros(
            (batch_size, cache_capacity, num_qo_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        lse_cache = torch.full(
            (batch_size, cache_capacity, num_qo_heads),
            -float("inf"),
            device=device,
            dtype=torch.float32,
        )
        match_query_cache = _randn_bf16(
            (batch_size, cache_capacity, num_qo_heads, head_dim),
            seed=int(seed) + 500 * gqa_index + 11,
            device=device,
        )
        match_queries = _randn_bf16(
            (batch_size, num_qo_heads, head_dim),
            seed=int(seed) + 500 * gqa_index + 13,
            device=device,
        )
        match_request_length = torch.full((batch_size,), int(cache_capacity), device=device, dtype=torch.int32)
        match_req_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        match_hit = torch.empty((batch_size, num_qo_heads), device=device, dtype=torch.bool)
        match_left = torch.empty((batch_size, num_qo_heads), device=device, dtype=torch.int32)
        match_idx = torch.empty((batch_size, num_qo_heads), device=device, dtype=torch.int32)
        match_plan = mac_match_ext.mac_ring_match_schedule(
            int(batch_size),
            int(num_qo_heads),
            int(cache_capacity),
            int(head_dim),
            int(0),
            float(0.95),
            bool(False),
        )

        def mac_match_run():
            return call_mac_ring_match(
                mac_match_ext,
                q_cache=match_query_cache,
                request_length=match_request_length,
                queries=match_queries,
                req_ids=match_req_ids,
                threshold=float(match_threshold),
                rows_per_stage=int(match_plan["rows_per_stage"]),
                load_warps=int(match_plan["load_warps"]),
                my_offset=int(match_offset),
                hit=match_hit,
                left=match_left,
                idx=match_idx,
            )

        mac_match_us = bench_host_us(mac_match_run, num_runs, warmup_runs)

        mac_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        baseline_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        mac_wrapper = MACDecodeWithPagedKVCacheWrapper(mac_workspace, kv_layout="NHD")
        baseline_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            baseline_workspace,
            kv_layout="NHD",
            use_tensor_cores=False,
        )
        attn_host_pinned = torch.empty(65536, dtype=torch.int32, device="cpu", pin_memory=True)
        req_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        indices = torch.randint(
            0,
            cache_capacity,
            (batch_size, num_qo_heads),
            generator=_cuda_generator(int(seed) + 500 * gqa_index + 17),
            device=device,
            dtype=torch.int32,
        )
        hit_table = (
            torch.rand(
                (batch_size, num_qo_heads),
                generator=_cuda_generator(int(seed) + 500 * gqa_index + 19),
                device=device,
            )
            < float(hit_rate)
        )
        last_page_len = torch.ones((batch_size,), device=device, dtype=torch.int32)

        print(
            f"[breakdown] {gqa_label:>9s} | "
            f"match={mac_match_us:8.2f} us"
        )

        for context_index, context_length in enumerate(context_lengths):
            q = _randn_bf16(
                (batch_size, num_qo_heads, head_dim),
                seed=int(seed) + 500 * gqa_index + 31 + context_index,
                device=device,
            )
            indptr = torch.arange(
                0,
                (batch_size + 1) * int(context_length),
                step=int(context_length),
                device=device,
                dtype=torch.int32,
            )
            page_indices = torch.arange(
                0,
                batch_size * int(context_length),
                device=device,
                dtype=torch.int32,
            )
            baseline_wrapper.plan(
                indptr,
                page_indices,
                last_page_len,
                int(num_qo_heads),
                int(num_kv_heads),
                int(head_dim),
                int(1),
                pos_encoding_mode="NONE",
                q_data_type=torch.bfloat16,
                data_type=torch.bfloat16,
            )

            def flashinfer_baseline_run():
                return baseline_wrapper.run_return_lse(q, (k_cache, v_cache))

            flashinfer_baseline_time_us = (
                measure_cuda_time_ms(
                    flashinfer_baseline_run,
                    num_runs=num_runs,
                    num_warmup_runs=warmup_runs,
                )
                * 1000.0
            )

            for skip_ratio in skip_ratios:
                kv_access = _skip_ratio_to_access(skip_ratio)
                attn_start_pos = torch.full(
                    (batch_size, num_qo_heads),
                    _attn_start_pos_val(int(context_length), kv_access),
                    device=device,
                    dtype=torch.int32,
                )

                def mac_plan_run():
                    mac_wrapper.plan(
                        indptr,
                        page_indices,
                        last_page_len,
                        int(num_qo_heads),
                        int(num_kv_heads),
                        int(head_dim),
                        int(1),
                        pos_encoding_mode="NONE",
                        q_data_type=torch.bfloat16,
                        data_type=torch.bfloat16,
                        attn_start_pos=attn_start_pos,
                        downdate_range=int(match_offset),
                        attn_start_pos_host_pinned_opt=attn_host_pinned,
                    )

                mac_plan_run()

                def mac_attention_run():
                    return mac_wrapper.forward_return_lse(
                        q,
                        (k_cache, v_cache),
                        int(batch_size),
                        int(cache_capacity),
                        attn_cache,
                        lse_cache,
                        indices,
                        hit_table,
                        req_ids,
                        True,
                        attn_start_pos,
                        int(match_offset),
                    )

                mac_plan_time_us = (
                    measure_cuda_time_ms(
                        mac_plan_run,
                        num_runs=num_runs,
                        num_warmup_runs=warmup_runs,
                    )
                    * 1000.0
                )
                mac_attn_time_us = (
                    measure_cuda_time_ms(
                        mac_attention_run,
                        num_runs=num_runs,
                        num_warmup_runs=warmup_runs,
                    )
                    * 1000.0
                )
                mac_total_time_us = mac_match_us + mac_plan_time_us + mac_attn_time_us

                rows.append(
                    {
                        "gqa_label": gqa_label,
                        "num_qo_heads": int(num_qo_heads),
                        "num_kv_heads": int(num_kv_heads),
                        "head_dim": int(head_dim),
                        "batch_size": int(batch_size),
                        "context_length": int(context_length),
                        "skip_ratio": round(float(skip_ratio), 2),
                        "KV_Access": round(float(kv_access), 2),
                        "mac_match_us": round(mac_match_us, 2),
                        "mac_plan_time_us": round(mac_plan_time_us, 2),
                        "mac_attn_time_us": round(mac_attn_time_us, 2),
                        "mac_total_time_us": round(mac_total_time_us, 2),
                        "flashinfer_baseline_time_us": round(flashinfer_baseline_time_us, 2),
                    }
                )
                print(
                    f"  ctx={context_length:6d} skip={skip_ratio:>4.2f} | "
                    f"plan={mac_plan_time_us:8.2f} us attn={mac_attn_time_us:8.2f} us "
                    f"total={mac_total_time_us:8.2f} us baseline={flashinfer_baseline_time_us:8.2f} us"
                )

    return rows


def _build_panel_b_rows(
    *,
    device: torch.device,
    head_dim: int,
    batch_size: int,
    context_lengths: Sequence[int],
    sigmas: Sequence[float],
    cache_capacity: int,
    num_runs: int,
    warmup_runs: int,
    hit_rate: float,
    workspace_bytes: int,
    match_offset: int,
    seed: int,
) -> List[dict]:
    rows: List[dict] = []
    _gqa_label, num_qo_heads, num_kv_heads = PANEL_B_GQA
    max_context_length = max(int(x) for x in context_lengths)
    num_blocks = int(batch_size * max_context_length)

    k_cache = torch.zeros(
        (num_blocks, 1, num_kv_heads, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.zeros_like(k_cache)
    attn_cache = torch.zeros(
        (batch_size, cache_capacity, num_qo_heads, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    lse_cache = torch.full(
        (batch_size, cache_capacity, num_qo_heads),
        -float("inf"),
        device=device,
        dtype=torch.float32,
    )
    mac_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    mac_wrapper = MACDecodeWithPagedKVCacheWrapper(mac_workspace, kv_layout="NHD")
    attn_host_pinned = torch.empty(65536, dtype=torch.int32, device="cpu", pin_memory=True)
    req_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
    indices = torch.randint(
        0,
        cache_capacity,
        (batch_size, num_qo_heads),
        generator=_cuda_generator(int(seed) + 7001),
        device=device,
        dtype=torch.int32,
    )
    hit_table = (
        torch.rand(
            (batch_size, num_qo_heads),
            generator=_cuda_generator(int(seed) + 7003),
            device=device,
        )
        < float(hit_rate)
    )
    last_page_len = torch.ones((batch_size,), device=device, dtype=torch.int32)

    def measure_mode(
        *,
        q: torch.Tensor,
        indptr: torch.Tensor,
        page_indices: torch.Tensor,
        attn_start_pos: torch.Tensor,
    ) -> Tuple[float, float, float]:
        def mac_plan_run():
            mac_wrapper.plan(
                indptr,
                page_indices,
                last_page_len,
                int(num_qo_heads),
                int(num_kv_heads),
                int(head_dim),
                int(1),
                pos_encoding_mode="NONE",
                q_data_type=torch.bfloat16,
                data_type=torch.bfloat16,
                attn_start_pos=attn_start_pos,
                downdate_range=int(match_offset),
                attn_start_pos_host_pinned_opt=attn_host_pinned,
            )

        mac_plan_run()

        def mac_attention_run():
            return mac_wrapper.forward_return_lse(
                q,
                (k_cache, v_cache),
                int(batch_size),
                int(cache_capacity),
                attn_cache,
                lse_cache,
                indices,
                hit_table,
                req_ids,
                True,
                attn_start_pos,
                int(match_offset),
            )

        plan_us = (
            measure_cuda_time_ms(
                mac_plan_run,
                num_runs=num_runs,
                num_warmup_runs=warmup_runs,
            )
            * 1000.0
        )
        attn_us = (
            measure_cuda_time_ms(
                mac_attention_run,
                num_runs=num_runs,
                num_warmup_runs=warmup_runs,
            )
            * 1000.0
        )
        return plan_us, attn_us, plan_us + attn_us

    for context_index, context_length in enumerate(context_lengths):
        q = _randn_bf16(
            (batch_size, num_qo_heads, head_dim),
            seed=int(seed) + 7100 + context_index,
            device=device,
        )
        indptr = torch.arange(
            0,
            (batch_size + 1) * int(context_length),
            step=int(context_length),
            device=device,
            dtype=torch.int32,
        )
        page_indices = torch.arange(
            0,
            batch_size * int(context_length),
            device=device,
            dtype=torch.int32,
        )

        for sigma_index, sigma in enumerate(sigmas):
            perfect_ratio = torch.full(
                (batch_size, num_qo_heads),
                float(sigma),
                device=device,
                dtype=torch.float32,
            )
            lb_ratio = _normal_clamped_ratios(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                mean=float(sigma),
                std=float(sigma),
                seed=int(seed) + 7200 + context_index * 17 + sigma_index,
                device=device,
            )
            no_lb_value = min(1.0, float(sigma) + float(sigma))
            no_lb_ratio = torch.full(
                (batch_size, num_qo_heads),
                float(no_lb_value),
                device=device,
                dtype=torch.float32,
            )

            attn_start_pos_perfect = _ratios_to_attn_start_pos(int(context_length), perfect_ratio)
            attn_start_pos_lb = _ratios_to_attn_start_pos(int(context_length), lb_ratio)
            attn_start_pos_no_lb = _ratios_to_attn_start_pos(int(context_length), no_lb_ratio)

            perfect_plan_us, perfect_attn_us, perfect_total_us = measure_mode(
                q=q,
                indptr=indptr,
                page_indices=page_indices,
                attn_start_pos=attn_start_pos_perfect,
            )
            lb_plan_us, lb_attn_us, lb_total_us = measure_mode(
                q=q,
                indptr=indptr,
                page_indices=page_indices,
                attn_start_pos=attn_start_pos_lb,
            )
            no_lb_plan_us, no_lb_attn_us, no_lb_total_us = measure_mode(
                q=q,
                indptr=indptr,
                page_indices=page_indices,
                attn_start_pos=attn_start_pos_no_lb,
            )

            rows.append(
                {
                    "gqa_label": PANEL_B_GQA[0],
                    "num_qo_heads": int(num_qo_heads),
                    "num_kv_heads": int(num_kv_heads),
                    "head_dim": int(head_dim),
                    "batch_size": int(batch_size),
                    "context_length": int(context_length),
                    "sigma": round(float(sigma), 2),
                    "mac_perfect_kv_access": round(float(sigma), 2),
                    "mac_lb_mean_kv_access": round(float(sigma), 2),
                    "mac_lb_std_kv_access": round(float(sigma), 2),
                    "mac_no_lb_kv_access": round(float(no_lb_value), 2),
                    "mac_perfect_plan_us": round(perfect_plan_us, 2),
                    "mac_perfect_attn_us": round(perfect_attn_us, 2),
                    "mac_perfect_total_us": round(perfect_total_us, 2),
                    "mac_lb_plan_us": round(lb_plan_us, 2),
                    "mac_lb_attn_us": round(lb_attn_us, 2),
                    "mac_lb_total_us": round(lb_total_us, 2),
                    "mac_no_lb_plan_us": round(no_lb_plan_us, 2),
                    "mac_no_lb_attn_us": round(no_lb_attn_us, 2),
                    "mac_no_lb_total_us": round(no_lb_total_us, 2),
                }
            )
            print(
                f"[panel b] ctx={context_length:6d} sigma={sigma:>3.1f} | "
                f"perfect={perfect_total_us:8.2f} us "
                f"w.LB={lb_total_us:8.2f} us "
                f"w.o.LB={no_lb_total_us:8.2f} us"
            )

    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark the data-driven MAC kernel-latency 2x2 figure: "
            "panel (a) match vs FlashInfer decode, and panels (c)/(d) "
            "MAC component breakdowns against full attention."
        )
    )
    ap.add_argument("--panel-a-lengths", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[32768, 65536, 131072])
    ap.add_argument("--skip-ratios", type=float, nargs="+", default=[0.99, 0.90, 0.80])
    ap.add_argument("--breakdown-batch-size", type=int, default=4)
    ap.add_argument("--panel-b-context-lengths", type=int, nargs="+", default=[32768, 65536, 131072])
    ap.add_argument("--panel-b-sigmas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--panel-b-batch-size", type=int, default=1)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--cache-capacity", type=int, default=512)
    ap.add_argument("--workspace-mb", type=int, default=512)
    ap.add_argument("--num-runs", type=int, default=100)
    ap.add_argument("--warmup-runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--hit-rate", type=float, default=0.7)
    ap.add_argument("--match-threshold", type=float, default=0.95)
    ap.add_argument("--match-offset", type=int, default=256)
    ap.add_argument(
        "--panel-a-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "bench_mac_kernel_latency_panel_a.csv"),
    )
    ap.add_argument(
        "--panel-b-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "bench_mac_kernel_latency_panel_b.csv"),
    )
    ap.add_argument(
        "--breakdown-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "bench_mac_kernel_latency_breakdown.csv"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    mac_match_ext = load_macMatch_extension(verbose=False)
    workspace_bytes = int(args.workspace_mb) * 1024 * 1024

    panel_a_rows = _build_panel_a_rows(
        mac_match_ext=mac_match_ext,
        device=device,
        head_dim=int(args.head_dim),
        lengths=_parse_int_list(args.panel_a_lengths),
        num_runs=int(args.num_runs),
        warmup_runs=int(args.warmup_runs),
        match_threshold=float(args.match_threshold),
        match_offset=int(args.match_offset),
        workspace_bytes=workspace_bytes,
        seed=int(args.seed),
    )
    panel_a_fieldnames = [
        "gqa_label",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "batch_size",
        "length",
        "mac_match_us",
        "flashinfer_decode_us",
    ]
    panel_a_csv = Path(args.panel_a_csv)
    _write_csv(panel_a_csv, panel_a_fieldnames, panel_a_rows)

    panel_b_rows = _build_panel_b_rows(
        device=device,
        head_dim=int(args.head_dim),
        batch_size=int(args.panel_b_batch_size),
        context_lengths=_parse_int_list(args.panel_b_context_lengths),
        sigmas=_parse_float_list(args.panel_b_sigmas),
        cache_capacity=int(args.cache_capacity),
        num_runs=int(args.num_runs),
        warmup_runs=int(args.warmup_runs),
        hit_rate=float(args.hit_rate),
        workspace_bytes=workspace_bytes,
        match_offset=int(args.match_offset),
        seed=int(args.seed),
    )
    panel_b_fieldnames = [
        "gqa_label",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "batch_size",
        "context_length",
        "sigma",
        "mac_perfect_kv_access",
        "mac_lb_mean_kv_access",
        "mac_lb_std_kv_access",
        "mac_no_lb_kv_access",
        "mac_perfect_plan_us",
        "mac_perfect_attn_us",
        "mac_perfect_total_us",
        "mac_lb_plan_us",
        "mac_lb_attn_us",
        "mac_lb_total_us",
        "mac_no_lb_plan_us",
        "mac_no_lb_attn_us",
        "mac_no_lb_total_us",
    ]
    panel_b_csv = Path(args.panel_b_csv)
    _write_csv(panel_b_csv, panel_b_fieldnames, panel_b_rows)

    breakdown_rows = _build_breakdown_rows(
        mac_match_ext=mac_match_ext,
        device=device,
        head_dim=int(args.head_dim),
        batch_size=int(args.breakdown_batch_size),
        context_lengths=_parse_int_list(args.context_lengths),
        skip_ratios=_parse_float_list(args.skip_ratios),
        cache_capacity=int(args.cache_capacity),
        num_runs=int(args.num_runs),
        warmup_runs=int(args.warmup_runs),
        hit_rate=float(args.hit_rate),
        workspace_bytes=workspace_bytes,
        match_threshold=float(args.match_threshold),
        match_offset=int(args.match_offset),
        seed=int(args.seed),
    )
    breakdown_fieldnames = [
        "gqa_label",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "batch_size",
        "context_length",
        "skip_ratio",
        "KV_Access",
        "mac_match_us",
        "mac_plan_time_us",
        "mac_attn_time_us",
        "mac_total_time_us",
        "flashinfer_baseline_time_us",
    ]
    breakdown_csv = Path(args.breakdown_csv)
    _write_csv(breakdown_csv, breakdown_fieldnames, breakdown_rows)

    print(f"Wrote: {panel_a_csv}")
    print(f"Wrote: {panel_b_csv}")
    print(f"Wrote: {breakdown_csv}")


if __name__ == "__main__":
    main()
