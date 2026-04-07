#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

try:
    import flashinfer
except ImportError as exc:
    raise RuntimeError(
        "flashinfer is required for the FlashInfer baseline in this benchmark. "
        "Install flashinfer before running bench_time_grid_mac_attention_speedup.py."
    ) from exc

from mac_attention import MACDecodeWithPagedKVCacheWrapper
from bench_mac_match import (
    bench as bench_host_us,
    call_mac_ring_match,
    load_macMatch_extension,
    measure_schedule_us,
)


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


def _cuda_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(int(seed))
    return gen


def generate_attention_cache(
    *,
    capacity: int,
    num_heads: int,
    head_dim: int,
    max_running_requests: int,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = _cuda_generator(seed)

    attn_cache_fp32 = torch.randn(
        (max_running_requests, capacity, num_heads, head_dim),
        generator=gen,
        device=device,
        dtype=torch.float32,
    ) * 0.5
    attn_cache = attn_cache_fp32.to(torch.bfloat16)
    del attn_cache_fp32

    lse_cache = torch.empty(
        (max_running_requests, capacity, num_heads),
        device=device,
        dtype=torch.float32,
    )
    lse_cache.fill_(-float("inf"))

    valid_mask = torch.rand(
        (max_running_requests, capacity, num_heads),
        generator=gen,
        device=device,
    ) >= 0.05
    seq_len = torch.randint(
        1,
        2049,
        (max_running_requests, capacity, num_heads),
        generator=gen,
        device=device,
        dtype=torch.int32,
    )
    lse_vals = torch.log(seq_len.to(torch.float32)) + 0.5 * torch.randn(
        (max_running_requests, capacity, num_heads),
        generator=gen,
        device=device,
        dtype=torch.float32,
    )
    lse_cache[valid_mask] = lse_vals[valid_mask]
    return attn_cache, lse_cache


def generate_query_cache(
    *,
    capacity: int,
    num_heads: int,
    head_dim: int,
    max_running_requests: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    gen = _cuda_generator(seed)
    return torch.randn(
        (max_running_requests, capacity, num_heads, head_dim),
        generator=gen,
        device=device,
        dtype=torch.bfloat16,
    )


def _global_index_to_local_slot(global_index: int, request_length: int, capacity: int) -> int:
    if request_length <= 0:
        raise ValueError("request_length must be positive")
    if global_index < 0 or global_index >= request_length:
        raise ValueError(
            f"global_index must be in [0, {request_length}), got global_index={global_index}"
        )
    if request_length < capacity:
        return int(global_index)

    base_global = request_length - capacity
    order_index = global_index - base_global
    tail = request_length % capacity
    return int((order_index + tail) % capacity)


def inject_exact_match_targets(
    *,
    query_cache: torch.Tensor,
    queries: torch.Tensor,
    req_ids: torch.Tensor,
    attn_start_pos: torch.Tensor,
    request_length: torch.Tensor,
    match_offset: int,
) -> torch.Tensor:
    batch_size, num_heads, _head_dim = queries.shape
    expected_idx = torch.empty((batch_size, num_heads), device=queries.device, dtype=torch.int32)

    for n in range(batch_size):
        req = int(req_ids[n].item())
        req_len = int(request_length[req].item())
        for h in range(num_heads):
            start_pos = int(attn_start_pos[n, h].item())
            global_index = start_pos + int(match_offset) - 1
            local_slot = _global_index_to_local_slot(global_index, req_len, int(query_cache.size(1)))
            query_cache[req, local_slot, h, :] = queries[n, h, :]
            expected_idx[n, h] = int(local_slot)

    return expected_idx


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark end-to-end MAC critical-path latency "
            "(macMatch kernel + MACDecode plan + MAC attention) versus FlashInfer baseline "
            "and emit the paper-style CSV schema."
        )
    )
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[32768, 65536, 131072, 262144])
    ap.add_argument("--kv-accesses", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--num-qo-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=1)
    ap.add_argument("--max-running-requests", type=int, default=128)
    ap.add_argument("--cache-capacity", type=int, default=512)
    ap.add_argument("--downdate-range", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=2621440)
    ap.add_argument("--workspace-bytes", type=int, default=2048 * 1024 * 1024)
    ap.add_argument("--num-runs", type=int, default=100)
    ap.add_argument("--warmup-runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--match-threshold", type=float, default=0.95)
    ap.add_argument("--match-offset", type=int, default=256)
    ap.add_argument("--match-target-util", type=float, default=0.95)
    ap.add_argument("--match-allow-rpt-32", action="store_true")
    ap.add_argument(
        "--csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "bench_time_grid_results.csv"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if int(args.page_size) != 1:
        raise ValueError("This benchmark currently requires page_size == 1")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_sizes = _parse_int_list(args.batch_sizes)
    context_lengths = _parse_int_list(args.context_lengths)
    kv_accesses = _parse_float_list(args.kv_accesses)

    num_qo_heads = int(args.num_qo_heads)
    num_kv_heads = int(args.num_kv_heads)
    head_dim = int(args.head_dim)
    page_size = int(args.page_size)
    max_running_requests = int(args.max_running_requests)
    cache_capacity = int(args.cache_capacity)
    downdate_range = int(args.downdate_range)
    num_blocks = int(args.num_blocks)
    workspace_bytes = int(args.workspace_bytes)
    num_runs = int(args.num_runs)
    warmup_runs = int(args.warmup_runs)

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    k_cache = torch.empty((num_blocks, page_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.empty_like(k_cache)

    mac_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    baseline_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    mac_match_ext = load_macMatch_extension(verbose=False)

    mac_wrapper = MACDecodeWithPagedKVCacheWrapper(mac_workspace, kv_layout="NHD")
    baseline_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        baseline_workspace,
        kv_layout="NHD",
        use_tensor_cores=False,
    )

    attn_host_pinned = torch.empty(65536, dtype=torch.int32, device="cpu", pin_memory=True)

    for b in batch_sizes:
        if b > max_running_requests:
            raise ValueError(
                f"batch size {b} exceeds max_running_requests={max_running_requests}"
            )

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "context_length",
        "KV_Access",
        "mac_match_rows_per_stage",
        "mac_match_load_warps",
        "mac_match_schedule_us",
        "mac_match_kernel_us",
        "mac_match_us",
        "mac_plan_time_us",
        "mac_attn_time_us",
        "mac_total_time_us",
        "flashinfer_baseline_time_us",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for batch_size in batch_sizes:
            last_page_len = torch.ones((batch_size,), device=device, dtype=torch.int32)
            req_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

            match_plan, mac_match_schedule_us = measure_schedule_us(
                mac_match_ext,
                N=batch_size,
                H=num_qo_heads,
                M=cache_capacity,
                D=head_dim,
                load_warps_override=0,
                target_util=float(args.match_target_util),
                allow_rpt_32=bool(args.match_allow_rpt_32),
                num_runs=num_runs,
                num_warmup_runs=warmup_runs,
            )
            mac_match_rows_per_stage = int(match_plan["rows_per_stage"])
            mac_match_load_warps = int(match_plan["load_warps"])

            for kv_len in context_lengths:
                for kv_access in kv_accesses:
                    q = torch.randn(
                        (batch_size, num_qo_heads, head_dim),
                        device=device,
                        dtype=dtype,
                    )
                    attn_start_pos_target = torch.full(
                        (batch_size, num_qo_heads),
                        _attn_start_pos_val(kv_len, kv_access),
                        device=device,
                        dtype=torch.int32,
                    )
                    if int(attn_start_pos_target.max().item()) > (kv_len - int(args.match_offset)):
                        raise ValueError(
                            f"match_offset={int(args.match_offset)} is too large for kv_len={kv_len} "
                            f"and kv_access={kv_access}; cannot realize attn_start_pos via macMatch."
                        )

                    attn_cache, lse_cache = generate_attention_cache(
                        capacity=cache_capacity,
                        num_heads=num_qo_heads,
                        head_dim=head_dim,
                        max_running_requests=max_running_requests,
                        device=device,
                        seed=int(args.seed),
                    )
                    query_cache = generate_query_cache(
                        capacity=cache_capacity,
                        num_heads=num_qo_heads,
                        head_dim=head_dim,
                        max_running_requests=max_running_requests,
                        device=device,
                        seed=int(args.seed) + 17,
                    )
                    indptr = torch.arange(
                        0,
                        (batch_size + 1) * kv_len,
                        step=kv_len,
                        device=device,
                        dtype=torch.int32,
                    )
                    page_indices = torch.randint(
                        0,
                        num_blocks,
                        (batch_size * kv_len,),
                        device=device,
                        dtype=torch.int32,
                    )
                    request_length = torch.full(
                        (max_running_requests,),
                        kv_len,
                        device=device,
                        dtype=torch.int32,
                    )
                    expected_idx = inject_exact_match_targets(
                        query_cache=query_cache,
                        queries=q,
                        req_ids=req_ids,
                        attn_start_pos=attn_start_pos_target,
                        request_length=request_length,
                        match_offset=int(args.match_offset),
                    )

                    hit = torch.empty((batch_size, num_qo_heads), device=device, dtype=torch.bool)
                    left = torch.empty((batch_size, num_qo_heads), device=device, dtype=torch.int32)
                    idx = torch.empty((batch_size, num_qo_heads), device=device, dtype=torch.int32)

                    baseline_wrapper.plan(
                        indptr,
                        page_indices,
                        last_page_len,
                        num_qo_heads,
                        num_kv_heads,
                        head_dim,
                        page_size,
                        pos_encoding_mode="NONE",
                        q_data_type=dtype,
                        data_type=dtype,
                    )

                    def flashinfer_baseline_run():
                        return baseline_wrapper.run_return_lse(q, (k_cache, v_cache))

                    def mac_match_run():
                        return call_mac_ring_match(
                            mac_match_ext,
                            q_cache=query_cache,
                            request_length=request_length,
                            queries=q,
                            req_ids=req_ids,
                            threshold=float(args.match_threshold),
                            rows_per_stage=mac_match_rows_per_stage,
                            load_warps=mac_match_load_warps,
                            my_offset=int(args.match_offset),
                            hit=hit,
                            left=left,
                            idx=idx,
                        )

                    def mac_plan_run():
                        mac_wrapper.plan(
                            indptr,
                            page_indices,
                            last_page_len,
                            num_qo_heads,
                            num_kv_heads,
                            head_dim,
                            page_size,
                            pos_encoding_mode="NONE",
                            q_data_type=dtype,
                            data_type=dtype,
                            attn_start_pos=left,
                            downdate_range=downdate_range,
                            attn_start_pos_host_pinned_opt=attn_host_pinned,
                        )

                    match_hit, match_left, match_idx = mac_match_run()
                    if not bool(match_hit.all().item()):
                        raise RuntimeError("macMatch expected all hits for the injected benchmark inputs")
                    if not torch.equal(match_left, attn_start_pos_target):
                        raise RuntimeError("macMatch left output does not match the requested attn_start_pos")
                    if not torch.equal(match_idx, expected_idx):
                        raise RuntimeError("macMatch idx output does not match the injected local slots")

                    mac_plan_run()

                    def mac_attention_run():
                        return mac_wrapper.forward_return_lse(
                            q,
                            (k_cache, v_cache),
                            max_running_requests,
                            cache_capacity,
                            attn_cache,
                            lse_cache,
                            idx,
                            hit,
                            req_ids,
                            True,
                            left,
                            downdate_range,
                        )

                    def mac_critical_path_run():
                        _hit, _left, _idx = mac_match_run()
                        mac_wrapper.plan(
                            indptr,
                            page_indices,
                            last_page_len,
                            num_qo_heads,
                            num_kv_heads,
                            head_dim,
                            page_size,
                            pos_encoding_mode="NONE",
                            q_data_type=dtype,
                            data_type=dtype,
                            attn_start_pos=_left,
                            downdate_range=downdate_range,
                            attn_start_pos_host_pinned_opt=attn_host_pinned,
                        )
                        return mac_wrapper.forward_return_lse(
                            q,
                            (k_cache, v_cache),
                            max_running_requests,
                            cache_capacity,
                            attn_cache,
                            lse_cache,
                            _idx,
                            _hit,
                            req_ids,
                            True,
                            _left,
                            downdate_range,
                        )

                    flashinfer_baseline_time_us = measure_cuda_time_ms(
                        flashinfer_baseline_run,
                        num_runs=num_runs,
                        num_warmup_runs=warmup_runs,
                    ) * 1000.0
                    mac_match_kernel_us = bench_host_us(
                        mac_match_run,
                        num_runs,
                        warmup_runs,
                    )
                    mac_plan_time_us = measure_cuda_time_ms(
                        mac_plan_run,
                        num_runs=num_runs,
                        num_warmup_runs=warmup_runs,
                    ) * 1000.0
                    mac_attn_time_us = measure_cuda_time_ms(
                        mac_attention_run,
                        num_runs=num_runs,
                        num_warmup_runs=warmup_runs,
                    ) * 1000.0
                    mac_match_us = float(mac_match_kernel_us)
                    mac_total_time_us = bench_host_us(
                        mac_critical_path_run,
                        num_runs,
                        warmup_runs,
                    )

                    writer.writerow(
                        {
                            "batch_size": batch_size,
                            "context_length": kv_len,
                            "KV_Access": kv_access,
                            "mac_match_rows_per_stage": mac_match_rows_per_stage,
                            "mac_match_load_warps": mac_match_load_warps,
                            "mac_match_schedule_us": round(mac_match_schedule_us, 2),
                            "mac_match_kernel_us": round(mac_match_kernel_us, 2),
                            "mac_match_us": round(mac_match_us, 2),
                            "mac_plan_time_us": round(mac_plan_time_us, 2),
                            "mac_attn_time_us": round(mac_attn_time_us, 2),
                            "mac_total_time_us": round(mac_total_time_us, 2),
                            "flashinfer_baseline_time_us": round(flashinfer_baseline_time_us, 2),
                        }
                    )
                    f.flush()
                    print(
                        f"B={batch_size:2d} kv={kv_len:6d} access={kv_access:>4.2f} | "
                        f"match={mac_match_us:8.2f} us plan={mac_plan_time_us:8.2f} us "
                        f"attn={mac_attn_time_us:8.2f} us total={mac_total_time_us:8.2f} us "
                        f"baseline={flashinfer_baseline_time_us:8.2f} us"
                    )

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
