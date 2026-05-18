#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time

import torch

from mac_attention.integrations.sglang.bridge import load_mac_persistent_decode_extension
from tools.check_mac_persistent_decode import _workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--past-len", type=int, default=65535)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--semantic-pos-ahead", type=int, default=256)
    parser.add_argument("--tile-tokens", type=int, default=32)
    parser.add_argument("--match-tile-slots", type=int, default=32)
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--debug", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, hkv, group, dim = 1, 8, 4, 128
    hq = hkv * group
    kv_len = args.past_len + 1

    ext = load_mac_persistent_decode_extension(
        verbose=False,
        fast_math=os.environ.get("MAC_PERSISTENT_FAST_MATH", "1").lower()
        not in {"0", "false", "no", "off"},
    )

    q_pre = torch.randn(batch, hq, dim, device=device, dtype=dtype)
    q_post = torch.randn(batch, hq, dim, device=device, dtype=dtype)
    k = torch.randn(kv_len, hkv, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    req_to_token = torch.arange(args.past_len, device=device, dtype=torch.int32).reshape(1, -1)
    req_ids = torch.zeros(batch, device=device, dtype=torch.int32)
    past_lens = torch.full((batch,), args.past_len, device=device, dtype=torch.int32)
    out_cache_loc = torch.tensor([args.past_len], device=device, dtype=torch.int64)

    query_cache = torch.randn(batch, args.capacity, hq, dim, device=device, dtype=dtype) + 10
    attn_cache = torch.randn(batch, args.capacity, hq, dim, device=device, dtype=dtype)
    lse_cache = torch.full((batch, args.capacity, hq), -float("inf"), device=device)

    match_pos = args.past_len - 1
    match_slot = match_pos % args.capacity
    query_cache[0, match_slot].copy_(q_pre[0])
    lse_cache[0, match_slot].fill_(8.0)

    out = torch.empty_like(q_post)
    out_lse = torch.empty(batch, hq, device=device)
    workspace = _workspace(
        batch,
        hq,
        hkv,
        group,
        dim,
        args.capacity,
        args.max_context,
        args.tile_tokens,
        args.match_tile_slots,
        args.semantic_pos_ahead,
    )

    def run_once() -> None:
        ext.mac_persistent_decode_bf16(
            q_post,
            q_pre,
            k,
            v,
            req_to_token,
            req_ids,
            past_lens,
            out_cache_loc,
            query_cache,
            attn_cache,
            lse_cache,
            out,
            out_lse,
            *workspace,
            args.max_context,
            args.tile_tokens,
            args.match_tile_slots,
            args.semantic_pos_ahead,
            2048,
            0,
            0,
            args.threshold,
            1 / math.sqrt(dim),
            args.debug,
            0,
            1.0,
            0.0,
            -1.0,
            0.0,
            64.0,
            0.0,
            1,
            -1,
            0,
        )

    for _ in range(args.warmups):
        run_once()
    torch.cuda.synchronize()

    tic = time.perf_counter()
    for _ in range(args.iters):
        run_once()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - tic) * 1000.0 / max(args.iters, 1)
    print(
        "profile_mac_persistent_decode "
        f"past_len={args.past_len} capacity={args.capacity} "
        f"semantic_pos_ahead={args.semantic_pos_ahead} avg_ms={elapsed_ms:.4f}",
        flush=True,
    )
    if args.debug >= 2:
        cycles = [int(x) for x in workspace[-1].detach().cpu().tolist()]
        names = ("match", "reduce", "complete", "tail", "merge")
        deltas = [max(cycles[i + 1] - cycles[i], 0) for i in range(5)]
        total = sum(deltas)
        if total:
            parts = " ".join(
                f"{name}={delta / total * 100.0:.1f}%({delta})"
                for name, delta in zip(names, deltas)
            )
            print(f"profile_mac_persistent_decode_phases total_cycles={total} {parts}", flush=True)


if __name__ == "__main__":
    main()
