import argparse
import csv
import math
from pathlib import Path

import torch

from mac_attention.integrations.sglang.bridge import load_mac_persistent_decode_extension


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _workspace(
    batch: int,
    hq: int,
    hkv: int,
    group: int,
    dim: int,
    capacity: int,
    max_context: int,
    tile_tokens: int,
    match_tile_slots: int,
    semantic_pos_ahead: int,
    partial_o_dtype: torch.dtype,
):
    max_match_tiles = _ceil_div(capacity, match_tile_slots)
    max_tiles_context = _ceil_div(max_context, tile_tokens)
    max_tiles_reduce = _ceil_div(max_context, tile_tokens * 32)
    max_tiles_tail = max(1, _ceil_div(max(semantic_pos_ahead, 1), tile_tokens))
    max_complete_tasks = batch * hq * max_tiles_context
    max_tail_tasks = batch * hkv * max_tiles_tail
    max_hit_tail_tasks = batch * hq * max_tiles_tail
    device = torch.device("cuda")
    return [
        torch.empty(batch, hq, max_match_tiles, device=device, dtype=torch.float32),
        torch.empty(batch, hq, max_match_tiles, device=device, dtype=torch.int32),
        torch.empty(batch, hq, max_match_tiles, device=device, dtype=torch.int32),
        *[torch.empty(batch, hq, device=device, dtype=torch.int32) for _ in range(6)],
        *[torch.empty(batch, hkv, device=device, dtype=torch.int32) for _ in range(6)],
        torch.empty(batch, hkv, device=device, dtype=torch.int32),
        torch.empty(max_complete_tasks, device=device, dtype=torch.int32),
        torch.empty(max_complete_tasks, device=device, dtype=torch.int32),
        torch.empty(max_tail_tasks, device=device, dtype=torch.int32),
        torch.empty(max_tail_tasks, device=device, dtype=torch.int32),
        torch.empty(max_hit_tail_tasks, device=device, dtype=torch.int32),
        torch.empty(max_hit_tail_tasks, device=device, dtype=torch.int32),
        torch.empty(9, device=device, dtype=torch.int32),
        torch.empty(batch, hkv, max_tiles_context, group, dim, device=device, dtype=partial_o_dtype),
        torch.empty(batch, hkv, max_tiles_context, group, device=device, dtype=torch.float32),
        torch.empty(batch, hkv, max_tiles_reduce, group, dim, device=device, dtype=partial_o_dtype),
        torch.empty(batch, hkv, max_tiles_reduce, group, device=device, dtype=torch.float32),
        torch.empty(batch, hkv, max_tiles_tail, group, dim, device=device, dtype=partial_o_dtype),
        torch.empty(batch, hkv, max_tiles_tail, group, device=device, dtype=torch.float32),
        torch.empty(14, device=device, dtype=torch.int64),
    ]


def _make_case(
    batch: int,
    kv_len: int,
    hq: int,
    hkv: int,
    dim: int,
    capacity: int,
    hit_pattern: str,
):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    group = hq // hkv
    past_len = kv_len - 1
    q = torch.randn(batch, hq, dim, device=device, dtype=dtype)
    k = torch.randn(batch * kv_len, hkv, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)

    req_to_token = torch.empty(batch, kv_len, device=device, dtype=torch.int32)
    out_cache_loc = torch.empty(batch, device=device, dtype=torch.int32)
    for n in range(batch):
        base = n * kv_len
        req_to_token[n, :past_len] = torch.arange(base, base + past_len, device=device, dtype=torch.int32)
        req_to_token[n, past_len:] = base + past_len
        out_cache_loc[n] = base + past_len
    req_ids = torch.arange(batch, device=device, dtype=torch.int32)
    past_lens = torch.full((batch,), past_len, device=device, dtype=torch.int32)

    query_cache = torch.randn(batch, capacity, hq, dim, device=device, dtype=dtype) + 10
    attn_cache = torch.zeros_like(query_cache)
    lse_cache = torch.zeros(batch, capacity, hq, device=device, dtype=torch.float32)

    if hit_pattern == "all_hit":
        query_cache.copy_(q[:, None, :, :].expand_as(query_cache))
    elif hit_pattern == "all_miss":
        pass
    else:
        raise ValueError(f"unsupported hit pattern: {hit_pattern}")

    return {
        "q": q,
        "k": k,
        "v": v,
        "req_to_token": req_to_token,
        "req_ids": req_ids,
        "past_lens": past_lens,
        "out_cache_loc": out_cache_loc,
        "query_cache": query_cache,
        "attn_cache": attn_cache,
        "lse_cache": lse_cache,
        "out": torch.empty_like(q),
        "out_lse": torch.empty((batch, hq), device=device, dtype=torch.float32),
        "group": group,
    }


def _bench_mode_value(text: str) -> int:
    normalized = text.replace("-", "_")
    if normalized == "off":
        return 0
    if normalized == "synthetic_head":
        return 1
    if normalized == "synthetic_group":
        return 2
    raise ValueError(f"unsupported bench mode: {text}")


def _bench_one(ext, args, batch: int, kv_len: int, hit_pattern: str, tile_tokens: int):
    case = _make_case(
        batch=batch,
        kv_len=kv_len,
        hq=args.hq,
        hkv=args.hkv,
        dim=args.dim,
        capacity=args.capacity,
        hit_pattern=hit_pattern,
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
        torch.float32 if args.partial_fp32 else torch.bfloat16,
    )
    bench_mode = _bench_mode_value(args.bench_mode)
    gen_min_limit = kv_len + 1 if hit_pattern == "all_miss" and bench_mode == 0 else 0

    def launch():
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
            gen_min_limit,
            args.lookback_right,
            0,
            args.threshold,
            1 / math.sqrt(args.dim),
            args.debug,
            bench_mode,
            args.bench_hit_rate,
            args.bench_hit_rate_std,
            args.bench_skip_ratio,
            args.bench_skip_ratio_std,
            args.bench_match_lag_mean,
            args.bench_match_lag_std,
            args.bench_seed,
            -1,
            0,
        )

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        launch()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters
    row = {
        "batch": batch,
        "kv_len": kv_len,
        "hit_pattern": hit_pattern,
        "bench_mode": args.bench_mode,
        "bench_hit_rate": args.bench_hit_rate if bench_mode != 0 else "",
        "tile_tokens": tile_tokens,
        "kernel_ms": ms,
        "projected_32_layer_tok_s": 1000.0 / (ms * 32.0),
    }
    counts = [int(x) for x in workspace[22].detach().cpu().tolist()]
    row.update(
        {
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
    )
    if args.debug >= 2:
        cycles = [int(x) for x in workspace[-1].detach().cpu().tolist()]
        deltas = [max(cycles[i + 1] - cycles[i], 0) for i in range(5)]
        total = sum(deltas)
        names = ("match", "reduce", "rect", "tail", "merge")
        for name, delta in zip(names, deltas):
            row[f"{name}_pct"] = (100.0 * delta / total) if total else 0.0
        detailed = [
            ("prepare_pct", 1, 2),
            ("complete_pct", 2, 6),
            ("reduce_pct_detail", 6, 7),
            ("tail_pct_detail", 7, 8),
            ("fallback_group_merge_pct", 8, 9),
            ("merge_write_pct", 9, 10),
            ("cache_subtract_pct", 10, 11),
        ]
        detail_total = max(cycles[11] - cycles[0], 0)
        for name, begin, end in detailed:
            delta = max(cycles[end] - cycles[begin], 0)
            row[name] = (100.0 * delta / detail_total) if detail_total else 0.0
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--kv-len", type=int, nargs="+", default=[65536])
    parser.add_argument("--hit-pattern", choices=["all_hit", "all_miss"], nargs="+", default=["all_hit"])
    parser.add_argument("--tile-tokens", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--semantic-pos-ahead", type=int, default=256)
    parser.add_argument("--match-tile-slots", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--lookback-right", type=int, default=0)
    parser.add_argument("--hq", type=int, default=32)
    parser.add_argument("--hkv", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--partial-fp32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--bench-mode",
        choices=["off", "synthetic_head", "synthetic-head", "synthetic_group", "synthetic-group"],
        default="off",
    )
    parser.add_argument("--bench-hit-rate", type=float, default=1.0)
    parser.add_argument("--bench-hit-rate-std", type=float, default=0.0)
    parser.add_argument("--bench-skip-ratio", type=float, default=-1.0)
    parser.add_argument("--bench-skip-ratio-std", type=float, default=0.0)
    parser.add_argument("--bench-match-lag-mean", type=float, default=64.0)
    parser.add_argument("--bench-match-lag-std", type=float, default=0.0)
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    ext = load_mac_persistent_decode_extension(False)
    rows = []
    for batch in args.batch:
        for kv_len in args.kv_len:
            if kv_len > args.max_context:
                raise ValueError(f"kv_len={kv_len} exceeds max_context={args.max_context}")
            for hit_pattern in args.hit_pattern:
                for tile_tokens in args.tile_tokens:
                    row = _bench_one(ext, args, batch, kv_len, hit_pattern, tile_tokens)
                    rows.append(row)
                    print(
                        "batch={batch} kv_len={kv_len} pattern={hit_pattern} tile={tile_tokens} "
                        "kernel_ms={kernel_ms:.4f} projected_32_layer_tok_s={projected_32_layer_tok_s:.2f}".format(
                            **row
                        ),
                        end="",
                    )
                    if args.debug >= 2:
                        print(
                            " phases="
                            f"match:{row['match_pct']:.1f}% "
                            f"reduce:{row['reduce_pct']:.1f}% "
                            f"rect:{row['rect_pct']:.1f}% "
                            f"tail:{row['tail_pct']:.1f}% "
                            f"merge:{row['merge_pct']:.1f}%",
                            flush=True,
                        )
                    else:
                        print(flush=True)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
