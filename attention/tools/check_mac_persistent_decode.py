import math
import os

import torch

from mac_attention.integrations.sglang.bridge import load_mac_persistent_decode_extension


def _ref_attention(q, k, v, start, end, sm_scale):
    hq, dim = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    outs = []
    lses = []
    for head in range(hq):
        if end <= start:
            outs.append(torch.zeros(dim, device=q.device, dtype=q.dtype))
            lses.append(torch.tensor(-float("inf"), device=q.device))
            continue
        kv_head = head // group
        kk = k[start:end, kv_head].float()
        vv = v[start:end, kv_head].float()
        logits = (kk * q[head].float()).sum(-1) * sm_scale
        lse = torch.logsumexp(logits, dim=0)
        out = (torch.exp(logits - lse)[:, None] * vv).sum(0).to(q.dtype)
        outs.append(out)
        lses.append(lse)
    return torch.stack(outs), torch.stack(lses)


def _merge(o1, l1, o2, l2):
    outs = []
    lses = []
    for head in range(o1.shape[0]):
        if not torch.isfinite(l1[head]):
            outs.append(o2[head])
            lses.append(l2[head])
            continue
        if not torch.isfinite(l2[head]):
            outs.append(o1[head])
            lses.append(l1[head])
            continue
        m = torch.maximum(l1[head], l2[head])
        w1 = torch.exp(l1[head] - m)
        w2 = torch.exp(l2[head] - m)
        denom = w1 + w2
        lse = m + torch.log(denom)
        out = ((w1 * o1[head].float() + w2 * o2[head].float()) / denom).to(o1.dtype)
        outs.append(out)
        lses.append(lse)
    return torch.stack(outs), torch.stack(lses)


def _workspace(batch, hq, hkv, group, dim, capacity, max_context, tile_tokens, match_tile_slots, semantic_pos_ahead):
    max_match_tiles = (capacity + match_tile_slots - 1) // match_tile_slots
    max_tiles_context = (max_context + tile_tokens - 1) // tile_tokens
    max_tiles_reduce = (max_context + tile_tokens * 32 - 1) // (tile_tokens * 32)
    max_tiles_tail = max(1, (max(semantic_pos_ahead, 1) + tile_tokens - 1) // tile_tokens)
    device = torch.device("cuda")
    max_complete_tasks = batch * hq * max_tiles_context
    max_tail_tasks = batch * hkv * max_tiles_tail
    max_hit_tail_tasks = batch * hq * max_tiles_tail
    partial_o_dtype = torch.bfloat16 if os.getenv("MAC_CHECK_PARTIAL_BF16", "1") == "1" else torch.float32
    return [
        torch.empty(batch, hq, max_match_tiles, device=device),
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
        torch.empty(batch, hkv, max_tiles_context, group, device=device),
        torch.empty(batch, hkv, max_tiles_reduce, group, dim, device=device, dtype=partial_o_dtype),
        torch.empty(batch, hkv, max_tiles_reduce, group, device=device),
        torch.empty(batch, hkv, max_tiles_tail, group, dim, device=device, dtype=partial_o_dtype),
        torch.empty(batch, hkv, max_tiles_tail, group, device=device),
        torch.empty(6, device=device, dtype=torch.int64),
    ]


def _run_ext(
    ext,
    q,
    k,
    v,
    req_to_token,
    req_ids,
    past_lens,
    out_cache_loc,
    query_cache,
    attn_cache,
    lse_cache,
    max_context,
    tile_tokens,
    match_tile_slots,
    semantic_pos_ahead,
    threshold=0.45,
):
    batch, hq, dim = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    out = torch.empty_like(q)
    out_lse = torch.empty(batch, hq, device=q.device)
    workspace = _workspace(
        batch,
        hq,
        hkv,
        group,
        dim,
        query_cache.shape[1],
        max_context,
        tile_tokens,
        match_tile_slots,
        semantic_pos_ahead,
    )
    ext.mac_persistent_decode_bf16(
        q,
        q,
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
        max_context,
        tile_tokens,
        match_tile_slots,
        semantic_pos_ahead,
        0,
        0,
        0,
        threshold,
        1 / math.sqrt(dim),
        0,
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
    torch.cuda.synchronize()
    return out, out_lse, workspace


def _check_hit_clean_half_open_cache_row(ext):
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, hkv, group, dim = 1, 2, 4, 128
    hq = hkv * group
    capacity, kv_len, semantic_pos_ahead = 8, 10, 3
    past_len = kv_len - 1
    sm_scale = 1 / math.sqrt(dim)
    k = torch.randn(kv_len, hkv, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q = torch.randn(batch, hq, dim, device=device, dtype=dtype)
    req_to_token = torch.zeros(1, 64, device=device, dtype=torch.int32)
    req_to_token[0, :past_len] = torch.arange(past_len, device=device, dtype=torch.int32)
    req_ids = torch.tensor([0], device=device, dtype=torch.int32)
    past_lens = torch.tensor([past_len], device=device, dtype=torch.int32)
    out_cache_loc = torch.tensor([past_len], device=device, dtype=torch.int64)
    query_cache = torch.randn(1, capacity, hq, dim, device=device, dtype=dtype)
    attn_cache = torch.zeros_like(query_cache)
    lse_cache = torch.full((1, capacity, hq), -float("inf"), device=device)
    match_pos = 5
    match_slot = match_pos % capacity
    query_cache[0, match_slot].copy_(q[0])
    prefix_end = max(match_pos + 1 - semantic_pos_ahead, 0)
    prefix_o, prefix_lse = _ref_attention(q[0], k, v, 0, prefix_end, sm_scale)
    attn_cache[0, match_slot].copy_(prefix_o)
    lse_cache[0, match_slot].copy_(prefix_lse)
    out, out_lse, _ = _run_ext(
        ext,
        q,
        k,
        v,
        req_to_token,
        req_ids,
        past_lens,
        out_cache_loc,
        query_cache,
        attn_cache,
        lse_cache,
        16,
        4,
        4,
        semantic_pos_ahead,
    )
    new_end = max(past_len + 1 - semantic_pos_ahead, 0)
    rect_o, rect_lse = _ref_attention(q[0], k, v, prefix_end, new_end, sm_scale)
    tail_o, tail_lse = _ref_attention(q[0], k, v, new_end, kv_len, sm_scale)
    cache_o, cache_lse = _merge(prefix_o, prefix_lse, rect_o, rect_lse)
    ref_o, ref_lse = _merge(cache_o, cache_lse, tail_o, tail_lse)
    torch.testing.assert_close(out[0], ref_o, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out_lse[0], ref_lse, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(attn_cache[0, past_len % capacity], cache_o, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse_cache[0, past_len % capacity], cache_lse, rtol=3e-2, atol=3e-2)


def _check_miss_multi_request(ext):
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, hkv, group, dim = 2, 2, 4, 128
    hq = hkv * group
    capacity, kv_len, semantic_pos_ahead = 8, 12, 4
    past_len = kv_len - 1
    sm_scale = 1 / math.sqrt(dim)
    k = torch.randn(batch * kv_len, hkv, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q = torch.randn(batch, hq, dim, device=device, dtype=dtype)
    req_to_token = torch.zeros(batch, 64, device=device, dtype=torch.int32)
    for n in range(batch):
        req_to_token[n, :past_len] = torch.arange(
            n * kv_len, n * kv_len + past_len, device=device, dtype=torch.int32
        )
    req_ids = torch.arange(batch, device=device, dtype=torch.int32)
    past_lens = torch.full((batch,), past_len, device=device, dtype=torch.int32)
    out_cache_loc = torch.tensor([past_len, kv_len + past_len], device=device, dtype=torch.int32)
    query_cache = torch.randn(batch, capacity, hq, dim, device=device, dtype=dtype) + 10
    attn_cache = torch.zeros_like(query_cache)
    lse_cache = torch.full((batch, capacity, hq), -float("inf"), device=device)
    out, out_lse, _ = _run_ext(
        ext,
        q,
        k,
        v,
        req_to_token,
        req_ids,
        past_lens,
        out_cache_loc,
        query_cache,
        attn_cache,
        lse_cache,
        16,
        4,
        4,
        semantic_pos_ahead,
    )
    for n in range(batch):
        kk = k[n * kv_len : n * kv_len + kv_len]
        vv = v[n * kv_len : n * kv_len + kv_len]
        ref_o, ref_lse = _ref_attention(q[n], kk, vv, 0, kv_len, sm_scale)
        cache_o, cache_lse = _ref_attention(q[n], kk, vv, 0, kv_len - semantic_pos_ahead, sm_scale)
        torch.testing.assert_close(out[n], ref_o, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(out_lse[n], ref_lse, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(attn_cache[n, past_len % capacity], cache_o, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(lse_cache[n, past_len % capacity], cache_lse, rtol=3e-2, atol=3e-2)


def _check_large_miss_reduced_rect(ext):
    torch.manual_seed(22)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, hkv, group, dim = 1, 2, 4, 128
    hq = hkv * group
    capacity, kv_len, semantic_pos_ahead = 8, 192, 4
    tile_tokens, max_context = 4, 256
    past_len = kv_len - 1
    sm_scale = 1 / math.sqrt(dim)
    k = torch.randn(kv_len, hkv, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q = torch.randn(batch, hq, dim, device=device, dtype=dtype)
    req_to_token = torch.zeros(1, max_context, device=device, dtype=torch.int32)
    req_to_token[0, :past_len] = torch.arange(past_len, device=device, dtype=torch.int32)
    req_ids = torch.tensor([0], device=device, dtype=torch.int32)
    past_lens = torch.tensor([past_len], device=device, dtype=torch.int32)
    out_cache_loc = torch.tensor([past_len], device=device, dtype=torch.int32)
    query_cache = torch.randn(1, capacity, hq, dim, device=device, dtype=dtype) + 10
    attn_cache = torch.zeros_like(query_cache)
    lse_cache = torch.full((1, capacity, hq), -float("inf"), device=device)
    out, out_lse, workspace = _run_ext(
        ext,
        q,
        k,
        v,
        req_to_token,
        req_ids,
        past_lens,
        out_cache_loc,
        query_cache,
        attn_cache,
        lse_cache,
        max_context,
        tile_tokens,
        4,
        semantic_pos_ahead,
    )
    ref_o, ref_lse = _ref_attention(q[0], k, v, 0, kv_len, sm_scale)
    cache_o, cache_lse = _ref_attention(q[0], k, v, 0, kv_len - semantic_pos_ahead, sm_scale)
    torch.testing.assert_close(out[0], ref_o, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out_lse[0], ref_lse, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(attn_cache[0, past_len % capacity], cache_o, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse_cache[0, past_len % capacity], cache_lse, rtol=3e-2, atol=3e-2)
    assert int(workspace[22][3].item()) > 0


def _check_mixed_group_reduced_rect(ext):
    old_mixed = os.environ.get("MAC_PERSISTENT_MIXED_GROUP_FALLBACK")
    os.environ["MAC_PERSISTENT_MIXED_GROUP_FALLBACK"] = "1"
    torch.manual_seed(23)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, hkv, group, dim = 1, 1, 4, 128
    hq = hkv * group
    capacity, kv_len, semantic_pos_ahead = 8, 192, 4
    tile_tokens, max_context = 4, 256
    past_len = kv_len - 1
    sm_scale = 1 / math.sqrt(dim)
    k = torch.randn(kv_len, hkv, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q = torch.randn(batch, hq, dim, device=device, dtype=dtype)
    req_to_token = torch.zeros(1, max_context, device=device, dtype=torch.int32)
    req_to_token[0, :past_len] = torch.arange(past_len, device=device, dtype=torch.int32)
    req_ids = torch.tensor([0], device=device, dtype=torch.int32)
    past_lens = torch.tensor([past_len], device=device, dtype=torch.int32)
    out_cache_loc = torch.tensor([past_len], device=device, dtype=torch.int32)
    query_cache = torch.randn(1, capacity, hq, dim, device=device, dtype=dtype) + 10
    attn_cache = torch.zeros_like(query_cache)
    lse_cache = torch.full((1, capacity, hq), -float("inf"), device=device)

    hit_heads = torch.tensor([True, True, False, True], device=device)
    hit_idx = torch.tensor([0, 1, 3], device=device)
    match_pos = 186
    match_slot = match_pos % capacity
    prefix_end = max(match_pos + 1 - semantic_pos_ahead, 0)
    prefix_o, prefix_lse = _ref_attention(q[0], k, v, 0, prefix_end, sm_scale)
    query_cache[0, match_slot, hit_idx] = q[0, hit_idx]
    attn_cache[0, match_slot, hit_idx] = prefix_o[hit_idx]
    lse_cache[0, match_slot, hit_idx] = prefix_lse[hit_idx]

    out, out_lse, workspace = _run_ext(
        ext,
        q,
        k,
        v,
        req_to_token,
        req_ids,
        past_lens,
        out_cache_loc,
        query_cache,
        attn_cache,
        lse_cache,
        max_context,
        tile_tokens,
        4,
        semantic_pos_ahead,
    )
    new_end = max(past_len + 1 - semantic_pos_ahead, 0)
    rect_o, rect_lse = _ref_attention(q[0], k, v, prefix_end, new_end, sm_scale)
    tail_o, tail_lse = _ref_attention(q[0], k, v, new_end, kv_len, sm_scale)
    hit_cache_o, hit_cache_lse = _merge(prefix_o, prefix_lse, rect_o, rect_lse)
    hit_out_o, hit_out_lse = _merge(hit_cache_o, hit_cache_lse, tail_o, tail_lse)
    full_o, full_lse = _ref_attention(q[0], k, v, 0, kv_len, sm_scale)
    full_cache_o, full_cache_lse = _ref_attention(q[0], k, v, 0, new_end, sm_scale)

    ref_o = full_o.clone()
    ref_lse = full_lse.clone()
    cache_o = full_cache_o.clone()
    cache_lse = full_cache_lse.clone()
    ref_o[hit_heads] = hit_out_o[hit_heads]
    ref_lse[hit_heads] = hit_out_lse[hit_heads]
    cache_o[hit_heads] = hit_cache_o[hit_heads]
    cache_lse[hit_heads] = hit_cache_lse[hit_heads]

    try:
        torch.testing.assert_close(workspace[15][0], torch.full((hkv,), 2, device=device, dtype=torch.int32))
        torch.testing.assert_close(out[0], ref_o, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(out_lse[0], ref_lse, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(attn_cache[0, past_len % capacity], cache_o, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(lse_cache[0, past_len % capacity], cache_lse, rtol=3e-2, atol=3e-2)
    finally:
        if old_mixed is None:
            os.environ.pop("MAC_PERSISTENT_MIXED_GROUP_FALLBACK", None)
        else:
            os.environ["MAC_PERSISTENT_MIXED_GROUP_FALLBACK"] = old_mixed


def _check_short_context_empty_cache(ext):
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, hkv, group, dim = 1, 2, 4, 128
    hq = hkv * group
    capacity, kv_len, semantic_pos_ahead = 8, 3, 4
    past_len = kv_len - 1
    sm_scale = 1 / math.sqrt(dim)
    k = torch.randn(kv_len, hkv, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q = torch.randn(batch, hq, dim, device=device, dtype=dtype)
    req_to_token = torch.zeros(1, 32, device=device, dtype=torch.int32)
    req_to_token[0, :past_len] = torch.arange(past_len, device=device, dtype=torch.int32)
    req_ids = torch.tensor([0], device=device, dtype=torch.int32)
    past_lens = torch.tensor([past_len], device=device, dtype=torch.int32)
    out_cache_loc = torch.tensor([past_len], device=device, dtype=torch.int32)
    query_cache = torch.randn(1, capacity, hq, dim, device=device, dtype=dtype) + 10
    attn_cache = torch.ones_like(query_cache)
    lse_cache = torch.zeros((1, capacity, hq), device=device)
    out, out_lse, _ = _run_ext(
        ext,
        q,
        k,
        v,
        req_to_token,
        req_ids,
        past_lens,
        out_cache_loc,
        query_cache,
        attn_cache,
        lse_cache,
        16,
        4,
        4,
        semantic_pos_ahead,
    )
    ref_o, ref_lse = _ref_attention(q[0], k, v, 0, kv_len, sm_scale)
    dest_slot = past_len % capacity
    torch.testing.assert_close(out[0], ref_o, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out_lse[0], ref_lse, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(query_cache[0, dest_slot], q[0], rtol=0, atol=0)
    torch.testing.assert_close(attn_cache[0, dest_slot], torch.zeros_like(q[0]), rtol=0, atol=0)
    assert torch.all(torch.isneginf(lse_cache[0, dest_slot]))


def _check_tie_break_after_wraparound(ext):
    torch.manual_seed(4)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, hkv, group, dim = 1, 2, 4, 128
    hq = hkv * group
    capacity, kv_len, semantic_pos_ahead = 4, 10, 2
    past_len = kv_len - 1
    sm_scale = 1 / math.sqrt(dim)
    k = torch.randn(kv_len, hkv, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q = torch.randn(batch, hq, dim, device=device, dtype=dtype)
    req_to_token = torch.zeros(1, 32, device=device, dtype=torch.int32)
    req_to_token[0, :past_len] = torch.arange(past_len, device=device, dtype=torch.int32)
    req_ids = torch.tensor([0], device=device, dtype=torch.int32)
    past_lens = torch.tensor([past_len], device=device, dtype=torch.int32)
    out_cache_loc = torch.tensor([past_len], device=device, dtype=torch.int32)
    query_cache = torch.randn(1, capacity, hq, dim, device=device, dtype=dtype) + 10
    attn_cache = torch.zeros_like(query_cache)
    lse_cache = torch.full((1, capacity, hq), -float("inf"), device=device)
    first_pos, later_pos = 6, 7
    first_slot, later_slot = first_pos % capacity, later_pos % capacity
    query_cache[0, first_slot].copy_(q[0])
    query_cache[0, later_slot].copy_(q[0])
    first_prefix_end = max(first_pos + 1 - semantic_pos_ahead, 0)
    first_prefix_o, first_prefix_lse = _ref_attention(q[0], k, v, 0, first_prefix_end, sm_scale)
    attn_cache[0, first_slot].copy_(first_prefix_o)
    lse_cache[0, first_slot].copy_(first_prefix_lse)
    attn_cache[0, later_slot].copy_(torch.randn_like(first_prefix_o))
    lse_cache[0, later_slot].copy_(torch.randn_like(first_prefix_lse))
    out, out_lse, workspace = _run_ext(
        ext,
        q,
        k,
        v,
        req_to_token,
        req_ids,
        past_lens,
        out_cache_loc,
        query_cache,
        attn_cache,
        lse_cache,
        16,
        4,
        2,
        semantic_pos_ahead,
    )
    new_end = max(past_len + 1 - semantic_pos_ahead, 0)
    rect_o, rect_lse = _ref_attention(q[0], k, v, first_prefix_end, new_end, sm_scale)
    tail_o, tail_lse = _ref_attention(q[0], k, v, new_end, kv_len, sm_scale)
    cache_o, cache_lse = _merge(first_prefix_o, first_prefix_lse, rect_o, rect_lse)
    ref_o, ref_lse = _merge(cache_o, cache_lse, tail_o, tail_lse)
    torch.testing.assert_close(workspace[5][0], torch.full((hq,), first_pos, device=device, dtype=torch.int32))
    torch.testing.assert_close(out[0], ref_o, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out_lse[0], ref_lse, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(attn_cache[0, past_len % capacity], cache_o, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse_cache[0, past_len % capacity], cache_lse, rtol=3e-2, atol=3e-2)


def main():
    assert torch.cuda.is_available(), "CUDA required"
    ext = load_mac_persistent_decode_extension(False)
    checks = [
        ("hit_clean_half_open_cache_row", _check_hit_clean_half_open_cache_row),
        ("miss_multi_request", _check_miss_multi_request),
        ("large_miss_reduced_rect", _check_large_miss_reduced_rect),
        ("mixed_group_reduced_rect", _check_mixed_group_reduced_rect),
        ("short_context_empty_cache", _check_short_context_empty_cache),
        ("tie_break_after_wraparound", _check_tie_break_after_wraparound),
    ]
    for name, fn in checks:
        fn(ext)
        print(f"PASS {name}")


if __name__ == "__main__":
    main()
