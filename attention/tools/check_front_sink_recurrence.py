"""Two-step front-sink EXCLUSION recurrence check (GQA-8, mixed -> all-hit).

The single-step checks in check_mac_persistent_decode.py cannot see bugs that
live in the cache-write -> later-reuse recurrence (that class of bug caused the
2026-07-05 Qwen3 repetition collapse). This tool runs TWO chained decode steps
through the real kernel and verifies the exclusion invariant end to end:

Step 1: MIXED GQA-8 group (6 hit / 2 miss heads), front_sink_tokens = F > 0.
        Every written ring entry must cover exactly [min(F, new_end), new_end):
        hit heads (matched entry [F, pend) PLUS band), miss heads (fresh
        [F, new_end)). No metadata, no sentinels.
Step 2: all 8 heads match the step-1 entry (pure-hit group). Reuse must be
        EXACT for every head: sink [0, F) recomputed under the step-2 query,
        merged with the entry, band, and tail.

Usage: python attention/tools/check_front_sink_recurrence.py
"""
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MAC_PERSISTENT_MIXED_GROUP_FALLBACK", "1")

from mac_attention.integrations.sglang.bridge import load_mac_persistent_decode_extension
from check_mac_persistent_decode import _ref_attention, _merge, _workspace


def _run(ext, q_post, q_pre, k, v, r2t, req, pl, ocl, qc, ac, lc,
         maxctx, tile, mts, sem, thr, F):
    b, hq, dim = q_post.shape
    hkv = k.shape[1]
    group = hq // hkv
    out = torch.empty_like(q_post)
    out_lse = torch.empty(b, hq, device=q_post.device)
    ws = _workspace(b, hq, hkv, group, dim, qc.shape[1], maxctx, tile, mts, sem, F)
    ext.mac_persistent_decode_bf16(
        q_post, q_pre, k, v, r2t, req, pl, ocl, qc, ac, lc, out, out_lse, *ws,
        maxctx, tile, mts, sem, F, 0, 0, 0, thr, 1 / math.sqrt(dim),
        0, 0, 1.0, 0.0, -1.0, 0.0, 64.0, 0.0, 1, -1, 0, qc, torch.empty(0), torch.empty(0), torch.empty(0))
    torch.cuda.synchronize()
    return out, out_lse, ws


def _rel(a, b):
    return (torch.norm(a.float() - b.float()) / (torch.norm(b.float()) + 1e-9)).item()


def main():
    assert torch.cuda.is_available(), "CUDA required"
    ext = load_mac_persistent_decode_extension(False)
    torch.manual_seed(11)
    dev, dt = torch.device("cuda"), torch.bfloat16
    hkv, group = 1, 8
    hq = hkv * group
    dim = 128
    cap, sem, tile, F, mts = 16, 4, 4, 3, 4
    sm = 1 / math.sqrt(dim)
    thr = 0.45
    kv_len2 = 21
    kv_len1 = 20
    past1 = kv_len1 - 1
    past2 = past1 + 1
    k = torch.randn(kv_len2, hkv, dim, device=dev, dtype=dt)
    v = torch.randn_like(k)
    r2t = torch.zeros(1, 64, device=dev, dtype=torch.int32)
    r2t[0, :past2] = torch.arange(past2, device=dev, dtype=torch.int32)
    req = torch.tensor([0], device=dev, dtype=torch.int32)

    q_pre1 = torch.randn(1, hq, dim, device=dev, dtype=dt)
    q_post1 = torch.randn(1, hq, dim, device=dev, dtype=dt)
    q_build = torch.randn(hq, dim, device=dev, dtype=dt)
    hit_heads = list(range(6))

    qc = torch.randn(1, cap, hq, dim, device=dev, dtype=dt) + 10
    ac = torch.zeros_like(qc)
    lc = torch.full((1, cap, hq), -float("inf"), device=dev)

    mpos1 = 10
    mslot1 = mpos1 % cap
    pend1 = mpos1 + 1 - sem
    nend1 = past1 + 1 - sem
    entry_o, entry_l = _ref_attention(q_build, k, v, F, pend1, sm)  # exclusion: [F, pend1)
    for h in hit_heads:
        qc[0, mslot1, h] = (q_pre1[0, h].float() + 0.03 * torch.randn(dim, device=dev)).to(dt)
        ac[0, mslot1, h] = entry_o[h]
        lc[0, mslot1, h] = entry_l[h]

    pl = torch.tensor([past1], device=dev, dtype=torch.int32)
    ocl = torch.tensor([past1], device=dev, dtype=torch.int32)
    _, _, ws1 = _run(ext, q_post1.clone(), q_pre1.clone(), k, v, r2t, req, pl, ocl,
                     qc, ac, lc, 32, tile, mts, sem, thr, F)
    assert ws1[3][0].tolist() == [1] * 6 + [0] * 2, f"step1 hit pattern: {ws1[3][0].tolist()}"

    band1_o, band1_l = _ref_attention(q_post1[0], k, v, pend1, nend1, sm)
    e_hit_o, e_hit_l = _merge(entry_o, entry_l, band1_o, band1_l)        # [F, nend1)
    e_miss_o, e_miss_l = _ref_attention(q_post1[0], k, v, F, nend1, sm)  # [F, nend1)
    dslot1 = past1 % cap
    err_hit = max(_rel(ac[0, dslot1, h], e_hit_o[h]) for h in hit_heads)
    err_miss = max(_rel(ac[0, dslot1, h], e_miss_o[h]) for h in (6, 7))
    print(f"step1 entry rel_l2: hit-heads={err_hit:.4f} miss-heads={err_miss:.4f}")
    ok1 = err_hit < 0.08 and err_miss < 0.08
    print("PASS step1 written entries exclude sink" if ok1 else "FAIL step1 entries")

    q_pre2 = q_pre1.clone()
    q_post2 = torch.randn(1, hq, dim, device=dev, dtype=dt)
    pl2 = torch.tensor([past2], device=dev, dtype=torch.int32)
    ocl2 = torch.tensor([past2], device=dev, dtype=torch.int32)
    out2, _, ws2 = _run(ext, q_post2.clone(), q_pre2, k, v, r2t, req, pl2, ocl2,
                        qc, ac, lc, 32, tile, mts, sem, thr, F)
    assert ws2[3][0].tolist() == [1] * hq, f"step2 not all-hit: {ws2[3][0].tolist()}"
    assert ws2[5][0].tolist() == [past1] * hq, f"step2 wrong match pos: {ws2[5][0].tolist()}"

    nend2 = past2 + 1 - sem
    sink2_o, sink2_l = _ref_attention(q_post2[0], k, v, 0, F, sm)
    rect2_o, rect2_l = _ref_attention(q_post2[0], k, v, nend1, nend2, sm)
    tail2_o, tail2_l = _ref_attention(q_post2[0], k, v, nend2, kv_len2, sm)
    ref = torch.zeros(hq, dim, device=dev, dtype=dt)
    for h in range(hq):
        base_o = e_hit_o[h:h + 1] if h in hit_heads else e_miss_o[h:h + 1]
        base_l = e_hit_l[h:h + 1] if h in hit_heads else e_miss_l[h:h + 1]
        o, l = _merge(base_o, base_l, rect2_o[h:h + 1], rect2_l[h:h + 1])
        o, l = _merge(o, l, sink2_o[h:h + 1], sink2_l[h:h + 1])
        o, l = _merge(o, l, tail2_o[h:h + 1], tail2_l[h:h + 1])
        ref[h] = o[0]
    err2 = _rel(out2[0], ref)
    print(f"step2 rel_l2(kernel, exact recurrence ref) = {err2:.4f}")
    ok2 = err2 < 0.08
    print("PASS step2 exact reuse after mixed write" if ok2 else "FAIL step2")
    print("VERDICT:", "EXCLUSION RECURRENCE OK" if (ok1 and ok2) else "EXCLUSION RECURRENCE BROKEN")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
