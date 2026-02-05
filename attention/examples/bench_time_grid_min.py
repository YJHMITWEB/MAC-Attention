import math
import time

import torch

from mac_attention import MACDecodeWithPagedKVCacheWrapper


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = 2
    Hkv = 2
    group_size = 4
    Hq = Hkv * group_size
    D = 128
    kv_len = 4096
    page_size = 1

    total_pages = B * kv_len
    k_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)
    v_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)
    q = torch.randn(B, Hq, D, device=device, dtype=dtype)

    indptr = torch.tensor([0, kv_len, 2 * kv_len], device=device, dtype=torch.int32)
    indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.ones((B,), device=device, dtype=torch.int32)
    attn_start_pos = torch.zeros((B, Hq), device=device, dtype=torch.int32)

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = MACDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")

    # JIT + plan
    t0 = time.time()
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        Hq,
        Hkv,
        D,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        data_type=dtype,
        attn_start_pos=attn_start_pos,
        downdate_range=0,
    )
    torch.cuda.synchronize()
    print(f"plan() time: {(time.time() - t0) * 1e3:.2f} ms")

    dummy_cache = torch.empty(1, device=device, dtype=dtype)
    dummy_f32 = torch.empty(1, device=device, dtype=torch.float32)
    dummy_i32 = torch.empty(1, device=device, dtype=torch.int32)
    dummy_bool = torch.empty(1, device=device, dtype=torch.bool)

    def run_once():
        return wrapper.forward_return_lse(
            q,
            (k_pages, v_pages),
            0,
            0,
            dummy_cache,
            dummy_f32,
            dummy_i32,
            dummy_bool,
            dummy_i32,
            False,
            attn_start_pos,
            sm_scale=1.0 / math.sqrt(D),
        )

    # Warmup
    for _ in range(10):
        run_once()
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(50):
        run_once()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 50.0
    print(f"avg forward_return_lse: {ms:.3f} ms")


if __name__ == "__main__":
    main()

