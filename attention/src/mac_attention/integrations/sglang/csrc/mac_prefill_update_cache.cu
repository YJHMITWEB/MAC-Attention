// mac_prefill_update_cache.cu
//
// Cooperative row-update kernel for MAC ring caches.
// - Copies rows from source tensors into (q_cache, attn_cache, lse_cache) at circular positions.
// - Performs a single in-kernel atomic bump to request_length[req] after copies.
// - Chooses a right-sized thread count based on row byte size.

#include <ATen/cuda/CUDAContext.h>   // getCurrentCUDAStream
#include <c10/cuda/CUDAGuard.h>      // c10::cuda::CUDAGuard
#include <c10/cuda/CUDAException.h>  // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <c10/cuda/CUDAStream.h>     // CUDAStreamGuard
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <vector>

using torch::Tensor;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).dtype() == (dt), #x " has wrong dtype")

// --------------------------------- utils ----------------------------------

__device__ __forceinline__ void copy_bytes_16(void* __restrict__ dst_void,
                                              const void* __restrict__ src_void,
                                              size_t n_bytes) {
  if (n_bytes == 0) return;
  uintptr_t du = reinterpret_cast<uintptr_t>(dst_void);
  uintptr_t su = reinterpret_cast<uintptr_t>(src_void);
  uint8_t* __restrict__ d8 = (uint8_t*)dst_void;
  const uint8_t* __restrict__ s8 = (const uint8_t*)src_void;

  if (((du | su) & 0xF) == 0) {
    size_t n16 = n_bytes / 16;
    uint4* __restrict__ d16 = (uint4*)dst_void;
    const uint4* __restrict__ s16 = (const uint4*)src_void;
#pragma unroll 4
    for (size_t i = threadIdx.x; i < n16; i += blockDim.x) {
      uint4 v = s16[i];
      d16[i] = v;
    }
    size_t rem = n_bytes - n16 * 16;
    if (rem) {
      size_t base = n16 * 16;
      for (size_t i = threadIdx.x; i < rem; i += blockDim.x) {
        d8[base + i] = s8[base + i];
      }
    }
  } else if (((du | su) & 0x7) == 0) {
    size_t n8 = n_bytes / 8;
    uint2* __restrict__ d8b = (uint2*)dst_void;
    const uint2* __restrict__ s8b = (const uint2*)src_void;
#pragma unroll 4
    for (size_t i = threadIdx.x; i < n8; i += blockDim.x) {
      uint2 v = s8b[i];
      d8b[i] = v;
    }
    size_t rem = n_bytes - n8 * 8;
    if (rem) {
      size_t base = n8 * 8;
      for (size_t i = threadIdx.x; i < rem; i += blockDim.x) {
        d8[base + i] = s8[base + i];
      }
    }
  } else {
    for (size_t i = threadIdx.x; i < n_bytes; i += blockDim.x) {
      d8[i] = s8[i];
    }
  }
}

struct UpdateParams {
  void* q_cache;        // [R,M,H,D] bf16
  void* q_post_cache;   // [R,M,H,D] bf16 (optional; NaN-sentinel filled)
  void* attn_cache;     // [R,M,H,D] bf16
  void* lse_cache;      // [R,M,H]    f32
  const void* q_src;    // [sumN,H,D] bf16
  const void* attn_src; // [sumN,H,D] bf16
  const void* lse_src;  // [sumN,H]   f32
  int R, M, H, D;
  int32_t src_rows;
  const int32_t* req_ids;   // [B]
  int32_t B;
  const int32_t* offsets;   // [B+1] (rows)
  const int32_t* lens;      // [B]
  int32_t* request_length;  // [R] legacy indexed-by-req, or [B] prefix lengths
  bool request_length_by_batch;
};

// ------------------------------- helpers ----------------------------------

static void fill_params(UpdateParams& p,
                        Tensor q_cache,
                        Tensor attn_cache,
                        Tensor lse_cache,
                        Tensor request_length,
                        Tensor req_ids,
                        Tensor src_q,
                        Tensor src_attn,
                        Tensor src_lse,
                        Tensor offsets,
                        Tensor lens,
                        int capacity,
                        int num_heads,
                        int head_dim,
                        Tensor q_post_cache) {
  CHECK_CUDA(q_cache);
  CHECK_CONTIG(q_cache);
  CHECK_DTYPE(q_cache, torch::kBFloat16);
  CHECK_CUDA(attn_cache);
  CHECK_CONTIG(attn_cache);
  CHECK_DTYPE(attn_cache, torch::kBFloat16);
  CHECK_CUDA(lse_cache);
  CHECK_CONTIG(lse_cache);
  CHECK_DTYPE(lse_cache, torch::kFloat32);
  CHECK_CUDA(request_length);
  CHECK_CONTIG(request_length);
  CHECK_DTYPE(request_length, torch::kInt32);

  CHECK_CUDA(req_ids);
  CHECK_CONTIG(req_ids);
  CHECK_DTYPE(req_ids, torch::kInt32);
  CHECK_CUDA(lens);
  CHECK_CONTIG(lens);
  CHECK_DTYPE(lens, torch::kInt32);
  CHECK_CUDA(offsets);
  CHECK_CONTIG(offsets);
  CHECK_DTYPE(offsets, torch::kInt32);

  CHECK_CUDA(src_q);
  CHECK_CONTIG(src_q);
  CHECK_DTYPE(src_q, torch::kBFloat16);
  CHECK_CUDA(src_attn);
  CHECK_CONTIG(src_attn);
  CHECK_DTYPE(src_attn, torch::kBFloat16);
  CHECK_CUDA(src_lse);
  CHECK_CONTIG(src_lse);
  CHECK_DTYPE(src_lse, torch::kFloat32);

  int R = (int)q_cache.size(0);
  TORCH_CHECK(R > 0, "R must be > 0");
  TORCH_CHECK(q_cache.dim() == 4, "q_cache must be [R,M,H,D]");
  TORCH_CHECK(q_cache.size(1) == capacity, "capacity (M) mismatch");
  TORCH_CHECK(q_cache.size(2) == num_heads, "num_heads (H) mismatch");
  TORCH_CHECK(q_cache.size(3) == head_dim, "head_dim (D) mismatch");
  TORCH_CHECK(attn_cache.sizes() == q_cache.sizes(), "attn_cache must match q_cache");
  TORCH_CHECK(
      lse_cache.size(0) == R && lse_cache.size(1) == capacity && lse_cache.size(2) == num_heads,
      "lse_cache shape mismatch");

  TORCH_CHECK(src_q.dim() == 3 && src_q.size(1) == num_heads && src_q.size(2) == head_dim,
              "src_q must be [sumN,H,D]");
  TORCH_CHECK(src_attn.sizes() == src_q.sizes(), "src_attn must match src_q");
  TORCH_CHECK(src_lse.dim() == 2 && src_lse.size(1) == num_heads, "src_lse must be [sumN,H]");
  TORCH_CHECK(src_lse.size(0) == src_q.size(0), "src_lse row count must match src_q");
  TORCH_CHECK(offsets.size(0) == (req_ids.size(0) + 1), "offsets must be [B+1]");
  TORCH_CHECK(lens.size(0) == req_ids.size(0), "lens must be [B]");
  TORCH_CHECK(request_length.size(0) == R || request_length.size(0) == req_ids.size(0),
              "request_length must be [R] or [B]");

  p.q_cache = q_cache.data_ptr();
  if (q_post_cache.defined() && q_post_cache.numel() > 0) {
    CHECK_CUDA(q_post_cache);
    CHECK_CONTIG(q_post_cache);
    CHECK_DTYPE(q_post_cache, torch::kBFloat16);
    TORCH_CHECK(q_post_cache.sizes() == q_cache.sizes(), "q_post_cache must match q_cache");
    p.q_post_cache = q_post_cache.data_ptr();
  } else {
    p.q_post_cache = nullptr;
  }
  p.attn_cache = attn_cache.data_ptr();
  p.lse_cache = lse_cache.data_ptr();
  p.q_src = src_q.data_ptr();
  p.attn_src = src_attn.data_ptr();
  p.lse_src = src_lse.data_ptr();
  p.R = R;
  p.M = capacity;
  p.H = num_heads;
  p.D = head_dim;
  p.src_rows = (int32_t)src_q.size(0);
  p.req_ids = req_ids.data_ptr<int32_t>();
  p.B = (int)req_ids.size(0);
  p.offsets = offsets.data_ptr<int32_t>();
  p.lens = lens.data_ptr<int32_t>();
  p.request_length = request_length.data_ptr<int32_t>();
  p.request_length_by_batch = (request_length.size(0) == req_ids.size(0));
}

static inline int next_pow2_clamped(int v) {
  int x = 1;
  while (x < v && x < 256) x <<= 1;
  if (x < 32) x = 32;
  if (x > 256) x = 256;
  return x;
}

// ------------------------------- kernels ----------------------------------

__global__ void mac_prefill_update_cache_kernel(UpdateParams p) {
  const int by = blockIdx.y;
  if (by >= p.B) return;

  const int req = p.req_ids[by];
  const int32_t Ni = p.lens[by];
  if (Ni <= 0 || p.M <= 0) return;
  if (req < 0 || req >= p.R) return;

  const int start = (Ni > p.M) ? (Ni - p.M) : 0;
  const int keep = Ni - start;
  const int32_t base_off = p.offsets[by];
  if (base_off < 0 || base_off + Ni > p.src_rows) return;

  int32_t L_before =
      p.request_length_by_batch ? p.request_length[by] : p.request_length[req];
  if (L_before < 0) L_before = 0;

  const size_t row_bf16 = (size_t)p.H * (size_t)p.D * sizeof(__nv_bfloat16);
  const size_t row_f32 = (size_t)p.H * sizeof(float);

  uint8_t* __restrict__ q_cache_r = (uint8_t*)p.q_cache + ((size_t)req * p.M) * row_bf16;
  uint8_t* __restrict__ q_post_cache_r =
      p.q_post_cache ? ((uint8_t*)p.q_post_cache + ((size_t)req * p.M) * row_bf16) : nullptr;
  uint8_t* __restrict__ attn_cache_r = (uint8_t*)p.attn_cache + ((size_t)req * p.M) * row_bf16;
  uint8_t* __restrict__ lse_cache_r = (uint8_t*)p.lse_cache + ((size_t)req * p.M) * row_f32;

  const uint8_t* __restrict__ q_src = (const uint8_t*)p.q_src;
  const uint8_t* __restrict__ a_src = (const uint8_t*)p.attn_src;
  const uint8_t* __restrict__ l_src = (const uint8_t*)p.lse_src;

  const int base_dest = (int)((L_before + start) % p.M);

  for (int bx = blockIdx.x; bx < keep; bx += gridDim.x) {
    int dest_slot = base_dest + bx;
    if (dest_slot >= p.M) dest_slot -= p.M;

    const size_t dst_q = (size_t)dest_slot * row_bf16;
    const size_t dst_a = (size_t)dest_slot * row_bf16;
    const size_t dst_l = (size_t)dest_slot * row_f32;

    const size_t src_row = (size_t)(base_off + start + bx);
    const size_t src_q_off = src_row * row_bf16;
    const size_t src_a_off = src_row * row_bf16;
    const size_t src_l_off = src_row * row_f32;

    copy_bytes_16(q_cache_r + dst_q, q_src + src_q_off, row_bf16);
    copy_bytes_16(attn_cache_r + dst_a, a_src + src_a_off, row_bf16);
    copy_bytes_16(lse_cache_r + dst_l, l_src + src_l_off, row_f32);

    // Front-sink downdate needs the POST-RoPE matched query, which prefill does
    // not have here. Write a NaN sentinel so the decode kernel skips front-sink
    // for slots still owned by prefill (falls back to fixed-band). Decode
    // overwrites this slot with the real post-RoPE query once it regenerates it.
    if (q_post_cache_r != nullptr) {
      __nv_bfloat16* __restrict__ qpc = (__nv_bfloat16*)(q_post_cache_r + dst_q);
      const int n_elem = p.H * p.D;
      // bf16 quiet-NaN bit pattern (exp all ones, nonzero mantissa).
      const __nv_bfloat16 nan_bf16 = __ushort_as_bfloat16((unsigned short)0x7fc0);
      for (int i = threadIdx.x; i < n_elem; i += blockDim.x) {
        qpc[i] = nan_bf16;
      }
    }
  }

  if (!p.request_length_by_batch && blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(&p.request_length[req], Ni);
  }
}

__global__ void mac_mark_persistent_ready_kernel(bool* __restrict__ ready,
                                                int64_t* __restrict__ cache_epoch,
                                                const int64_t* __restrict__ request_epoch,
                                                const int32_t* __restrict__ req_ids,
                                                int32_t B,
                                                int32_t R) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B) return;
  int req = req_ids[i];
  if (req < 0 || req >= R) return;
  cache_epoch[req] = request_epoch[req];
  ready[req] = true;
}

__global__ void mac_invalidate_persistent_ready_kernel(bool* __restrict__ ready,
                                                       int64_t* __restrict__ cache_epoch,
                                                       int64_t* __restrict__ request_epoch,
                                                       const int32_t* __restrict__ req_ids,
                                                       int32_t B,
                                                       int32_t R) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B) return;
  int req = req_ids[i];
  if (req < 0 || req >= R) return;
  ready[req] = false;
  cache_epoch[req] = -1;
  request_epoch[req] += 1;
}

__global__ void mac_update_request_state_kernel(int32_t* __restrict__ req_len,
                                                int32_t* __restrict__ match_req_len,
                                                int32_t* __restrict__ cache_lens,
                                                const int32_t* __restrict__ req_ids,
                                                const int32_t* __restrict__ seq_lens,
                                                const int32_t* __restrict__ cur_lens,
                                                int32_t B,
                                                int32_t R) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B) return;
  int32_t seq_len = seq_lens[i];
  int32_t cur_len = cur_lens[i];
  int32_t cache_len = seq_len - cur_len;
  if (cache_len < 0) cache_len = 0;
  cache_lens[i] = cache_len;

  int req = req_ids[i];
  if (req < 0 || req >= R) return;
  req_len[req] = seq_len;
  match_req_len[req] = cache_len;
}

// ------------------------------- launcher ---------------------------------

void mac_prefill_update_cache(Tensor q_cache,
                              Tensor attn_cache,
                              Tensor lse_cache,
                              Tensor request_length,
                              Tensor req_ids,
                              Tensor src_q,
                              Tensor src_attn,
                              Tensor src_lse,
                              Tensor offsets,
                              Tensor lens,
                              int capacity,
                              int num_heads,
                              int head_dim,
                              Tensor q_post_cache) {
  if (capacity <= 0 || req_ids.size(0) == 0) return;

  UpdateParams p;
  fill_params(p,
              q_cache,
              attn_cache,
              lse_cache,
              request_length,
              req_ids,
              src_q,
              src_attn,
              src_lse,
              offsets,
              lens,
              capacity,
              num_heads,
              head_dim,
              q_post_cache);

  c10::cuda::CUDAGuard dev_guard(q_cache.device());
  c10::cuda::CUDAStreamGuard stream_guard(at::cuda::getCurrentCUDAStream(q_cache.get_device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q_cache.get_device()).stream();

  const size_t row_bf16 = (size_t)num_heads * (size_t)head_dim * sizeof(__nv_bfloat16);
  const size_t row_f32 = (size_t)num_heads * sizeof(float);
  const int row16_q = (int)((row_bf16 + 15) / 16);
  const int row16_l = (int)((row_f32 + 15) / 16);
  const int row16 = (row16_q > row16_l) ? row16_q : row16_l;
  const int threads = next_pow2_clamped(row16);

  const dim3 grid(1, (unsigned)p.B);

  mac_prefill_update_cache_kernel<<<grid, threads, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mac_mark_persistent_ready(Tensor ready,
                               Tensor cache_epoch,
                               Tensor request_epoch,
                               Tensor req_ids) {
  if (req_ids.size(0) == 0) return;
  CHECK_CUDA(ready);
  CHECK_CONTIG(ready);
  CHECK_DTYPE(ready, torch::kBool);
  CHECK_CUDA(cache_epoch);
  CHECK_CONTIG(cache_epoch);
  CHECK_DTYPE(cache_epoch, torch::kInt64);
  CHECK_CUDA(request_epoch);
  CHECK_CONTIG(request_epoch);
  CHECK_DTYPE(request_epoch, torch::kInt64);
  CHECK_CUDA(req_ids);
  CHECK_CONTIG(req_ids);
  CHECK_DTYPE(req_ids, torch::kInt32);
  TORCH_CHECK(cache_epoch.size(0) == ready.size(0), "cache_epoch must match ready");
  TORCH_CHECK(request_epoch.size(0) == ready.size(0), "request_epoch must match ready");

  c10::cuda::CUDAGuard dev_guard(ready.device());
  c10::cuda::CUDAStreamGuard stream_guard(at::cuda::getCurrentCUDAStream(ready.get_device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(ready.get_device()).stream();
  int32_t B = (int32_t)req_ids.size(0);
  int32_t R = (int32_t)ready.size(0);
  int threads = 128;
  int blocks = (B + threads - 1) / threads;
  mac_mark_persistent_ready_kernel<<<blocks, threads, 0, stream>>>(
      ready.data_ptr<bool>(),
      cache_epoch.data_ptr<int64_t>(),
      request_epoch.data_ptr<int64_t>(),
      req_ids.data_ptr<int32_t>(),
      B,
      R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mac_invalidate_persistent_ready(Tensor ready,
                                     Tensor cache_epoch,
                                     Tensor request_epoch,
                                     Tensor req_ids) {
  if (req_ids.size(0) == 0) return;
  CHECK_CUDA(ready);
  CHECK_CONTIG(ready);
  CHECK_DTYPE(ready, torch::kBool);
  CHECK_CUDA(cache_epoch);
  CHECK_CONTIG(cache_epoch);
  CHECK_DTYPE(cache_epoch, torch::kInt64);
  CHECK_CUDA(request_epoch);
  CHECK_CONTIG(request_epoch);
  CHECK_DTYPE(request_epoch, torch::kInt64);
  CHECK_CUDA(req_ids);
  CHECK_CONTIG(req_ids);
  CHECK_DTYPE(req_ids, torch::kInt32);
  TORCH_CHECK(cache_epoch.size(0) == ready.size(0), "cache_epoch must match ready");
  TORCH_CHECK(request_epoch.size(0) == ready.size(0), "request_epoch must match ready");

  c10::cuda::CUDAGuard dev_guard(ready.device());
  c10::cuda::CUDAStreamGuard stream_guard(at::cuda::getCurrentCUDAStream(ready.get_device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(ready.get_device()).stream();
  int32_t B = (int32_t)req_ids.size(0);
  int32_t R = (int32_t)ready.size(0);
  int threads = 128;
  int blocks = (B + threads - 1) / threads;
  mac_invalidate_persistent_ready_kernel<<<blocks, threads, 0, stream>>>(
      ready.data_ptr<bool>(),
      cache_epoch.data_ptr<int64_t>(),
      request_epoch.data_ptr<int64_t>(),
      req_ids.data_ptr<int32_t>(),
      B,
      R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mac_update_request_state(Tensor req_len,
                              Tensor match_req_len,
                              Tensor cache_lens,
                              Tensor req_ids,
                              Tensor seq_lens,
                              Tensor cur_lens) {
  if (req_ids.size(0) == 0) return;
  CHECK_CUDA(req_len);
  CHECK_CONTIG(req_len);
  CHECK_DTYPE(req_len, torch::kInt32);
  CHECK_CUDA(match_req_len);
  CHECK_CONTIG(match_req_len);
  CHECK_DTYPE(match_req_len, torch::kInt32);
  CHECK_CUDA(cache_lens);
  CHECK_CONTIG(cache_lens);
  CHECK_DTYPE(cache_lens, torch::kInt32);
  CHECK_CUDA(req_ids);
  CHECK_CONTIG(req_ids);
  CHECK_DTYPE(req_ids, torch::kInt32);
  CHECK_CUDA(seq_lens);
  CHECK_CONTIG(seq_lens);
  CHECK_DTYPE(seq_lens, torch::kInt32);
  CHECK_CUDA(cur_lens);
  CHECK_CONTIG(cur_lens);
  CHECK_DTYPE(cur_lens, torch::kInt32);
  TORCH_CHECK(match_req_len.size(0) == req_len.size(0), "match_req_len must match req_len");
  TORCH_CHECK(seq_lens.size(0) >= req_ids.size(0), "seq_lens is shorter than req_ids");
  TORCH_CHECK(cur_lens.size(0) >= req_ids.size(0), "cur_lens is shorter than req_ids");
  TORCH_CHECK(cache_lens.size(0) >= req_ids.size(0), "cache_lens is shorter than req_ids");

  c10::cuda::CUDAGuard dev_guard(req_len.device());
  c10::cuda::CUDAStreamGuard stream_guard(at::cuda::getCurrentCUDAStream(req_len.get_device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(req_len.get_device()).stream();
  int32_t B = (int32_t)req_ids.size(0);
  int32_t R = (int32_t)req_len.size(0);
  int threads = 128;
  int blocks = (B + threads - 1) / threads;
  mac_update_request_state_kernel<<<blocks, threads, 0, stream>>>(
      req_len.data_ptr<int32_t>(),
      match_req_len.data_ptr<int32_t>(),
      cache_lens.data_ptr<int32_t>(),
      req_ids.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      cur_lens.data_ptr<int32_t>(),
      B,
      R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// --------------------------------- pybind ---------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mac_prefill_update_cache",
        &mac_prefill_update_cache,
        "Cooperative MAC ring cache update (row-parallel, in-kernel length bump)");
  m.def("mac_mark_persistent_ready",
        &mac_mark_persistent_ready,
        "Guarded MAC persistent-cache ready marker");
  m.def("mac_invalidate_persistent_ready",
        &mac_invalidate_persistent_ready,
        "Guarded MAC persistent-cache invalidation");
  m.def("mac_update_request_state",
        &mac_update_request_state,
        "Guarded MAC request length/cache length metadata update");
}
