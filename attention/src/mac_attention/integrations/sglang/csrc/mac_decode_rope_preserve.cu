#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

using torch::Tensor;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).dtype() == (dt), #x " has wrong dtype")

namespace {

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float v) {
  return __float2bfloat16_rn(v);
}

template <typename PosT>
__device__ __forceinline__ int64_t load_pos(const void* ptr, int64_t n) {
  return static_cast<int64_t>(reinterpret_cast<const PosT*>(ptr)[n]);
}

template <typename CacheLocT>
__device__ __forceinline__ int64_t load_cache_loc(const void* ptr, int64_t n) {
  return static_cast<int64_t>(reinterpret_cast<const CacheLocT*>(ptr)[n]);
}

template <typename PosT, typename CacheLocT>
__global__ void decode_rope_preserve_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ q_pre,
    __nv_bfloat16* __restrict__ k_buffer,
    __nv_bfloat16* __restrict__ v_buffer,
    const float* __restrict__ cos_sin_cache,
    const void* __restrict__ positions,
    const void* __restrict__ cache_locs,
    int64_t N,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t rotary_dim,
    int64_t q_stride0,
    int64_t q_stride1,
    int64_t k_stride0,
    int64_t k_stride1,
    int64_t v_stride0,
    int64_t v_stride1,
    int64_t kv_pool_tokens,
    int64_t cos_sin_stride0,
    bool is_neox) {
  const int64_t half = rotary_dim >> 1;
  const int64_t q_pair_total = N * Hq * half;
  const int64_t k_pair_total = N * Hkv * half;
  const int64_t q_pass_total = N * Hq * (D - rotary_dim);
  const int64_t k_pass_total = N * Hkv * (D - rotary_dim);
  const int64_t v_total = N * Hkv * D;
  const int64_t total = q_pair_total + k_pair_total + q_pass_total + k_pass_total + v_total;

  for (int64_t task = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total;
       task += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t t = task;
    if (t < q_pair_total) {
      int64_t pair = t % half;
      int64_t tmp = t / half;
      int64_t h = tmp % Hq;
      int64_t n = tmp / Hq;
      int64_t pos = load_pos<PosT>(positions, n);
      const float* cs = cos_sin_cache + pos * cos_sin_stride0;
      int64_t d0 = is_neox ? pair : (pair << 1);
      int64_t d1 = is_neox ? (pair + half) : (d0 + 1);
      int64_t q_base = n * q_stride0 + h * D * q_stride1;
      int64_t q0 = q_base + d0 * q_stride1;
      int64_t q1 = q_base + d1 * q_stride1;
      __nv_bfloat16 x0_b = q[q0];
      __nv_bfloat16 x1_b = q[q1];
      int64_t out_base = (n * Hq + h) * D;
      q_pre[out_base + d0] = x0_b;
      q_pre[out_base + d1] = x1_b;
      float x0 = bf16_to_float(x0_b);
      float x1 = bf16_to_float(x1_b);
      float c = cs[pair];
      float s = cs[half + pair];
      q[q0] = float_to_bf16(x0 * c - x1 * s);
      q[q1] = float_to_bf16(x1 * c + x0 * s);
      continue;
    }
    t -= q_pair_total;
    if (t < k_pair_total) {
      int64_t pair = t % half;
      int64_t tmp = t / half;
      int64_t h = tmp % Hkv;
      int64_t n = tmp / Hkv;
      int64_t pos = load_pos<PosT>(positions, n);
      int64_t cache_loc = load_cache_loc<CacheLocT>(cache_locs, n);
      const float* cs = cos_sin_cache + pos * cos_sin_stride0;
      int64_t d0 = is_neox ? pair : (pair << 1);
      int64_t d1 = is_neox ? (pair + half) : (d0 + 1);
      int64_t k_base = n * k_stride0 + h * D * k_stride1;
      int64_t k0 = k_base + d0 * k_stride1;
      int64_t k1 = k_base + d1 * k_stride1;
      __nv_bfloat16 x0_b = k[k0];
      __nv_bfloat16 x1_b = k[k1];
      float x0 = bf16_to_float(x0_b);
      float x1 = bf16_to_float(x1_b);
      float c = cs[pair];
      float s = cs[half + pair];
      __nv_bfloat16 y0 = float_to_bf16(x0 * c - x1 * s);
      __nv_bfloat16 y1 = float_to_bf16(x1 * c + x0 * s);
      k[k0] = y0;
      k[k1] = y1;
      int64_t pool_base = (cache_loc * Hkv + h) * D;
      k_buffer[pool_base + d0] = y0;
      k_buffer[pool_base + d1] = y1;
      continue;
    }
    t -= k_pair_total;
    if (t < q_pass_total) {
      int64_t d = rotary_dim + (t % (D - rotary_dim));
      int64_t tmp = t / (D - rotary_dim);
      int64_t h = tmp % Hq;
      int64_t n = tmp / Hq;
      int64_t q_off = n * q_stride0 + (h * D + d) * q_stride1;
      q_pre[(n * Hq + h) * D + d] = q[q_off];
      continue;
    }
    t -= q_pass_total;
    if (t < k_pass_total) {
      int64_t d = rotary_dim + (t % (D - rotary_dim));
      int64_t tmp = t / (D - rotary_dim);
      int64_t h = tmp % Hkv;
      int64_t n = tmp / Hkv;
      int64_t cache_loc = load_cache_loc<CacheLocT>(cache_locs, n);
      int64_t k_off = n * k_stride0 + (h * D + d) * k_stride1;
      k_buffer[(cache_loc * Hkv + h) * D + d] = k[k_off];
      continue;
    }
    t -= k_pass_total;
    int64_t d = t % D;
    int64_t tmp = t / D;
    int64_t h = tmp % Hkv;
    int64_t n = tmp / Hkv;
    int64_t cache_loc = load_cache_loc<CacheLocT>(cache_locs, n);
    int64_t v_off = n * v_stride0 + (h * D + d) * v_stride1;
    v_buffer[(cache_loc * Hkv + h) * D + d] = v[v_off];
  }
}

void check_bf16_2d(const Tensor& t, const char* name) {
  CHECK_CUDA(t);
  CHECK_DTYPE(t, at::kBFloat16);
  TORCH_CHECK(t.dim() == 2, name, " must have rank 2");
  TORCH_CHECK(t.stride(1) == 1, name, " must have contiguous last dimension");
}

void check_bf16_3d_contig(const Tensor& t, const char* name) {
  CHECK_CUDA(t);
  CHECK_DTYPE(t, at::kBFloat16);
  TORCH_CHECK(t.dim() == 3, name, " must have rank 3");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void mac_decode_rope_preserve_q_bf16(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor q_pre,
    Tensor k_buffer,
    Tensor v_buffer,
    Tensor cos_sin_cache,
    Tensor positions,
    Tensor cache_locs,
    int64_t head_dim,
    bool is_neox) {
  check_bf16_2d(q, "q");
  check_bf16_2d(k, "k");
  check_bf16_2d(v, "v");
  check_bf16_3d_contig(q_pre, "q_pre");
  check_bf16_3d_contig(k_buffer, "k_buffer");
  check_bf16_3d_contig(v_buffer, "v_buffer");
  CHECK_CUDA(cos_sin_cache);
  CHECK_DTYPE(cos_sin_cache, at::kFloat);
  TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must have rank 2");
  CHECK_CUDA(positions);
  CHECK_CUDA(cache_locs);
  TORCH_CHECK(positions.dim() == 1, "positions must have rank 1");
  TORCH_CHECK(cache_locs.dim() == 1, "cache_locs must have rank 1");
  TORCH_CHECK(
      positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong,
      "positions must be int32 or int64");
  TORCH_CHECK(
      cache_locs.scalar_type() == at::kInt || cache_locs.scalar_type() == at::kLong,
      "cache_locs must be int32 or int64");

  int64_t N = q.size(0);
  TORCH_CHECK(k.size(0) == N && v.size(0) == N, "q/k/v batch mismatch");
  TORCH_CHECK(positions.size(0) >= N && cache_locs.size(0) >= N, "metadata batch too small");
  TORCH_CHECK(head_dim > 0 && q.size(1) % head_dim == 0 && k.size(1) % head_dim == 0,
              "invalid head_dim");
  int64_t Hq = q.size(1) / head_dim;
  int64_t Hkv = k.size(1) / head_dim;
  TORCH_CHECK(v.size(1) == Hkv * head_dim, "v shape mismatch");
  TORCH_CHECK(q_pre.size(0) >= N && q_pre.size(1) == Hq && q_pre.size(2) == head_dim,
              "q_pre shape mismatch");
  TORCH_CHECK(k_buffer.size(1) == Hkv && k_buffer.size(2) == head_dim,
              "k_buffer shape mismatch");
  TORCH_CHECK(v_buffer.sizes() == k_buffer.sizes(), "v_buffer shape mismatch");
  int64_t rotary_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rotary_dim > 0 && rotary_dim <= head_dim && (rotary_dim % 2) == 0,
              "invalid rotary_dim");

  c10::cuda::CUDAGuard guard(q.device());
  int threads = 256;
  int64_t half = rotary_dim >> 1;
  int64_t total =
      N * Hq * half +
      N * Hkv * half +
      N * Hq * (head_dim - rotary_dim) +
      N * Hkv * (head_dim - rotary_dim) +
      N * Hkv * head_dim;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  int device = q.get_device();
  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  blocks = blocks < sm_count * 4 ? blocks : sm_count * 4;
  if (blocks < 1) blocks = 1;

  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  __nv_bfloat16* q_ptr = reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>());
  __nv_bfloat16* k_ptr = reinterpret_cast<__nv_bfloat16*>(k.data_ptr<at::BFloat16>());
  const __nv_bfloat16* v_ptr = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>());
  __nv_bfloat16* q_pre_ptr = reinterpret_cast<__nv_bfloat16*>(q_pre.data_ptr<at::BFloat16>());
  __nv_bfloat16* k_buf_ptr = reinterpret_cast<__nv_bfloat16*>(k_buffer.data_ptr<at::BFloat16>());
  __nv_bfloat16* v_buf_ptr = reinterpret_cast<__nv_bfloat16*>(v_buffer.data_ptr<at::BFloat16>());
  const float* cs_ptr = cos_sin_cache.data_ptr<float>();
  const void* pos_ptr = positions.data_ptr();
  const void* loc_ptr = cache_locs.data_ptr();
  bool pos_i64 = positions.scalar_type() == at::kLong;
  bool loc_i64 = cache_locs.scalar_type() == at::kLong;

#define LAUNCH(PosT, LocT)                                                                 \
  decode_rope_preserve_kernel<PosT, LocT><<<blocks, threads, 0, stream>>>(                \
      q_ptr, k_ptr, v_ptr, q_pre_ptr, k_buf_ptr, v_buf_ptr, cs_ptr, pos_ptr, loc_ptr, N,   \
      Hq, Hkv, head_dim, rotary_dim, q.stride(0), q.stride(1), k.stride(0), k.stride(1),   \
      v.stride(0), v.stride(1), k_buffer.size(0), cos_sin_cache.stride(0), is_neox)

  if (pos_i64 && loc_i64) {
    LAUNCH(int64_t, int64_t);
  } else if (pos_i64 && !loc_i64) {
    LAUNCH(int64_t, int32_t);
  } else if (!pos_i64 && loc_i64) {
    LAUNCH(int32_t, int64_t);
  } else {
    LAUNCH(int32_t, int32_t);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "mac_decode_rope_preserve_q_bf16",
      &mac_decode_rope_preserve_q_bf16,
      "Decode-only fused pre-RoPE Q preserve + RoPE + KV cache write (BF16)");
}
