#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <torch/extension.h>

#include <cstdint>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <limits>

using torch::Tensor;
namespace cg = cooperative_groups;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).dtype() == (dt), #x " has wrong dtype")

namespace {

constexpr int kHeadDim = 128;
constexpr int kMaxWarps = 8;
constexpr int kMaxStageTokens = 16;
constexpr int kMaxFuseTailTilesInMerge = 8;
constexpr int kRectReduceChunk = 32;
constexpr int kFullFallbackReduceChunk = 64;
constexpr int kDenseFullFallbackReduceChunk = 64;
constexpr int kDenseFullFallbackReduceChunkFp32 = 64;
constexpr int kDenseFullFallbackReduceChunkMultiRequest = 8;
constexpr int kFullFallbackReduceChunkMax = 256;
constexpr float kInf = INFINITY;
constexpr float kLog2e = 1.4426950408889634074f;
constexpr float kLn2 = 0.6931471805599453094f;
constexpr int kGroupModeMacHit = 0;
constexpr int kGroupModeFullFallback = 1;
constexpr int kGroupModeMixedFallback = 2;
constexpr int kDirectZ2Group = 4;
constexpr int kDirectZ2ZParts = 2;
constexpr int kDirectZ2StageTokens = kDirectZ2Group * kDirectZ2ZParts;
constexpr int kDirectZ2SmemBytes =
    2 * 2 * kDirectZ2StageTokens * kHeadDim * static_cast<int>(sizeof(__nv_bfloat16)) +
    kDirectZ2Group * kDirectZ2ZParts * kHeadDim * static_cast<int>(sizeof(float)) +
    kDirectZ2Group * kDirectZ2ZParts * static_cast<int>(sizeof(float));

struct Params {
  int32_t N;
  int32_t Hq;
  int32_t Hkv;
  int32_t D;
  int32_t group_size;
  int32_t M;
  int32_t R;
  int32_t req_to_token_stride;
  int32_t max_context;
  int32_t max_match_tiles;
  int32_t max_tiles_context;
  int32_t max_tiles_reduce;
  int32_t max_tiles_tail;
  int32_t tile_tokens;
  int32_t stage_tokens;
  int32_t match_tile_slots;
  int32_t match_early_exit;
  int32_t semantic_pos_ahead;
  int32_t gen_min_limit;
  int32_t lookback_right;
  int32_t candidate_mode;
  int32_t debug_enabled;
  int32_t phase_cycles_count;
  int32_t out_cache_loc_is_i64;
  int32_t group_rect_max_spread;
  int32_t fuse_hit_tail_in_merge;
  int32_t fuse_fallback_tail_in_merge;
  int32_t fuse_mixed_fallback_tail_in_merge;
  int32_t mixed_group_fallback;
  int32_t hit_tail_group;
  int32_t all_hit_direct;
  int32_t hit_complete_head_direct;
  int32_t full_fallback_group_direct;
  int32_t full_fallback_head_direct;
  int32_t full_fallback_group_merge;
  int32_t full_fallback_head_reduce;
  int32_t full_fallback_warp_reduce;
  int32_t mixed_head_reduce;
  int32_t mixed_early_miss_direct;
  int32_t partial_o_bf16;
  int32_t long_rect_tile_tokens;
  int32_t long_rect_min_tokens;
  int32_t full_fallback_min_chunk_tokens;
  int32_t full_fallback_target_ctas;
  int32_t full_fallback_dense_target_ctas;
  int32_t full_fallback_per_mode_tiles;
  int32_t full_fallback_producer_coarsen;
  int32_t mixed_fallback_heavy_target_ctas;
  int32_t mixed_head_direct_target_ctas;
  int32_t mixed_head_broad_target_ctas;
  int32_t mixed_head_dense_full_target_ctas;
  int32_t full_fallback_partial_reduce_chunk;
  int32_t full_fallback_multi_request_reduce_chunk;
  int32_t mixed_reduce_chunk;
  int32_t mixed_no_full_reduce_chunk;
  int32_t full_fallback_reduce_threshold;
  int32_t full_fallback_dense_mixed_no_reduce;
  int32_t full_fallback_wide_mixed_no_reduce;
  int32_t mixed_group_direct_z2;
  int32_t tail_group_direct_z2;
  int32_t parallel_z2_schedule;
  int32_t mixed_misspack_z2;
  int32_t sparse_fallback_unfuse_tail;
  int32_t mixed_group_direct_min_miss_heads;
  int32_t bench_mode;
  int32_t bench_exact_quota;
  int32_t bench_seed;
  int32_t bench_layer_id;
  int32_t bench_miss_mask;
  float threshold;
  float threshold_distance;
  float sm_scale;
  float bench_hit_rate;
  float bench_hit_rate_std;
  float bench_skip_ratio;
  float bench_skip_ratio_std;
  float bench_match_lag_mean;
  float bench_match_lag_std;
};

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float v) {
  return __float2bfloat16_rn(v);
}

__device__ __forceinline__ void store_partial_o(void* __restrict__ ptr,
                                                int64_t idx,
                                                float value,
                                                int partial_o_bf16) {
  if (partial_o_bf16) {
    reinterpret_cast<__nv_bfloat16*>(ptr)[idx] = float_to_bf16(value);
  } else {
    reinterpret_cast<float*>(ptr)[idx] = value;
  }
}

__device__ __forceinline__ float load_partial_o(const void* __restrict__ ptr,
                                                int64_t idx,
                                                int partial_o_bf16) {
  if (partial_o_bf16) {
    return bf16_to_float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
  }
  return reinterpret_cast<const float*>(ptr)[idx];
}

struct alignas(16) Bf16x8 {
  __nv_bfloat16 v[8];
};

__device__ __forceinline__ Bf16x8 load_bf16x8(const __nv_bfloat16* __restrict__ ptr) {
  return *reinterpret_cast<const Bf16x8*>(ptr);
}

__device__ __forceinline__ void store_bf16x8(__nv_bfloat16* __restrict__ ptr,
                                             const Bf16x8& value) {
  *reinterpret_cast<Bf16x8*>(ptr) = value;
}

__device__ __forceinline__ void cp_async_cg_16(void* __restrict__ dst_shared,
                                               const void* __restrict__ src_global) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(dst_shared));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
               "l"(src_global), "n"(16), "r"(16));
#else
  *reinterpret_cast<Bf16x8*>(dst_shared) = *reinterpret_cast<const Bf16x8*>(src_global);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#else
  (void)N;
#endif
}

__device__ __forceinline__ void z2_stage_half_sync(int tz) {
#if defined(__CUDA_ARCH__)
  if (tz == 0) {
    asm volatile("bar.sync 1, 64;\n" ::: "memory");
  } else {
    asm volatile("bar.sync 2, 64;\n" ::: "memory");
  }
#else
  (void)tz;
#endif
}

__device__ __forceinline__ float warp_sum(float v) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset);
  }
  return v;
}

__device__ __forceinline__ float subwarp16_sum(float v, unsigned mask) {
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset, 16);
  }
  return v;
}

__device__ __forceinline__ float subwarp16_xor_sum(float v, unsigned mask) {
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(mask, v, offset, 16);
  }
  return v;
}

__device__ __forceinline__ float ptx_exp2_approx(float x) {
#if defined(__CUDA_ARCH__)
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return exp2f(x);
#endif
}

__device__ __forceinline__ float ptx_rcp_approx(float x) {
#if defined(__CUDA_ARCH__)
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return 1.0f / x;
#endif
}

__device__ __forceinline__ int ceil_div_i32(int a, int b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ int bench_target_miss_units(const Params& p, int unit_total) {
  if (unit_total <= 0) return 0;
  float miss_rate = 1.0f - p.bench_hit_rate;
  miss_rate = fminf(fmaxf(miss_rate, 0.0f), 1.0f);
  int miss_units;
  if (p.bench_exact_quota != 0 && p.bench_hit_rate_std == 0.0f) {
    miss_units = static_cast<int>(floorf(miss_rate * static_cast<float>(unit_total) + 0.5f));
  } else {
    miss_units = static_cast<int>(ceilf(miss_rate * static_cast<float>(unit_total)));
  }
  if (miss_units < 0) miss_units = 0;
  if (miss_units > unit_total) miss_units = unit_total;
  return miss_units;
}

__device__ __forceinline__ bool full_fallback_warp_reduce_enabled(const Params& p) {
  return p.full_fallback_warp_reduce != 0 && p.full_fallback_group_direct != 0 &&
         p.group_size == 4 && p.D == kHeadDim && blockDim.x >= 128;
}

__device__ __forceinline__ bool full_fallback_head_reduce_enabled(const Params& p) {
  return p.full_fallback_head_reduce != 0 && full_fallback_warp_reduce_enabled(p) &&
         p.group_size == 4 && p.D == kHeadDim && blockDim.x == 128;
}

__device__ __forceinline__ bool mixed_head_reduce_enabled(const Params& p) {
  return p.mixed_head_reduce != 0 && p.group_size == 4 && p.D == kHeadDim &&
         blockDim.x == 128;
}

__device__ __forceinline__ bool mixed_group_direct_z2_enabled(const Params& p) {
  return p.mixed_group_direct_z2 != 0 && p.full_fallback_group_direct != 0 &&
         p.group_size == 4 && p.D == kHeadDim && blockDim.x == 128;
}

__device__ __forceinline__ bool mixed_misspack_z2_enabled(const Params& p) {
  return p.mixed_misspack_z2 != 0 && mixed_group_direct_z2_enabled(p);
}

__device__ __forceinline__ bool tail_group_direct_z2_enabled(const Params& p) {
  return p.tail_group_direct_z2 != 0 && p.group_size == 4 && p.D == kHeadDim &&
         blockDim.x == 128;
}

__device__ __forceinline__ bool complete_group_direct_z2_selected(
    const Params& p, int mode, int miss_heads) {
  if (mode == kGroupModeFullFallback) {
    return p.full_fallback_group_direct != 0 && p.group_size == 4 && p.D == kHeadDim &&
           blockDim.x == 128;
  }
  return mode == kGroupModeMixedFallback && p.mixed_early_miss_direct != 0 &&
         mixed_group_direct_z2_enabled(p) && p.mixed_group_direct_min_miss_heads > 0 &&
         miss_heads >= p.mixed_group_direct_min_miss_heads;
}

__device__ __forceinline__ bool tail_group_direct_z2_active(
    const Params& p, const int32_t* __restrict__ task_counts) {
  if (!tail_group_direct_z2_enabled(p) || task_counts == nullptr) return false;
  return task_counts[1] > 0;
}

__device__ __forceinline__ int rect_reduce_chunk_for_mode(const Params& p, int mode) {
  if (mode == kGroupModeFullFallback && full_fallback_warp_reduce_enabled(p)) {
    return kFullFallbackReduceChunk;
  }
  return kRectReduceChunk;
}

__device__ __forceinline__ int balanced_rect_reduce_chunk_for_mode(
    const Params& p, int mode, const int32_t* __restrict__ task_counts) {
  int chunk = rect_reduce_chunk_for_mode(p, mode);
  if (mode == kGroupModeFullFallback && task_counts != nullptr && task_counts[5] > 0) {
    int total_groups = p.N * p.Hkv;
    int full_fallback_groups = task_counts[6];
    if (full_fallback_groups > 0 && full_fallback_groups * 4 > total_groups) {
      int partial_chunk = p.full_fallback_partial_reduce_chunk;
      if (full_fallback_groups < total_groups && p.N >= 2 &&
          p.full_fallback_multi_request_reduce_chunk > 0) {
        partial_chunk = p.full_fallback_multi_request_reduce_chunk;
      }
      if (partial_chunk > 0) {
        chunk = partial_chunk;
      }
    }
  }
  if (mode == kGroupModeMixedFallback && p.mixed_reduce_chunk > 0) {
    chunk = p.mixed_reduce_chunk;
    if (task_counts != nullptr && task_counts[6] == 0 &&
        p.mixed_no_full_reduce_chunk > 0) {
      chunk = p.mixed_no_full_reduce_chunk;
    }
  }
  return chunk;
}

__device__ __forceinline__ int rect_reduce_threshold_for_mode(const Params& p, int mode) {
  if (mode == kGroupModeFullFallback && full_fallback_warp_reduce_enabled(p)) {
    return p.full_fallback_reduce_threshold;
  }
  return kRectReduceChunk;
}

__device__ __forceinline__ int clamp_reduce_chunk_to_workspace(
    int chunk, int orig_tiles, int max_tiles_reduce) {
  if (max_tiles_reduce > 0 && orig_tiles > 0) {
    int min_chunk = ceil_div_i32(orig_tiles, max_tiles_reduce);
    if (chunk < min_chunk) chunk = min_chunk;
  }
  return chunk;
}

__device__ __forceinline__ int max_rect_reduce_chunks_for_mode(
    const Params& p, int mode, const int32_t* __restrict__ task_counts) {
  if (mode == kGroupModeFullFallback && task_counts != nullptr && task_counts[7] > 0) {
    return task_counts[7];
  }
  if (mode == kGroupModeMixedFallback && task_counts != nullptr && task_counts[8] > 0) {
    return task_counts[8];
  }
  int chunk = clamp_reduce_chunk_to_workspace(
      balanced_rect_reduce_chunk_for_mode(p, mode, task_counts), p.max_tiles_context,
      p.max_tiles_reduce);
  return ceil_div_i32(p.max_tiles_context, chunk);
}

__device__ __forceinline__ int rect_tile_tokens_for(const Params& p, int mode, int rect_tokens) {
  if (mode != kGroupModeMacHit && p.long_rect_tile_tokens > p.tile_tokens &&
      rect_tokens >= p.long_rect_min_tokens) {
    return p.long_rect_tile_tokens;
  }
  return p.tile_tokens;
}

__device__ __forceinline__ int dense_full_fallback_target_ctas_for(
    const Params& p, const int32_t* __restrict__ task_counts) {
  int base_target = p.full_fallback_target_ctas;
  if (base_target <= 0) base_target = static_cast<int>(gridDim.x);
  if (base_target < 1) base_target = 1;
  int dense_target = p.full_fallback_dense_target_ctas;
  if (dense_target <= 0) dense_target = base_target;
  if (dense_target < base_target) dense_target = base_target;

  int total_groups = p.N * p.Hkv;
  int full_fallback_groups = task_counts != nullptr ? task_counts[6] : total_groups;
  if (total_groups <= 0 || full_fallback_groups <= 0 || dense_target == base_target) {
    return base_target;
  }

  long long dense_excess =
      static_cast<long long>(full_fallback_groups) * 8LL -
      static_cast<long long>(total_groups) * 5LL;
  if (dense_excess <= 0) {
    return base_target;
  }
  long long ramp_span = static_cast<long long>(total_groups);
  if (ramp_span < 1LL) ramp_span = 1LL;
  if (dense_excess > ramp_span) dense_excess = ramp_span;
  long long delta = static_cast<long long>(dense_target) - static_cast<long long>(base_target);
  long long ramp =
      (delta * dense_excess + ramp_span - 1LL) / ramp_span;
  return base_target + static_cast<int>(ramp);
}

__device__ __forceinline__ int full_fallback_tile_tokens_for_mixed_schedule(
    const Params& p, const int32_t* __restrict__ task_counts, int rect_tokens) {
  if (p.full_fallback_per_mode_tiles == 0 || task_counts == nullptr || rect_tokens <= 0) {
    return 0;
  }
  int total_groups = p.N * p.Hkv;
  int full_fallback_groups = task_counts[6];
  if (full_fallback_groups <= 0 || full_fallback_groups >= total_groups) {
    return 0;
  }

  int target_ctas = dense_full_fallback_target_ctas_for(p, task_counts);
  if (full_fallback_groups * 2 <= total_groups) {
    int sparse_target = p.full_fallback_target_ctas;
    if (sparse_target <= 0) sparse_target = static_cast<int>(gridDim.x);
    if (sparse_target < 1) sparse_target = 1;
    sparse_target += max(1, sparse_target / 4);
    if (target_ctas < sparse_target) {
      target_ctas = sparse_target;
    }
  }
  long long raw =
      (static_cast<long long>(rect_tokens) * static_cast<long long>(full_fallback_groups) +
       static_cast<long long>(target_ctas) - 1LL) /
      static_cast<long long>(target_ctas);
  int min_chunk = p.full_fallback_min_chunk_tokens;
  if (min_chunk < p.tile_tokens) min_chunk = p.tile_tokens;
  if (raw < min_chunk) raw = min_chunk;
  if (raw > rect_tokens) raw = rect_tokens;
  int chosen = static_cast<int>(raw);
  if (chosen < p.tile_tokens) chosen = p.tile_tokens;
  return chosen;
}

__device__ __forceinline__ bool all_hit_direct_active(
    const Params& p, const int32_t* __restrict__ task_counts) {
  return p.all_hit_direct != 0 && task_counts != nullptr && task_counts[5] == 0;
}

__device__ __forceinline__ bool all_groups_hit(
    const int32_t* __restrict__ task_counts) {
  return task_counts != nullptr && task_counts[5] == 0;
}

__device__ __forceinline__ bool sparse_fallback_unfuse_tail_active(
    const Params& p, const int32_t* __restrict__ task_counts) {
  if (p.sparse_fallback_unfuse_tail == 0 || task_counts == nullptr ||
      p.max_tiles_tail > kMaxFuseTailTilesInMerge) {
    return false;
  }
  int total_groups = p.N * p.Hkv;
  int fallback_groups = task_counts[6];
  return fallback_groups > 0 && fallback_groups < total_groups &&
         fallback_groups * 2 <= total_groups;
}

__device__ __forceinline__ bool mixed_fallback_tail_fuse_active(
    const Params& p,
    int tail_tiles) {
  return p.fuse_mixed_fallback_tail_in_merge != 0 &&
         p.fuse_hit_tail_in_merge != 0 &&
         tail_tiles > 0 && tail_tiles <= kMaxFuseTailTilesInMerge;
}

__device__ __forceinline__ bool full_fallback_dense_mixed_no_reduce_active(
    const Params& p, const int32_t* __restrict__ task_counts) {
  if (p.full_fallback_dense_mixed_no_reduce == 0 || task_counts == nullptr) return false;
  int total_groups = p.N * p.Hkv;
  int full_fallback_groups = task_counts[6];
  if (full_fallback_groups <= 0 || full_fallback_groups >= total_groups) return false;
  if (full_fallback_groups * 4 >= total_groups * 3) return true;
  return p.full_fallback_wide_mixed_no_reduce != 0 && total_groups >= 32 &&
         full_fallback_groups * 2 >= total_groups;
}

__device__ __forceinline__ int rect_reduce_threshold_for_counts(
    const Params& p, int mode, const int32_t* __restrict__ task_counts) {
  if (mode == kGroupModeFullFallback &&
      full_fallback_dense_mixed_no_reduce_active(p, task_counts)) {
    return INT_MAX;
  }
  return rect_reduce_threshold_for_mode(p, mode);
}

__device__ __forceinline__ int balanced_rect_tile_tokens_for(
    const Params& p,
    const int32_t* __restrict__ task_counts,
    int mode,
    int rect_tokens) {
  if (task_counts != nullptr) {
    if (mode == kGroupModeFullFallback) {
      int full_fallback_tile_tokens =
          full_fallback_tile_tokens_for_mixed_schedule(p, task_counts, rect_tokens);
      if (full_fallback_tile_tokens > 0) {
        return full_fallback_tile_tokens;
      }
    }
    int balanced = task_counts[4];
    if (balanced > 0 && !all_groups_hit(task_counts)) {
      return balanced;
    }
  }
  return rect_tile_tokens_for(p, mode, rect_tokens);
}

__device__ __forceinline__ int full_fallback_producer_coarsen_for(
    const Params& p,
    const int32_t* __restrict__ task_counts,
    int rect_tokens) {
  if (p.full_fallback_producer_coarsen <= 1 || task_counts == nullptr ||
      rect_tokens <= 0 || p.full_fallback_group_direct == 0 || p.group_size != 4 ||
      p.D != kHeadDim) {
    return 1;
  }
  int total_groups = p.N * p.Hkv;
  int full_fallback_groups = task_counts[6];
  if (total_groups <= 0 || full_fallback_groups <= 0 ||
      full_fallback_groups >= total_groups) {
    return 1;
  }
  // Coarsen only when full-KV groups are dense inside a mixed schedule. In
  // that regime the full-KV producers create many partial states and the later
  // reduce/merge phases dominate. Pure all-full schedules keep the existing
  // high-exposure split-K shape.
  if (full_fallback_groups * 4 <= total_groups * 3) {
    return 1;
  }
  int factor = p.full_fallback_producer_coarsen;
  if (factor < 1) factor = 1;
  if (factor > 8) factor = 8;
  return factor;
}

__device__ __forceinline__ int producer_rect_tile_tokens_for(
    const Params& p,
    const int32_t* __restrict__ task_counts,
    int mode,
    int rect_tokens) {
  int tile_tokens = balanced_rect_tile_tokens_for(p, task_counts, mode, rect_tokens);
  if (mode == kGroupModeFullFallback) {
    int factor = full_fallback_producer_coarsen_for(p, task_counts, rect_tokens);
    if (factor > 1 && tile_tokens > 0 && rect_tokens > tile_tokens) {
      long long coarsened =
          static_cast<long long>(tile_tokens) * static_cast<long long>(factor);
      if (coarsened > rect_tokens) coarsened = rect_tokens;
      tile_tokens = static_cast<int>(coarsened);
    }
  }
  return tile_tokens;
}

__device__ __forceinline__ int cache_end_for_pos(int p, int S) {
  int end = p + 1 - S;
  return end > 0 ? end : 0;
}

__device__ __forceinline__ int ring_slot_to_pos(int past_len, int M, int slot) {
  if (past_len < M) {
    return slot;
  }
  int base = past_len - M;
  int tail = past_len % M;
  int order_idx = slot - tail;
  if (order_idx < 0) order_idx += M;
  return base + order_idx;
}

__device__ __forceinline__ bool better_pair(float dist, int pos, float best_dist, int best_pos) {
  return (dist < best_dist) || (dist == best_dist && pos < best_pos);
}

__device__ __forceinline__ float clamp_float(float v, float lo, float hi) {
  return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ uint32_t mix_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ float uniform01_u32(uint32_t x) {
  return static_cast<float>(mix_u32(x) & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float pseudo_normal_u32(uint32_t x) {
  float s = uniform01_u32(x) + uniform01_u32(x ^ 0x9e3779b9u) +
            uniform01_u32(x ^ 0x85ebca6bu) + uniform01_u32(x ^ 0xc2b2ae35u);
  // Sum of four U(0, 1) samples has stddev sqrt(1/3); scale to approximately N(0, 1).
  return (s - 2.0f) * 1.7320508075688772f;
}

__device__ __forceinline__ int pos_to_ring_slot(int past_len, int M, int pos) {
  if (past_len < M) {
    return pos;
  }
  int base = past_len - M;
  int tail = past_len % M;
  int order_idx = pos - base;
  int slot = order_idx + tail;
  if (slot >= M) slot -= M;
  if (slot < 0) slot += M;
  return slot;
}

__device__ __forceinline__ int physical_token(
    int n,
    int req,
    int logical_pos,
    int past_len,
    const int32_t* __restrict__ req_to_token,
    int stride,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    int out_cache_loc_is_i64) {
  if (logical_pos == past_len) {
    return out_cache_loc_is_i64 ? static_cast<int>(out_cache_loc_i64[n]) : out_cache_loc_i32[n];
  }
  return req_to_token[static_cast<int64_t>(req) * stride + logical_pos];
}

__device__ __forceinline__ void merge_scalar(float& state_lse, bool& state_valid,
                                             float other_lse, float& w_state,
                                             float& w_other, bool& take_other) {
  bool other_valid = isfinite(other_lse);
  take_other = false;
  w_state = 1.0f;
  w_other = 0.0f;
  if (!other_valid) {
    return;
  }
  if (!state_valid) {
    state_lse = other_lse;
    state_valid = true;
    take_other = true;
    return;
  }
  float m = fmaxf(state_lse, other_lse);
  float es = expf(state_lse - m);
  float eo = expf(other_lse - m);
  float denom = es + eo;
  float out_lse = m + logf(denom);
  w_state = es / denom;
  w_other = eo / denom;
  state_lse = out_lse;
}

__device__ void phase1_match_scan(
    const Params& p,
    const __nv_bfloat16* __restrict__ q_pre,
    const __nv_bfloat16* __restrict__ query_cache,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    float* __restrict__ match_dist,
    int32_t* __restrict__ match_pos,
    int32_t* __restrict__ match_slot) {
  const int total = p.N * p.Hq * p.max_match_tiles;
  const int warps = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  for (int task = blockIdx.x * warps + warp; task < total; task += gridDim.x * warps) {
    int t = task;
    int tile = t % p.max_match_tiles;
    t /= p.max_match_tiles;
    int hq = t % p.Hq;
    int n = t / p.Hq;

    int past_len = past_lens[n];
    int candidate_begin = max(0, past_len - p.M);
    int candidate_end = past_len;
    if (p.candidate_mode == 1) {
      candidate_end = max(candidate_begin, past_len - p.lookback_right);
    }
    bool eligible = past_len >= p.gen_min_limit && candidate_end > candidate_begin;

    float best_d = kInf;
    int best_p = INT_MAX;
    int best_s = -1;
    int slot_begin = tile * p.match_tile_slots;
    int slot_end = min(p.M, slot_begin + p.match_tile_slots);
    int req = req_ids[n];
    float q_vals[4];
    int dims[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = lane + i * 32;
      dims[i] = d;
      q_vals[i] = 0.0f;
      if (d < p.D) {
        int64_t q_off = (static_cast<int64_t>(n) * p.Hq + hq) * p.D + d;
        q_vals[i] = bf16_to_float(q_pre[q_off]);
      }
    }

    for (int slot = slot_begin; slot < slot_end; ++slot) {
      int logical_pos = ring_slot_to_pos(past_len, p.M, slot);
      bool valid = eligible && logical_pos >= candidate_begin && logical_pos < candidate_end;
      float acc = 0.0f;
      if (valid) {
        int64_t c_off = (((static_cast<int64_t>(req) * p.M + slot) * p.Hq + hq) * p.D);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int d = dims[i];
          if (d >= p.D) continue;
          float a = q_vals[i];
          float b = bf16_to_float(query_cache[c_off + d]);
          float diff = a - b;
          acc = fmaf(diff, diff, acc);
          if (p.match_early_exit != 0) {
            float partial = warp_sum(acc);
            int over_threshold =
                __shfl_sync(0xffffffffu, partial > p.threshold_distance ? 1 : 0, 0);
            if (over_threshold != 0) {
              break;
            }
          }
        }
      }
      acc = warp_sum(acc);
      if (lane == 0 && valid && better_pair(acc, logical_pos, best_d, best_p)) {
        best_d = acc;
        best_p = logical_pos;
        best_s = slot;
      }
    }

    if (lane == 0) {
      int64_t off = (static_cast<int64_t>(n) * p.Hq + hq) * p.max_match_tiles + tile;
      match_dist[off] = best_d;
      match_pos[off] = best_p;
      match_slot[off] = best_s;
    }
  }
  __syncthreads();
}

__device__ void phase2_reduce_schedule(
    const Params& p,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ req_ids,
    const float* __restrict__ lse_cache,
    const float* __restrict__ match_dist,
    const int32_t* __restrict__ match_pos,
    const int32_t* __restrict__ match_slot,
    int32_t* __restrict__ head_hit,
    int32_t* __restrict__ head_match_slot,
    int32_t* __restrict__ head_match_pos,
    int32_t* __restrict__ head_prefix_end,
    int32_t* __restrict__ head_rect_start,
    int32_t* __restrict__ head_new_end,
    int32_t* __restrict__ group_rect_begin,
    int32_t* __restrict__ group_rect_end,
    int32_t* __restrict__ group_rect_tiles,
    int32_t* __restrict__ group_tail_begin,
    int32_t* __restrict__ group_tail_end,
    int32_t* __restrict__ group_tail_tiles,
    int32_t* __restrict__ group_mode,
    int32_t* __restrict__ complete_task_go,
    int32_t* __restrict__ complete_task_tile,
    int32_t* __restrict__ tail_task_go,
    int32_t* __restrict__ tail_task_tile,
    int32_t* __restrict__ hit_tail_task_ho,
    int32_t* __restrict__ hit_tail_task_tile,
    int32_t* __restrict__ task_counts,
    cg::grid_group grid) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    task_counts[0] = 0;  // complete work rows
    task_counts[1] = 0;  // group tail rows
    task_counts[2] = 0;  // hit tail head rows
    task_counts[3] = 0;  // fallback groups requiring rect reduction
    task_counts[4] = 0;  // load-balanced rect chunk size
    task_counts[5] = 0;  // non-hit group count; zero means true all-hit fast path
    task_counts[6] = 0;  // full-fallback group count; all groups means pure FI-like fallback
    task_counts[7] = 0;  // max reduced chunks among full-fallback groups
    task_counts[8] = 0;  // max reduced chunks among mixed-fallback groups
  }

  const int head_total = p.N * p.Hq;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  const unsigned mask = 0xffffffffu;
  for (int task = blockIdx.x * warps + warp; task < head_total;
       task += gridDim.x * warps) {
    int hq = task % p.Hq;
    int n = task / p.Hq;
    float best_d = kInf;
    int best_p = INT_MAX;
    int best_s = -1;
    for (int tile = lane; tile < p.max_match_tiles; tile += 32) {
      int64_t off = (static_cast<int64_t>(n) * p.Hq + hq) * p.max_match_tiles + tile;
      float d = match_dist[off];
      int pos = match_pos[off];
      int slot = match_slot[off];
      if (better_pair(d, pos, best_d, best_p)) {
        best_d = d;
        best_p = pos;
        best_s = slot;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other_d = __shfl_down_sync(mask, best_d, offset);
      int other_p = __shfl_down_sync(mask, best_p, offset);
      int other_s = __shfl_down_sync(mask, best_s, offset);
      if (better_pair(other_d, other_p, best_d, best_p)) {
        best_d = other_d;
        best_p = other_p;
        best_s = other_s;
      }
    }
    if (lane == 0) {
      int past_len = past_lens[n];
      int candidate_begin = max(0, past_len - p.M);
      int candidate_end = (p.candidate_mode == 1)
                              ? max(candidate_begin, past_len - p.lookback_right)
                              : past_len;
      bool eligible = past_len >= p.gen_min_limit && candidate_end > candidate_begin;
      bool hit = eligible && best_s >= 0 && best_d < p.threshold_distance;
      if (p.bench_mode != 0 || p.bench_miss_mask >= 0) {
        // bench_mode 1: independent per-query-head hit decisions.
        // bench_mode 2: one shared hit/match decision per GQA KV group. This
        // matches the scheduler's fast-path unit: a GQA group only gets the
        // all-hit MAC path when every query head in the group is valid.
        int hkv = hq / p.group_size;
        int lane_i = hq - hkv * p.group_size;
        uint32_t bench_head_or_group =
            (p.bench_mode == 2) ? static_cast<uint32_t>(hkv) : static_cast<uint32_t>(hq);
        uint32_t base_key = static_cast<uint32_t>(p.bench_seed) ^
                            (static_cast<uint32_t>(p.bench_layer_id) * 0x9e3779b9u) ^
                            (static_cast<uint32_t>(past_len) * 0x85ebca6bu);
        uint32_t key = base_key ^ (static_cast<uint32_t>(n) * 0xc2b2ae35u) ^
                       (bench_head_or_group * 0x27d4eb2fu);
        float hit_rate = clamp_float(
            p.bench_hit_rate + p.bench_hit_rate_std * pseudo_normal_u32(key ^ 0x4cf5ad43u),
            0.0f, 1.0f);
        if (p.bench_miss_mask >= 0) {
          bool synthetic_miss = lane_i >= 0 && lane_i < 30 &&
                                (((p.bench_miss_mask >> lane_i) & 1) != 0);
          hit = eligible && past_len > 0 && p.M > 0 && !synthetic_miss;
        } else if (p.bench_exact_quota != 0 && p.bench_hit_rate_std == 0.0f) {
          // Single-request synthetic-group benchmarking has very few samples
          // (typically Hkv=8). Independent Bernoulli sampling makes hit=0.9
          // randomly become 6/8 or 8/8 groups, which swamps the kernel tuning
          // signal. Use the closest representable exact quota by default and
          // rotate the miss set across token/layer/seed for deterministic
          // coverage without per-request variance.
          int unit_total = (p.bench_mode == 2) ? (p.N * p.Hkv) : (p.N * p.Hq);
          int unit = (p.bench_mode == 2) ? (n * p.Hkv + hkv) : (n * p.Hq + hq);
          int miss_units = bench_target_miss_units(p, unit_total);
          int offset = unit_total > 0 ? static_cast<int>(base_key % static_cast<uint32_t>(unit_total)) : 0;
          int rotated = unit + offset;
          if (rotated >= unit_total) rotated -= unit_total;
          bool synthetic_miss = rotated < miss_units;
          hit = eligible && past_len > 0 && p.M > 0 && !synthetic_miss;
        } else {
          hit = eligible && past_len > 0 && p.M > 0 &&
                uniform01_u32(key ^ 0x165667b1u) < hit_rate;
        }
        if (hit) {
          int max_lag = min(p.M, past_len);
          int new_end_tmp = cache_end_for_pos(past_len, p.semantic_pos_ahead);
          float lag_f = p.bench_match_lag_mean +
                        p.bench_match_lag_std * pseudo_normal_u32(key ^ 0xd3a2646cu);
          bool requested_skip_ratio = false;
          if (p.bench_skip_ratio >= 0.0f) {
            requested_skip_ratio = true;
            float skip_ratio = clamp_float(
                p.bench_skip_ratio +
                    p.bench_skip_ratio_std * pseudo_normal_u32(key ^ 0xfd7046c5u),
                0.0f, 0.999999f);
            int target_prefix_end = static_cast<int>(roundf(skip_ratio * static_cast<float>(past_len)));
            if (target_prefix_end > new_end_tmp - 1) target_prefix_end = new_end_tmp - 1;
            if (target_prefix_end < 0) target_prefix_end = 0;
            lag_f = static_cast<float>(new_end_tmp - target_prefix_end);
          }
          int lag = static_cast<int>(roundf(lag_f));
          if (lag < 1) lag = 1;
          // For explicit skip-ratio benchmarking, preserve the requested
          // prefix length even when it is outside the configured lookback
          // window. The later group-validity check will turn those heads into
          // fallback if the recompute span exceeds M. This keeps the synthetic
          // skip-ratio knob honest instead of silently clamping it back to the
          // largest reachable skip for the current lookback window.
          if (requested_skip_ratio) {
            if (lag > past_len) lag = past_len;
          } else if (lag > max_lag) {
            lag = max_lag;
          }
          if (max_lag <= 0 || new_end_tmp <= 0) {
            hit = false;
            best_p = INT_MAX;
            best_s = -1;
          } else {
            best_p = past_len - lag;
            if (best_p < 0) best_p = 0;
            best_s = pos_to_ring_slot(past_len, p.M, best_p);
            best_d = 0.0f;
          }
        } else {
          best_p = INT_MAX;
          best_s = -1;
          best_d = kInf;
        }
      }
      int prefix_end = 0;
      int rect_start = 0;
      int mpos = -1;
      int mslot = -1;
      if (hit) {
        mpos = best_p;
        mslot = best_s;
        prefix_end = cache_end_for_pos(mpos, p.semantic_pos_ahead);
        rect_start = prefix_end;
      }
      int new_end = cache_end_for_pos(past_len, p.semantic_pos_ahead);
      int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
      head_hit[ho] = hit ? 1 : 0;
      head_match_slot[ho] = mslot;
      head_match_pos[ho] = mpos;
      head_prefix_end[ho] = prefix_end;
      head_rect_start[ho] = rect_start;
      head_new_end[ho] = new_end;
    }
  }

  grid.sync();

  const int group_total = p.N * p.Hkv;
  for (int task = blockIdx.x * blockDim.x + threadIdx.x; task < group_total;
       task += gridDim.x * blockDim.x) {
    int hkv = task % p.Hkv;
    int n = task / p.Hkv;
    int req = req_ids[n];
    int past_len = past_lens[n];
    int kv_len = past_len + 1;
    int new_end = cache_end_for_pos(past_len, p.semantic_pos_ahead);
    int complete_begin = INT_MAX;
    int valid_count = 0;
    int hq0 = hkv * p.group_size;
    for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
      int hq = hq0 + lane_i;
      int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
      int head_begin = head_rect_start[ho];
      int head_end = head_new_end[ho];
      int slot = head_match_slot[ho];
      bool valid = head_hit[ho] != 0 && slot >= 0 && head_end == new_end;
      int complete_len = head_end - head_begin;
      if (valid) {
        float prefix_lse = lse_cache[(static_cast<int64_t>(req) * p.M + slot) * p.Hq + hq];
        valid = isfinite(prefix_lse) && complete_len >= 0 && complete_len <= p.M;
      }
      if (valid) {
        valid_count += 1;
        complete_begin = min(complete_begin, head_begin);
      }
    }

    int mode = kGroupModeFullFallback;
    if (valid_count == p.group_size) {
      mode = kGroupModeMacHit;
    } else if (valid_count > 0 && p.mixed_group_fallback != 0) {
      mode = kGroupModeMixedFallback;
    }
    int rect_end = new_end;
    int tail_begin = new_end;
    if (mode == kGroupModeFullFallback) {
      complete_begin = 0;
      for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
        int hq = hq0 + lane_i;
        int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
        head_hit[ho] = 0;
        head_match_slot[ho] = -1;
        head_match_pos[ho] = -1;
        head_prefix_end[ho] = 0;
        head_rect_start[ho] = 0;
        head_new_end[ho] = rect_end;
      }
    } else if (mode == kGroupModeMixedFallback) {
      complete_begin = 0;
      for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
        int hq = hq0 + lane_i;
        int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
        int head_begin = head_rect_start[ho];
        int head_end = head_new_end[ho];
        int slot = head_match_slot[ho];
        bool valid = head_hit[ho] != 0 && slot >= 0 && head_end == new_end;
        int complete_len = head_end - head_begin;
        if (valid) {
          float prefix_lse = lse_cache[(static_cast<int64_t>(req) * p.M + slot) * p.Hq + hq];
          valid = isfinite(prefix_lse) && complete_len >= 0 && complete_len <= p.M;
        }
        if (valid) continue;
        head_hit[ho] = 0;
        head_match_slot[ho] = -1;
        head_match_pos[ho] = -1;
        head_prefix_end[ho] = 0;
        head_rect_start[ho] = 0;
        head_new_end[ho] = new_end;
      }
    }

    complete_begin = max(0, min(complete_begin, rect_end));
    int rect_tokens = rect_end - complete_begin;
    int tail_tiles = (kv_len > tail_begin) ? ceil_div_i32(kv_len - tail_begin, p.tile_tokens) : 0;
    int64_t go = static_cast<int64_t>(n) * p.Hkv + hkv;
    group_rect_begin[go] = complete_begin;
    group_rect_end[go] = rect_end;
    group_rect_tiles[go] = rect_tokens;  // temporary until load-balanced chunk size is chosen
    group_tail_begin[go] = tail_begin;
    group_tail_end[go] = kv_len;
    group_tail_tiles[go] = tail_tiles;
    group_mode[go] = mode;
    if (mode != kGroupModeMacHit) {
      atomicAdd(&task_counts[5], 1);
    }
    if (mode == kGroupModeFullFallback) {
      atomicAdd(&task_counts[6], 1);
    }
  }

  grid.sync();

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    bool all_hit = task_counts[5] == 0;
    int chosen = p.tile_tokens;
    if (!all_hit) {
      int min_chunk = p.full_fallback_min_chunk_tokens;
      if (min_chunk < p.tile_tokens) min_chunk = p.tile_tokens;
      int max_tokens = 0;
      long long weighted_tokens = 0;
      bool all_full_fallback = true;
      for (int go_i = 0; go_i < group_total; ++go_i) {
        int mode = group_mode[go_i];
        if (mode != kGroupModeFullFallback) all_full_fallback = false;
        int tokens = group_rect_end[go_i] - group_rect_begin[go_i];
        if (tokens > max_tokens) max_tokens = tokens;
        if (tokens <= 0) continue;
        int mult = 1;
        if (mode == kGroupModeMacHit && p.hit_complete_head_direct != 0) {
          mult = p.group_size;
        } else if (mode == kGroupModeFullFallback &&
                   !(p.full_fallback_group_direct != 0 && p.group_size == 4) &&
                   p.full_fallback_head_direct != 0) {
          mult = p.group_size;
        } else if (mode == kGroupModeMixedFallback && p.mixed_early_miss_direct != 0) {
          mult = p.group_size;
        }
        weighted_tokens += static_cast<long long>(tokens) * static_cast<long long>(mult);
      }
      // The old MAC load-balancer was global: it tried to expose enough CTA
      // work across hit and miss groups together. Keep the larger full-
      // fallback chunk floor only for true all-full-fallback. In any mixed
      // schedule a single missed long group would otherwise emit too few tiles
      // (e.g. hit=0.9), leaving many persistent CTAs idle.
      if (!all_full_fallback) {
        min_chunk = p.tile_tokens;
      }
      if (max_tokens > 0 && max_tokens >= p.long_rect_min_tokens) {
        int mixed_groups = task_counts[5] - task_counts[6];
        int target_ctas = dense_full_fallback_target_ctas_for(p, task_counts);
        if (!all_full_fallback && mixed_groups > 0 && p.mixed_early_miss_direct != 0 &&
            p.mixed_head_direct_target_ctas > 0) {
          // Mixed GQA groups commonly have one long miss head plus several
          // short hit heads. Mixed-only production rows are trajectory
          // sensitive, so keep their existing high-exposure shape. When full
          // fallback groups are also present, they already provide CTA breadth
          // and the mixed producers should avoid overproducing partial states
          // for the reducer.
          target_ctas = p.mixed_head_direct_target_ctas;
          if (task_counts[6] > 0 && p.mixed_head_broad_target_ctas > 0) {
            target_ctas = p.mixed_head_broad_target_ctas;
          }
          if (task_counts[6] > 0 && p.mixed_head_dense_full_target_ctas > 0 &&
              task_counts[6] * 4 >= group_total * 3) {
            // Dense mixed schedules need more producer exposure as they
            // approach all-full, but the 75% boundary should not immediately
            // jump to the maximum target and overproduce partial states.
            int base_target = target_ctas;
            int dense_target = p.mixed_head_dense_full_target_ctas;
            if (dense_target < base_target) dense_target = base_target;
            long long dense_excess =
                static_cast<long long>(task_counts[6]) * 4LL -
                static_cast<long long>(group_total) * 3LL;
            if (dense_excess > 0 && group_total > 0) {
              long long delta =
                  static_cast<long long>(dense_target) - static_cast<long long>(base_target);
              long long ramp =
                  (delta * dense_excess + static_cast<long long>(group_total) - 1LL) /
                  static_cast<long long>(group_total);
              target_ctas = base_target + static_cast<int>(ramp);
            } else {
              target_ctas = base_target;
            }
          }
        } else if (!all_full_fallback && p.mixed_fallback_heavy_target_ctas > 0 &&
                   task_counts[5] * 2 >= group_total) {
          target_ctas = p.mixed_fallback_heavy_target_ctas;
        }
        if (target_ctas <= 0) target_ctas = gridDim.x;
        if (target_ctas < 1) target_ctas = 1;
        long long raw = (weighted_tokens + static_cast<long long>(target_ctas) - 1LL) /
                        static_cast<long long>(target_ctas);
        if (raw < min_chunk) raw = min_chunk;
        if (raw > max_tokens) raw = max_tokens;
        chosen = static_cast<int>(raw);
        if (chosen < p.tile_tokens) chosen = p.tile_tokens;
      }
    }
    task_counts[4] = chosen;
  }

  grid.sync();

  if (p.parallel_z2_schedule != 0 && task_counts[5] > 0 && task_counts[6] == 0 &&
      mixed_group_direct_z2_enabled(p)) {
    __shared__ int sh_z2_base;
    for (int go_i = blockIdx.x; go_i < group_total; go_i += gridDim.x) {
      int64_t go = static_cast<int64_t>(go_i);
      int mode = group_mode[go];
      int n = go_i / p.Hkv;
      int hkv = go_i - n * p.Hkv;
      int hq0 = hkv * p.group_size;
      int miss_heads = 0;
      if (mode == kGroupModeMixedFallback) {
#pragma unroll
        for (int lane_i = 0; lane_i < 4; ++lane_i) {
          if (lane_i >= p.group_size) continue;
          int hq = hq0 + lane_i;
          int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
          miss_heads += head_hit[ho] == 0 ? 1 : 0;
        }
      } else if (mode == kGroupModeFullFallback) {
        miss_heads = p.group_size;
      }

      int rect_tokens = group_rect_end[go] - group_rect_begin[go];
      int rect_tile_tokens = producer_rect_tile_tokens_for(p, task_counts, mode, rect_tokens);
      int rect_tiles = rect_tokens > 0 ? ceil_div_i32(rect_tokens, rect_tile_tokens) : 0;
      bool z2_selected = rect_tiles > 0 && complete_group_direct_z2_selected(p, mode, miss_heads);
      if (z2_selected) {
        if (threadIdx.x == 0) {
          group_rect_tiles[go] = rect_tiles;
          sh_z2_base = atomicAdd(&task_counts[0], rect_tiles);
          if (rect_tiles > rect_reduce_threshold_for_counts(p, mode, task_counts)) {
            atomicAdd(&task_counts[3], 1);
            int reduce_chunk = clamp_reduce_chunk_to_workspace(
                balanced_rect_reduce_chunk_for_mode(p, mode, task_counts), rect_tiles,
                p.max_tiles_reduce);
            int reduced_tiles = ceil_div_i32(rect_tiles, reduce_chunk);
            if (mode == kGroupModeFullFallback) {
              atomicMax(&task_counts[7], reduced_tiles);
            } else {
              atomicMax(&task_counts[8], reduced_tiles);
            }
          }
        }
        __syncthreads();
        int32_t task_go_value = static_cast<int32_t>(go);
        if (mode == kGroupModeMixedFallback) {
          if (mixed_misspack_z2_enabled(p) && miss_heads > 0 &&
              miss_heads * 2 <= p.group_size) {
            task_go_value = static_cast<int32_t>(go + group_total * 2);
          } else {
            task_go_value = static_cast<int32_t>(go + group_total);
          }
        }
        for (int tile = threadIdx.x; tile < rect_tiles; tile += blockDim.x) {
          int idx = sh_z2_base + tile;
          complete_task_go[idx] = task_go_value;
          complete_task_tile[idx] = tile;
        }
        __syncthreads();
      }
    }
    grid.sync();
  }

  for (int task = blockIdx.x * blockDim.x + threadIdx.x; task < group_total;
       task += gridDim.x * blockDim.x) {
    int hkv = task % p.Hkv;
    int n = task / p.Hkv;
    int req = req_ids[n];
    int past_len = past_lens[n];
    int kv_len = past_len + 1;
    int new_end = cache_end_for_pos(past_len, p.semantic_pos_ahead);
    int hq0 = hkv * p.group_size;
    int valid_count = 0;
    int valid_complete_begin = INT_MAX;
    bool valid_heads[kMaxWarps];
    for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
      int hq = hq0 + lane_i;
      int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
      int head_begin = head_rect_start[ho];
      int head_end = head_new_end[ho];
      int slot = head_match_slot[ho];
      bool valid = head_hit[ho] != 0 && slot >= 0 && head_end == new_end;
      int complete_len = head_end - head_begin;
      if (valid) {
        float prefix_lse = lse_cache[(static_cast<int64_t>(req) * p.M + slot) * p.Hq + hq];
        valid = isfinite(prefix_lse) && complete_len >= 0 && complete_len <= p.M;
      }
      valid_heads[lane_i] = valid;
      if (valid) {
        valid_count += 1;
        valid_complete_begin = min(valid_complete_begin, head_begin);
      }
    }

    int64_t go = static_cast<int64_t>(n) * p.Hkv + hkv;
    int mode = group_mode[go];
    int complete_begin = group_rect_begin[go];
    int rect_end = group_rect_end[go];
    int rect_tokens = rect_end - complete_begin;
    int rect_tile_tokens = producer_rect_tile_tokens_for(p, task_counts, mode, rect_tokens);
    int rect_tiles = rect_tokens > 0 ? ceil_div_i32(rect_tokens, rect_tile_tokens) : 0;
    int tail_begin = group_tail_begin[go];
    int tail_tiles = group_tail_tiles[go];
    group_rect_tiles[go] = rect_tiles;
    int miss_heads = p.group_size - valid_count;
    bool z2_complete_emitted_parallel =
        p.parallel_z2_schedule != 0 && task_counts[6] == 0 &&
        complete_group_direct_z2_selected(p, mode, miss_heads);

    if (rect_tiles > 0 &&
        !(mode == kGroupModeMacHit && all_hit_direct_active(p, task_counts)) &&
        !z2_complete_emitted_parallel) {
      if (mode == kGroupModeMacHit && p.hit_complete_head_direct != 0) {
        int base = atomicAdd(&task_counts[0], p.group_size * rect_tiles);
        for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
          int hq = hq0 + lane_i;
          int32_t ho = static_cast<int32_t>(n * p.Hq + hq);
          for (int tile = 0; tile < rect_tiles; ++tile) {
            int idx = base + lane_i * rect_tiles + tile;
            complete_task_go[idx] = -ho - 1;
            complete_task_tile[idx] = tile;
          }
        }
      } else if (mode == kGroupModeMixedFallback && p.mixed_early_miss_direct != 0) {
        int direct_until = max(0, min(valid_complete_begin, new_end));
        int direct_tiles = direct_until > complete_begin
                               ? (direct_until - complete_begin) / rect_tile_tokens
                               : 0;
        if (direct_tiles > rect_tiles) direct_tiles = rect_tiles;
        bool use_mixed_z2_task =
            complete_group_direct_z2_selected(p, mode, miss_heads);
        bool use_misspack_z2_task =
            use_mixed_z2_task && mixed_misspack_z2_enabled(p) && miss_heads > 0 &&
            miss_heads < p.group_size;
        if (direct_tiles > 0 && miss_heads > 0) {
          if (p.mixed_group_direct_min_miss_heads > 0 &&
              miss_heads >= p.mixed_group_direct_min_miss_heads) {
            int base = atomicAdd(&task_counts[0], direct_tiles);
            int32_t task_go_value = static_cast<int32_t>(go);
            if (use_misspack_z2_task) {
              task_go_value = static_cast<int32_t>(go + group_total * 2);
            } else if (use_mixed_z2_task) {
              task_go_value = static_cast<int32_t>(go + group_total);
            }
            for (int tile = 0; tile < direct_tiles; ++tile) {
              int idx = base + tile;
              complete_task_go[idx] = task_go_value;
              complete_task_tile[idx] = tile;
            }
          } else {
            int base = atomicAdd(&task_counts[0], miss_heads * direct_tiles);
            int out_lane = 0;
            for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
              if (valid_heads[lane_i]) continue;
              int hq = hq0 + lane_i;
              int32_t ho = static_cast<int32_t>(n * p.Hq + hq);
              for (int tile = 0; tile < direct_tiles; ++tile) {
                int idx = base + out_lane * direct_tiles + tile;
                complete_task_go[idx] = -ho - 1;
                complete_task_tile[idx] = tile;
              }
              out_lane += 1;
            }
          }
        }
        int group_tiles_to_emit = rect_tiles - direct_tiles;
        if (group_tiles_to_emit > 0) {
          int base = atomicAdd(&task_counts[0], group_tiles_to_emit);
          int32_t task_go_value = static_cast<int32_t>(go);
          if (use_mixed_z2_task) {
            task_go_value = static_cast<int32_t>(go + group_total);
          }
          for (int tile = direct_tiles; tile < rect_tiles; ++tile) {
            int idx = base + (tile - direct_tiles);
            complete_task_go[idx] = task_go_value;
            complete_task_tile[idx] = tile;
          }
        }
      } else if (mode == kGroupModeFullFallback && p.full_fallback_group_direct != 0 &&
                 p.group_size == 4) {
        int base = atomicAdd(&task_counts[0], rect_tiles);
        for (int tile = 0; tile < rect_tiles; ++tile) {
          complete_task_go[base + tile] = static_cast<int32_t>(go);
          complete_task_tile[base + tile] = tile;
        }
      } else if (mode == kGroupModeFullFallback && p.full_fallback_head_direct != 0) {
        int base = atomicAdd(&task_counts[0], p.group_size * rect_tiles);
        for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
          int hq = hq0 + lane_i;
          int32_t ho = static_cast<int32_t>(n * p.Hq + hq);
          for (int tile = 0; tile < rect_tiles; ++tile) {
            int idx = base + lane_i * rect_tiles + tile;
            complete_task_go[idx] = -ho - 1;
            complete_task_tile[idx] = tile;
          }
        }
      } else {
        int base = atomicAdd(&task_counts[0], rect_tiles);
        for (int tile = 0; tile < rect_tiles; ++tile) {
          complete_task_go[base + tile] = static_cast<int32_t>(go);
          complete_task_tile[base + tile] = tile;
        }
      }
      if (mode != kGroupModeMacHit &&
          rect_tiles > rect_reduce_threshold_for_counts(p, mode, task_counts)) {
        atomicAdd(&task_counts[3], 1);
        if (mode == kGroupModeFullFallback) {
          int reduce_chunk = clamp_reduce_chunk_to_workspace(
              balanced_rect_reduce_chunk_for_mode(p, mode, task_counts), rect_tiles,
              p.max_tiles_reduce);
          int reduced_tiles = ceil_div_i32(rect_tiles, reduce_chunk);
          atomicMax(&task_counts[7], reduced_tiles);
        } else if (mode == kGroupModeMixedFallback) {
          int reduce_chunk = clamp_reduce_chunk_to_workspace(
              balanced_rect_reduce_chunk_for_mode(p, mode, task_counts), rect_tiles,
              p.max_tiles_reduce);
          int reduced_tiles = ceil_div_i32(rect_tiles, reduce_chunk);
          atomicMax(&task_counts[8], reduced_tiles);
        }
      }
    }

    if (tail_tiles > 0) {
      bool unfuse_sparse_tail = sparse_fallback_unfuse_tail_active(p, task_counts);
      if (mode == kGroupModeMacHit) {
        if (!unfuse_sparse_tail && p.fuse_hit_tail_in_merge &&
            tail_tiles <= kMaxFuseTailTilesInMerge) {
          continue;
        }
        if (all_hit_direct_active(p, task_counts)) {
          continue;
        }
        if (p.hit_tail_group != 0) {
          int base = atomicAdd(&task_counts[1], tail_tiles);
          for (int tile = 0; tile < tail_tiles; ++tile) {
            tail_task_go[base + tile] = static_cast<int32_t>(go);
            tail_task_tile[base + tile] = tile;
          }
        } else {
          int base = atomicAdd(&task_counts[2], p.group_size * tail_tiles);
          for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
            int hq = hq0 + lane_i;
            int32_t ho = static_cast<int32_t>(n * p.Hq + hq);
            for (int tile = 0; tile < tail_tiles; ++tile) {
              int idx = base + lane_i * tail_tiles + tile;
              hit_tail_task_ho[idx] = ho;
              hit_tail_task_tile[idx] = tile;
            }
          }
        }
      } else {
        if (mode == kGroupModeFullFallback && !unfuse_sparse_tail &&
            p.fuse_fallback_tail_in_merge &&
            tail_tiles <= kMaxFuseTailTilesInMerge) {
          continue;
        } else if (mode == kGroupModeMixedFallback && !unfuse_sparse_tail &&
                   mixed_fallback_tail_fuse_active(p, tail_tiles)) {
          continue;
        } else {
          int base = atomicAdd(&task_counts[1], tail_tiles);
          for (int tile = 0; tile < tail_tiles; ++tile) {
            tail_task_go[base + tile] = static_cast<int32_t>(go);
            tail_task_tile[base + tile] = tile;
          }
        }
      }
    }
  }
}

template <bool IsTail>
__device__ void phase_attention_tiles_compact(
    const Params& p,
    const __nv_bfloat16* __restrict__ q_post,
    const __nv_bfloat16* __restrict__ k_buffer,
    const __nv_bfloat16* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    const int32_t* __restrict__ head_rect_start,
    const int32_t* __restrict__ head_new_end,
    const int32_t* __restrict__ group_begin,
    const int32_t* __restrict__ group_end,
    const int32_t* __restrict__ group_tiles,
    const int32_t* __restrict__ group_mode,
    const int32_t* __restrict__ task_go,
    const int32_t* __restrict__ task_tile,
    const int32_t* __restrict__ task_counts,
    void* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    int max_tiles,
    bool tail_group_direct_z2_on) {
  const int total = task_counts[IsTail ? 1 : 0];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  __shared__ __nv_bfloat16 sh_k_stage[kMaxStageTokens * kHeadDim];
  __shared__ __nv_bfloat16 sh_v_stage[kMaxStageTokens * kHeadDim];
  __shared__ int sh_phys_stage[kMaxStageTokens];

  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int encoded_task = task_go[task];
    int tile = task_tile[task];

    if constexpr (!IsTail) {
      if (encoded_task < 0 || encoded_task >= p.N * p.Hkv) {
        continue;
      }
    } else {
      if (encoded_task < 0 || encoded_task >= p.N * p.Hkv) {
        continue;
      }
    }

    int64_t go = static_cast<int64_t>(encoded_task);
    if constexpr (IsTail) {
      if (tail_group_direct_z2_on) {
        continue;
      }
    }
    int hkv = static_cast<int>(go % p.Hkv);
    int n = static_cast<int>(go / p.Hkv);
    if constexpr (!IsTail) {
      if (group_mode[go] == kGroupModeFullFallback && p.full_fallback_group_direct != 0 &&
          p.group_size == 4 && p.D == kHeadDim && blockDim.x == 128) {
        continue;
      }
    }
    int tiles = group_tiles[go];
    if (tile >= tiles) continue;

    int group_tokens = group_end[go] - group_begin[go];
    int tile_tokens = p.tile_tokens;
    if constexpr (!IsTail) {
      tile_tokens = producer_rect_tile_tokens_for(p, task_counts, group_mode[go], group_tokens);
    }
    int begin = group_begin[go] + tile * tile_tokens;
    int end = min(begin + tile_tokens, group_end[go]);
    int token_count = end - begin;
    if (token_count <= 0) continue;
    int req = req_ids[n];
    int past_len = past_lens[n];
    int hq = hkv * p.group_size + warp;
    bool tile_overlaps_head = warp < p.group_size;
    if constexpr (!IsTail) {
      tile_overlaps_head = false;
      if (warp < p.group_size) {
        int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
        tile_overlaps_head = end > head_rect_start[ho] && begin < head_new_end[ho];
      }
    }

    float m = -kInf;
    float denom = 1.0f;
    bool has_value = false;
    float q_vals[4];
    float o_vals[4];
    int dims[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = lane + i * 32;
      dims[i] = d;
      o_vals[i] = 0.0f;
      q_vals[i] = 0.0f;
      if (warp < p.group_size && d < p.D) {
        q_vals[i] = bf16_to_float(q_post[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d]);
      }
    }

    for (int stage_begin = begin; stage_begin < end; stage_begin += p.stage_tokens) {
      int stage_count = min(p.stage_tokens, end - stage_begin);
      if (threadIdx.x < stage_count) {
        int pos = stage_begin + threadIdx.x;
        sh_phys_stage[threadIdx.x] = physical_token(
            n, req, pos, past_len, req_to_token, p.req_to_token_stride,
            out_cache_loc_i32, out_cache_loc_i64, p.out_cache_loc_is_i64);
      }
      __syncthreads();

      int stage_elems = stage_count * p.D;
      for (int idx = threadIdx.x; idx < stage_elems; idx += blockDim.x) {
        int s = idx / p.D;
        int d = idx - s * p.D;
        int phys = sh_phys_stage[s];
        int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + d;
        sh_k_stage[idx] = k_buffer[off];
        sh_v_stage[idx] = v_buffer[off];
      }
      __syncthreads();

      for (int s = 0; s < stage_count; ++s) {
        int pos = stage_begin + s;
        bool contributes = warp < p.group_size;
        if constexpr (!IsTail) {
          if (contributes) {
            int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
            contributes = pos >= head_rect_start[ho] && pos < head_new_end[ho];
          }
        }

        if (contributes) {
          const int stage_off = s * p.D;
          float dot = 0.0f;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int d = dims[i];
            if (d < p.D) {
              dot = fmaf(q_vals[i], bf16_to_float(sh_k_stage[stage_off + d]), dot);
            }
          }
          dot = warp_sum(dot);
          float score = __shfl_sync(0xffffffffu, dot, 0) *
                        (IsTail ? (p.sm_scale * kLog2e) : p.sm_scale);
          float new_m = fmaxf(m, score);
          float alpha;
          float beta;
          if constexpr (IsTail) {
            alpha = has_value ? ptx_exp2_approx(m - new_m) : 0.0f;
            beta = ptx_exp2_approx(score - new_m);
          } else {
            alpha = has_value ? expf(m - new_m) : 0.0f;
            beta = expf(score - new_m);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int d = dims[i];
            if (d < p.D) {
              o_vals[i] =
                  o_vals[i] * alpha + beta * bf16_to_float(sh_v_stage[stage_off + d]);
            }
          }
          denom = denom * alpha + beta;
          m = new_m;
          has_value = true;
        }
      }
      __syncthreads();
    }

    if (warp < p.group_size && tile_overlaps_head) {
      int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * max_tiles + tile) *
                         p.group_size + warp);
      float out_lse = -kInf;
      if (denom > 0.0f) {
        if constexpr (IsTail) {
          out_lse = (m + __log2f(denom)) * kLn2;
        } else {
          out_lse = m + logf(denom);
        }
      }
      if (lane == 0) {
        partial_lse[lse_off] = out_lse;
      }
      int64_t o_base = lse_off * p.D;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int d = dims[i];
        if (d < p.D) {
          store_partial_o(partial_o, o_base + d,
                          denom > 0.0f ? (o_vals[i] / denom) : 0.0f,
                          p.partial_o_bf16);
        }
      }
    }
    __syncthreads();
  }
}

__device__ void phase_complete_attention_group_direct_z2(
    const Params& p,
    const __nv_bfloat16* __restrict__ q_post,
    const __nv_bfloat16* __restrict__ k_buffer,
    const __nv_bfloat16* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    const int32_t* __restrict__ group_begin,
    const int32_t* __restrict__ group_end,
    const int32_t* __restrict__ group_tiles,
    const int32_t* __restrict__ group_mode,
    const int32_t* __restrict__ head_hit,
    const int32_t* __restrict__ head_rect_start,
    const int32_t* __restrict__ head_new_end,
    const int32_t* __restrict__ task_go,
    const int32_t* __restrict__ task_tile,
    const int32_t* __restrict__ task_counts,
    void* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    int max_tiles,
    uint8_t* __restrict__ dyn_smem) {
  if (p.full_fallback_group_direct == 0 || p.group_size != 4 || p.D != kHeadDim ||
      blockDim.x != 128 || dyn_smem == nullptr) return;

  constexpr int kGroup = kDirectZ2Group;
  constexpr int kZParts = kDirectZ2ZParts;
  constexpr int kStageTokens = kDirectZ2StageTokens;
  const int tx = threadIdx.x & 15;
  const int local = threadIdx.x >> 4;  // linearized (ty + group * tz)
  const int ty = local & 3;
  const int tz = (local >> 2) & 1;
  constexpr bool active_z = true;
  const unsigned subwarp_mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
  const int dim_base = tx * 8;

  __nv_bfloat16* sh_k = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
  __nv_bfloat16* sh_v = sh_k + 2 * kStageTokens * kHeadDim;
  float* sh_o = reinterpret_cast<float*>(sh_v + 2 * kStageTokens * kHeadDim);
  float* sh_lse = sh_o + kGroup * kZParts * kHeadDim;

  const int total = task_counts[0];
  const int group_total = p.N * p.Hkv;
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int encoded_task = task_go[task];
    if (encoded_task < 0) continue;
    bool encoded_mixed_z2 =
        encoded_task >= group_total && encoded_task < group_total * 2;
    bool encoded_misspack_z2 =
        encoded_task >= group_total * 2 && encoded_task < group_total * 3;
    int64_t go = static_cast<int64_t>(
        encoded_misspack_z2 ? (encoded_task - group_total * 2)
                            : (encoded_mixed_z2 ? (encoded_task - group_total) : encoded_task));
    if (go < 0 || go >= group_total) continue;
    int mode = group_mode[go];
    bool misspack_z2 =
        encoded_misspack_z2 && mode == kGroupModeMixedFallback &&
        mixed_misspack_z2_enabled(p);
    bool mixed_z2 =
        (encoded_mixed_z2 || misspack_z2) && mode == kGroupModeMixedFallback &&
        mixed_group_direct_z2_enabled(p);
    if (mode != kGroupModeFullFallback && !mixed_z2) continue;

    int hkv = static_cast<int>(go % p.Hkv);
    int n = static_cast<int>(go / p.Hkv);
    int tile = task_tile[task];
    int tiles = group_tiles[go];
    if (tile >= tiles) continue;
    bool keep_log2_for_fallback_reduce =
        mode == kGroupModeFullFallback && full_fallback_warp_reduce_enabled(p) &&
        tiles > rect_reduce_threshold_for_counts(p, group_mode[go], task_counts);

    int group_tokens = group_end[go] - group_begin[go];
    int tile_tokens = producer_rect_tile_tokens_for(p, task_counts, group_mode[go], group_tokens);
    int begin = group_begin[go] + tile * tile_tokens;
    int end = min(begin + tile_tokens, group_end[go]);
    if (end <= begin) continue;

    int req = req_ids[n];
    int past_len = past_lens[n];
    int64_t req_token_base = static_cast<int64_t>(req) * p.req_to_token_stride;
    int miss_lanes[kDirectZ2Group];
    int miss_heads = 0;
    int lanes_per_miss_head = 1;
    int miss_slot = ty;
    bool misspack_lane_active = true;
    int out_lane = ty;
    int local_miss_part = 0;
    if (misspack_z2) {
#pragma unroll
      for (int lane_i = 0; lane_i < kDirectZ2Group; ++lane_i) {
        int hq_i = hkv * p.group_size + lane_i;
        int64_t ho_i = static_cast<int64_t>(n) * p.Hq + hq_i;
        if (lane_i < p.group_size && head_hit[ho_i] == 0) {
          miss_lanes[miss_heads++] = lane_i;
        }
      }
      lanes_per_miss_head = miss_heads > 0 ? max(1, kDirectZ2Group / miss_heads) : 1;
      miss_slot = ty / lanes_per_miss_head;
      misspack_lane_active =
          miss_slot < miss_heads && ty < miss_heads * lanes_per_miss_head;
      out_lane = misspack_lane_active ? miss_lanes[miss_slot] : ty;
      local_miss_part = ty - miss_slot * lanes_per_miss_head;
    }
    int hq = hkv * p.group_size + out_lane;
    int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
    int head_begin = mixed_z2 ? head_rect_start[ho] : begin;
    int head_end = mixed_z2 ? head_new_end[ho] : end;
    bool tile_overlaps_head =
        (!mixed_z2 || (end > head_begin && begin < head_end)) && misspack_lane_active;
    const float score_scale = p.sm_scale * kLog2e;

    float q_vals[8];
    float o_vals[8];
    float m = -kInf;
    float denom = 0.0f;
    bool has_value = false;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int d = dim_base + i;
      o_vals[i] = 0.0f;
      q_vals[i] = bf16_to_float(q_post[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d]);
    }

    int stage_begin = begin;
    int stage_count = min(kStageTokens, end - stage_begin);
    int cur_buf = 0;
    int load_slot = tz * kGroup + ty;
    if (active_z && load_slot < stage_count) {
      int load_pos = stage_begin + load_slot;
      int phys = req_to_token[req_token_base + load_pos];
      int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + dim_base;
      int sh_off = cur_buf * kStageTokens * p.D + load_slot * p.D + dim_base;
      cp_async_cg_16(&sh_k[sh_off], &k_buffer[off]);
      cp_async_cg_16(&sh_v[sh_off], &v_buffer[off]);
    }
    cp_async_commit();
    cp_async_wait_all();
    z2_stage_half_sync(tz);

    for (; stage_begin < end; stage_begin += kStageTokens) {
      int next_begin = stage_begin + kStageTokens;
      int next_count = min(kStageTokens, end - next_begin);
      int next_buf = cur_buf ^ 1;
      bool has_next = next_begin < end;
      int load_slot = tz * kGroup + ty;
      if (active_z && has_next && load_slot < next_count) {
        int load_pos = next_begin + load_slot;
        int phys = req_to_token[req_token_base + load_pos];
        int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + dim_base;
        int sh_off = next_buf * kStageTokens * p.D + load_slot * p.D + dim_base;
        cp_async_cg_16(&sh_k[sh_off], &k_buffer[off]);
        cp_async_cg_16(&sh_v[sh_off], &v_buffer[off]);
      }
      cp_async_commit();

      if (!mixed_z2 && stage_count == kStageTokens) {
        float scores[kGroup];
        float betas[kGroup];
        float tile_m = -kInf;
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
          int slot = tz * kGroup + j;
          float dot = 0.0f;
          int sh_off = cur_buf * kStageTokens * p.D + slot * p.D + dim_base;
          Bf16x8 k_vec = load_bf16x8(&sh_k[sh_off]);
          Bf16x8 v_vec = load_bf16x8(&sh_v[sh_off]);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            dot = fmaf(q_vals[i], bf16_to_float(k_vec.v[i]), dot);
          }
          dot = subwarp16_xor_sum(dot, subwarp_mask);
          float score = dot * score_scale;
          scores[j] = score;
          tile_m = fmaxf(tile_m, score);
        }
        float new_m = has_value ? fmaxf(m, tile_m) : tile_m;
        float alpha = has_value ? ptx_exp2_approx(m - new_m) : 0.0f;
        float beta_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
          float beta = ptx_exp2_approx(scores[j] - new_m);
          betas[j] = beta;
          beta_sum += beta;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          o_vals[i] *= alpha;
        }
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
          int slot = tz * kGroup + j;
          int sh_off = cur_buf * kStageTokens * p.D + slot * p.D + dim_base;
          Bf16x8 v_vec = load_bf16x8(&sh_v[sh_off]);
          float beta = betas[j];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            o_vals[i] = fmaf(beta, bf16_to_float(v_vec.v[i]), o_vals[i]);
          }
        }
        denom = denom * alpha + beta_sum;
        m = new_m;
        has_value = true;
      } else if (!mixed_z2) {
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
          int slot = tz * kGroup + j;
          if (slot < stage_count) {
            float dot = 0.0f;
            int sh_off = cur_buf * kStageTokens * p.D + slot * p.D + dim_base;
            Bf16x8 k_vec = load_bf16x8(&sh_k[sh_off]);
            Bf16x8 v_vec = load_bf16x8(&sh_v[sh_off]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              dot = fmaf(q_vals[i], bf16_to_float(k_vec.v[i]), dot);
            }
            dot = subwarp16_xor_sum(dot, subwarp_mask);
            float score = dot * score_scale;
            float new_m = fmaxf(m, score);
            float alpha = has_value ? ptx_exp2_approx(m - new_m) : 0.0f;
            float beta = ptx_exp2_approx(score - new_m);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              o_vals[i] = fmaf(beta, bf16_to_float(v_vec.v[i]), o_vals[i] * alpha);
            }
            denom = denom * alpha + beta;
            m = new_m;
            has_value = true;
          }
        }
      } else if (misspack_z2 && stage_count == kStageTokens) {
        for (int j = local_miss_part; j < kGroup;
             j += lanes_per_miss_head) {
          int slot = tz * kGroup + j;
          int pos = stage_begin + slot;
          bool contributes = tile_overlaps_head && pos >= head_begin && pos < head_end;
          if (contributes) {
            float dot = 0.0f;
            int sh_off = cur_buf * kStageTokens * p.D + slot * p.D + dim_base;
            Bf16x8 k_vec = load_bf16x8(&sh_k[sh_off]);
            Bf16x8 v_vec = load_bf16x8(&sh_v[sh_off]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              dot = fmaf(q_vals[i], bf16_to_float(k_vec.v[i]), dot);
            }
            dot = subwarp16_xor_sum(dot, subwarp_mask);
            float score = dot * score_scale;
            float new_m = fmaxf(m, score);
            float alpha = has_value ? ptx_exp2_approx(m - new_m) : 0.0f;
            float beta = ptx_exp2_approx(score - new_m);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              o_vals[i] = fmaf(beta, bf16_to_float(v_vec.v[i]), o_vals[i] * alpha);
            }
            denom = denom * alpha + beta;
            m = new_m;
            has_value = true;
          }
        }
      } else if (misspack_z2) {
        for (int j = local_miss_part; j < kGroup;
             j += lanes_per_miss_head) {
          int slot = tz * kGroup + j;
          int pos = stage_begin + slot;
          bool contributes =
              tile_overlaps_head && slot < stage_count && pos >= head_begin && pos < head_end;
          if (contributes) {
            float dot = 0.0f;
            int sh_off = cur_buf * kStageTokens * p.D + slot * p.D + dim_base;
            Bf16x8 k_vec = load_bf16x8(&sh_k[sh_off]);
            Bf16x8 v_vec = load_bf16x8(&sh_v[sh_off]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              dot = fmaf(q_vals[i], bf16_to_float(k_vec.v[i]), dot);
            }
            dot = subwarp16_xor_sum(dot, subwarp_mask);
            float score = dot * score_scale;
            float new_m = fmaxf(m, score);
            float alpha = has_value ? ptx_exp2_approx(m - new_m) : 0.0f;
            float beta = ptx_exp2_approx(score - new_m);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              o_vals[i] = fmaf(beta, bf16_to_float(v_vec.v[i]), o_vals[i] * alpha);
            }
            denom = denom * alpha + beta;
            m = new_m;
            has_value = true;
          }
        }
      } else if (stage_count == kStageTokens) {
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
          int slot = tz * kGroup + j;
          int pos = stage_begin + slot;
          bool contributes =
              tile_overlaps_head && (!mixed_z2 || (pos >= head_begin && pos < head_end));
          if (contributes) {
            float dot = 0.0f;
            int sh_off = cur_buf * kStageTokens * p.D + slot * p.D + dim_base;
            Bf16x8 k_vec = load_bf16x8(&sh_k[sh_off]);
            Bf16x8 v_vec = load_bf16x8(&sh_v[sh_off]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              dot = fmaf(q_vals[i], bf16_to_float(k_vec.v[i]), dot);
            }
            dot = subwarp16_xor_sum(dot, subwarp_mask);
            float score = dot * score_scale;
            float new_m = fmaxf(m, score);
            float alpha = has_value ? ptx_exp2_approx(m - new_m) : 0.0f;
            float beta = ptx_exp2_approx(score - new_m);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              o_vals[i] = fmaf(beta, bf16_to_float(v_vec.v[i]), o_vals[i] * alpha);
            }
            denom = denom * alpha + beta;
            m = new_m;
            has_value = true;
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
          int slot = tz * kGroup + j;
          int pos = stage_begin + slot;
          bool contributes =
              tile_overlaps_head && slot < stage_count &&
              (!mixed_z2 || (pos >= head_begin && pos < head_end));
          if (contributes) {
            float dot = 0.0f;
            int sh_off = cur_buf * kStageTokens * p.D + slot * p.D + dim_base;
            Bf16x8 k_vec = load_bf16x8(&sh_k[sh_off]);
            Bf16x8 v_vec = load_bf16x8(&sh_v[sh_off]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              dot = fmaf(q_vals[i], bf16_to_float(k_vec.v[i]), dot);
            }
            dot = subwarp16_xor_sum(dot, subwarp_mask);
            float score = dot * score_scale;
            float new_m = fmaxf(m, score);
            float alpha = has_value ? ptx_exp2_approx(m - new_m) : 0.0f;
            float beta = ptx_exp2_approx(score - new_m);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              o_vals[i] = fmaf(beta, bf16_to_float(v_vec.v[i]), o_vals[i] * alpha);
            }
            denom = denom * alpha + beta;
            m = new_m;
            has_value = true;
          }
        }
      }
      if (has_next) {
        cp_async_wait_all();
      }
      z2_stage_half_sync(tz);
      cur_buf = next_buf;
      stage_count = next_count;
    }

    int state = ty * kZParts + tz;
    bool emit_state = !misspack_z2 || tile_overlaps_head;
    if (active_z && emit_state && tx == 0) {
      float lse_log2 = denom > 0.0f ? (m + log2f(denom)) : -kInf;
      sh_lse[state] = lse_log2 * kLn2;
    }
    float inv_denom = denom > 0.0f ? ptx_rcp_approx(denom) : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int d = dim_base + i;
      if (active_z && emit_state) {
        sh_o[state * p.D + d] = o_vals[i] * inv_denom;
      }
    }
    __syncthreads();

    if (misspack_z2) {
      if (active_z && tile_overlaps_head && local_miss_part == 0 && tz == 0) {
        int state_base_ty = miss_slot * lanes_per_miss_head;
        float max_lse = -kInf;
#pragma unroll
        for (int lp = 0; lp < kGroup; ++lp) {
          if (lp < lanes_per_miss_head) {
#pragma unroll
            for (int z = 0; z < kZParts; ++z) {
              int state_i = (state_base_ty + lp) * kZParts + z;
              float lse_z = sh_lse[state_i];
              if (isfinite(lse_z)) {
                max_lse = fmaxf(max_lse, lse_z);
              }
            }
          }
        }
        float weights[kGroup * kZParts];
#pragma unroll
        for (int i = 0; i < kGroup * kZParts; ++i) {
          weights[i] = 0.0f;
        }
        float denom_merge = 0.0f;
#pragma unroll
        for (int lp = 0; lp < kGroup; ++lp) {
          if (lp < lanes_per_miss_head) {
#pragma unroll
            for (int z = 0; z < kZParts; ++z) {
              int out_i = lp * kZParts + z;
              int state_i = (state_base_ty + lp) * kZParts + z;
              float lse_z = sh_lse[state_i];
              float w =
                  isfinite(lse_z) ? ptx_exp2_approx((lse_z - max_lse) * kLog2e) : 0.0f;
              weights[out_i] = w;
              denom_merge += w;
            }
          }
        }
        float inv_merge = denom_merge > 0.0f ? ptx_rcp_approx(denom_merge) : 0.0f;
        float out_lse = denom_merge > 0.0f ? (max_lse + __logf(denom_merge)) : -kInf;
#pragma unroll
        for (int i = 0; i < kGroup * kZParts; ++i) {
          weights[i] *= inv_merge;
        }

        int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * max_tiles + tile) *
                           p.group_size + out_lane);
        if (tx == 0) {
          partial_lse[lse_off] = out_lse;
        }
        int64_t o_base = lse_off * p.D;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          int d = dim_base + i;
          float merged = 0.0f;
#pragma unroll
          for (int lp = 0; lp < kGroup; ++lp) {
            if (lp < lanes_per_miss_head) {
#pragma unroll
              for (int z = 0; z < kZParts; ++z) {
                int out_i = lp * kZParts + z;
                int state_i = (state_base_ty + lp) * kZParts + z;
                merged = fmaf(weights[out_i], sh_o[state_i * p.D + d], merged);
              }
            }
          }
          store_partial_o(partial_o, o_base + d, merged, p.partial_o_bf16);
        }
      }
    } else if (active_z && tz == 0) {
      int state0 = ty * kZParts;
      float max_lse = -kInf;
#pragma unroll
      for (int z = 0; z < kZParts; ++z) {
        float lse_z = sh_lse[state0 + z];
        if (isfinite(lse_z)) {
          max_lse = fmaxf(max_lse, lse_z);
        }
      }
      float weights[kZParts];
      float denom_merge = 0.0f;
#pragma unroll
      for (int z = 0; z < kZParts; ++z) {
        float lse_z = sh_lse[state0 + z];
        float w = isfinite(lse_z) ? ptx_exp2_approx((lse_z - max_lse) * kLog2e) : 0.0f;
        weights[z] = w;
        denom_merge += w;
      }
      float inv_merge = denom_merge > 0.0f ? ptx_rcp_approx(denom_merge) : 0.0f;
      float out_lse = denom_merge > 0.0f ? (max_lse + __logf(denom_merge)) : -kInf;
#pragma unroll
      for (int z = 0; z < kZParts; ++z) {
        weights[z] *= inv_merge;
      }

      int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * max_tiles + tile) *
                         p.group_size + ty);
      if (tx == 0) {
        partial_lse[lse_off] =
            keep_log2_for_fallback_reduce && isfinite(out_lse) ? (out_lse * kLog2e) : out_lse;
      }
      int64_t o_base = lse_off * p.D;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int d = dim_base + i;
        float merged = 0.0f;
#pragma unroll
        for (int z = 0; z < kZParts; ++z) {
          merged = fmaf(weights[z], sh_o[(state0 + z) * p.D + d], merged);
        }
        store_partial_o(partial_o, o_base + d, merged, p.partial_o_bf16);
      }
    }
    __syncthreads();
  }
}

__device__ void phase_tail_attention_group_direct_z2(
    const Params& p,
    const __nv_bfloat16* __restrict__ q_post,
    const __nv_bfloat16* __restrict__ k_buffer,
    const __nv_bfloat16* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    const int32_t* __restrict__ group_begin,
    const int32_t* __restrict__ group_end,
    const int32_t* __restrict__ group_tiles,
    const int32_t* __restrict__ group_mode,
    const int32_t* __restrict__ task_go,
    const int32_t* __restrict__ task_tile,
    const int32_t* __restrict__ task_counts,
    void* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    int max_tiles,
    uint8_t* __restrict__ dyn_smem,
    bool tail_group_direct_z2_on) {
  if (!tail_group_direct_z2_on || dyn_smem == nullptr) {
    return;
  }

  constexpr int kGroup = kDirectZ2Group;
  constexpr int kZParts = kDirectZ2ZParts;
  constexpr int kStageTokens = kDirectZ2StageTokens;
  const int tx = threadIdx.x & 15;
  const int local = threadIdx.x >> 4;
  const int ty = local & 3;
  const int tz = (local >> 2) & 1;
  const unsigned subwarp_mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
  const int dim_base = tx * 8;

  __nv_bfloat16* sh_k = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
  __nv_bfloat16* sh_v = sh_k + 2 * kStageTokens * kHeadDim;
  float* sh_o = reinterpret_cast<float*>(sh_v + 2 * kStageTokens * kHeadDim);
  float* sh_lse = sh_o + kGroup * kZParts * kHeadDim;

  const int total = task_counts[1];
  const int group_total = p.N * p.Hkv;
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int encoded_task = task_go[task];
    if (encoded_task < 0 || encoded_task >= group_total) continue;
    int64_t go = static_cast<int64_t>(encoded_task);
    int tile = task_tile[task];
    int tiles = group_tiles[go];
    if (tile >= tiles) continue;

    int hkv = static_cast<int>(go % p.Hkv);
    int n = static_cast<int>(go / p.Hkv);
    int begin = group_begin[go] + tile * p.tile_tokens;
    int end = min(begin + p.tile_tokens, group_end[go]);
    if (end <= begin) continue;

    int req = req_ids[n];
    int past_len = past_lens[n];
    int hq = hkv * p.group_size + ty;
    const float score_scale = p.sm_scale * kLog2e;

    float q_vals[8];
    float o_vals[8];
    float m = -kInf;
    float denom = 0.0f;
    bool has_value = false;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int d = dim_base + i;
      o_vals[i] = 0.0f;
      q_vals[i] = bf16_to_float(q_post[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d]);
    }

    int stage_begin = begin;
    int stage_count = min(kStageTokens, end - stage_begin);
    int cur_buf = 0;
    int load_slot = tz * kGroup + ty;
    if (load_slot < stage_count) {
      int load_pos = stage_begin + load_slot;
      int phys = physical_token(n, req, load_pos, past_len, req_to_token, p.req_to_token_stride,
                                out_cache_loc_i32, out_cache_loc_i64,
                                p.out_cache_loc_is_i64);
      int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + dim_base;
      int sh_off = cur_buf * kStageTokens * p.D + load_slot * p.D + dim_base;
      cp_async_cg_16(&sh_k[sh_off], &k_buffer[off]);
      cp_async_cg_16(&sh_v[sh_off], &v_buffer[off]);
    }
    cp_async_commit();
    cp_async_wait_all();
    z2_stage_half_sync(tz);

    for (; stage_begin < end; stage_begin += kStageTokens) {
      int next_begin = stage_begin + kStageTokens;
      int next_count = min(kStageTokens, end - next_begin);
      int next_buf = cur_buf ^ 1;
      bool has_next = next_begin < end;
      int load_slot_next = tz * kGroup + ty;
      if (has_next && load_slot_next < next_count) {
        int load_pos = next_begin + load_slot_next;
        int phys = physical_token(n, req, load_pos, past_len, req_to_token,
                                  p.req_to_token_stride, out_cache_loc_i32,
                                  out_cache_loc_i64, p.out_cache_loc_is_i64);
        int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + dim_base;
        int sh_off = next_buf * kStageTokens * p.D + load_slot_next * p.D + dim_base;
        cp_async_cg_16(&sh_k[sh_off], &k_buffer[off]);
        cp_async_cg_16(&sh_v[sh_off], &v_buffer[off]);
      }
      cp_async_commit();

      if (stage_count == kStageTokens) {
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
          int slot = tz * kGroup + j;
          float dot = 0.0f;
          int sh_off = cur_buf * kStageTokens * p.D + slot * p.D + dim_base;
          Bf16x8 k_vec = load_bf16x8(&sh_k[sh_off]);
          Bf16x8 v_vec = load_bf16x8(&sh_v[sh_off]);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            dot = fmaf(q_vals[i], bf16_to_float(k_vec.v[i]), dot);
          }
          dot = subwarp16_xor_sum(dot, subwarp_mask);
          float score = dot * score_scale;
          float new_m = fmaxf(m, score);
          float alpha = has_value ? ptx_exp2_approx(m - new_m) : 0.0f;
          float beta = ptx_exp2_approx(score - new_m);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            o_vals[i] = fmaf(beta, bf16_to_float(v_vec.v[i]), o_vals[i] * alpha);
          }
          denom = denom * alpha + beta;
          m = new_m;
          has_value = true;
        }
      } else {
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
          int slot = tz * kGroup + j;
          if (slot < stage_count) {
            float dot = 0.0f;
            int sh_off = cur_buf * kStageTokens * p.D + slot * p.D + dim_base;
            Bf16x8 k_vec = load_bf16x8(&sh_k[sh_off]);
            Bf16x8 v_vec = load_bf16x8(&sh_v[sh_off]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              dot = fmaf(q_vals[i], bf16_to_float(k_vec.v[i]), dot);
            }
            dot = subwarp16_xor_sum(dot, subwarp_mask);
            float score = dot * score_scale;
            float new_m = fmaxf(m, score);
            float alpha = has_value ? ptx_exp2_approx(m - new_m) : 0.0f;
            float beta = ptx_exp2_approx(score - new_m);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              o_vals[i] = fmaf(beta, bf16_to_float(v_vec.v[i]), o_vals[i] * alpha);
            }
            denom = denom * alpha + beta;
            m = new_m;
            has_value = true;
          }
        }
      }
      if (has_next) {
        cp_async_wait_all();
      }
      z2_stage_half_sync(tz);
      cur_buf = next_buf;
      stage_count = next_count;
    }

    int state = ty * kZParts + tz;
    if (tx == 0) {
      float lse_log2 = denom > 0.0f ? (m + log2f(denom)) : -kInf;
      sh_lse[state] = lse_log2 * kLn2;
    }
    float inv_denom = denom > 0.0f ? ptx_rcp_approx(denom) : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int d = dim_base + i;
      sh_o[state * p.D + d] = o_vals[i] * inv_denom;
    }
    __syncthreads();

    if (tz == 0) {
      int state0 = ty * kZParts;
      float max_lse = -kInf;
#pragma unroll
      for (int z = 0; z < kZParts; ++z) {
        float lse_z = sh_lse[state0 + z];
        if (isfinite(lse_z)) {
          max_lse = fmaxf(max_lse, lse_z);
        }
      }
      float weights[kZParts];
      float denom_merge = 0.0f;
#pragma unroll
      for (int z = 0; z < kZParts; ++z) {
        float lse_z = sh_lse[state0 + z];
        float w = isfinite(lse_z) ? ptx_exp2_approx((lse_z - max_lse) * kLog2e) : 0.0f;
        weights[z] = w;
        denom_merge += w;
      }
      float inv_merge = denom_merge > 0.0f ? ptx_rcp_approx(denom_merge) : 0.0f;
      float out_lse = denom_merge > 0.0f ? (max_lse + __logf(denom_merge)) : -kInf;
#pragma unroll
      for (int z = 0; z < kZParts; ++z) {
        weights[z] *= inv_merge;
      }

      int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * max_tiles + tile) *
                         p.group_size + ty);
      if (tx == 0) {
        partial_lse[lse_off] = out_lse;
      }
      int64_t o_base = lse_off * p.D;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int d = dim_base + i;
        float merged = 0.0f;
#pragma unroll
        for (int z = 0; z < kZParts; ++z) {
          merged = fmaf(weights[z], sh_o[(state0 + z) * p.D + d], merged);
        }
        store_partial_o(partial_o, o_base + d, merged, p.partial_o_bf16);
      }
    }
    __syncthreads();
  }
}

__device__ void phase_complete_attention_head_direct(
    const Params& p,
    const __nv_bfloat16* __restrict__ q_post,
    const __nv_bfloat16* __restrict__ k_buffer,
    const __nv_bfloat16* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    const int32_t* __restrict__ head_rect_start,
    const int32_t* __restrict__ head_new_end,
    const int32_t* __restrict__ group_begin,
    const int32_t* __restrict__ group_end,
    const int32_t* __restrict__ group_tiles,
    const int32_t* __restrict__ group_mode,
    const int32_t* __restrict__ task_go,
    const int32_t* __restrict__ task_tile,
    const int32_t* __restrict__ task_counts,
    void* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    int max_tiles) {
  const int total = task_counts[0];
  const int subwarp = threadIdx.x >> 4;
  const int lane = threadIdx.x & 15;
  const int subwarps = blockDim.x >> 4;
  const unsigned subwarp_mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;

  for (int task = blockIdx.x * subwarps + subwarp; task < total; task += gridDim.x * subwarps) {
    int encoded_task = task_go[task];
    if (encoded_task >= 0) continue;

    int tile = task_tile[task];
    int ho_i = -encoded_task - 1;
    int hq = ho_i % p.Hq;
    int n = ho_i / p.Hq;
    int hkv = hq / p.group_size;
    int lane_id = hq - hkv * p.group_size;
    int64_t go = static_cast<int64_t>(n) * p.Hkv + hkv;
    int tiles = group_tiles[go];
    if (tile >= tiles) continue;

    int group_tokens = group_end[go] - group_begin[go];
    int tile_tokens = producer_rect_tile_tokens_for(p, task_counts, group_mode[go], group_tokens);
    int begin = group_begin[go] + tile * tile_tokens;
    int end = min(begin + tile_tokens, group_end[go]);
    int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
    begin = max(begin, head_rect_start[ho]);
    end = min(end, head_new_end[ho]);
    if (end <= begin) continue;

    int req = req_ids[n];
    int past_len = past_lens[n];
    int64_t req_token_base = static_cast<int64_t>(req) * p.req_to_token_stride;
    float m = -kInf;
    float denom = 0.0f;
    bool has_value = false;
    float q_vals[8];
    float o_vals[8];
    int dim_base = lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int d = dim_base + i;
      o_vals[i] = 0.0f;
      q_vals[i] = bf16_to_float(q_post[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d]);
    }

    for (int pos = begin; pos < end; ++pos) {
      int phys = req_to_token[req_token_base + pos];
      float dot = 0.0f;
      float v_vals[8];
      int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + dim_base;
      Bf16x8 k_vec = load_bf16x8(&k_buffer[off]);
      Bf16x8 v_vec = load_bf16x8(&v_buffer[off]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        dot = fmaf(q_vals[i], bf16_to_float(k_vec.v[i]), dot);
        v_vals[i] = bf16_to_float(v_vec.v[i]);
      }
      dot = subwarp16_xor_sum(dot, subwarp_mask);
      float score = dot * (p.sm_scale * kLog2e);
      float new_m = fmaxf(m, score);
      float alpha = has_value ? ptx_exp2_approx(m - new_m) : 0.0f;
      float beta = ptx_exp2_approx(score - new_m);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o_vals[i] = o_vals[i] * alpha + beta * v_vals[i];
      }
      denom = denom * alpha + beta;
      m = new_m;
      has_value = true;
    }

    int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * max_tiles + tile) *
                       p.group_size + lane_id);
    if (lane == 0) {
      partial_lse[lse_off] = denom > 0.0f ? (m * kLn2 + __logf(denom)) : -kInf;
    }
    int64_t o_base = lse_off * p.D;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int d = dim_base + i;
      store_partial_o(partial_o, o_base + d,
                      denom > 0.0f ? (o_vals[i] / denom) : 0.0f,
                      p.partial_o_bf16);
    }
  }
}

__device__ void phase_all_hit_complete_direct(
    const Params& p,
    const __nv_bfloat16* __restrict__ q_post,
    const __nv_bfloat16* __restrict__ k_buffer,
    const __nv_bfloat16* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    const int32_t* __restrict__ head_rect_start,
    const int32_t* __restrict__ head_new_end,
    const int32_t* __restrict__ group_begin,
    const int32_t* __restrict__ group_end,
    const int32_t* __restrict__ group_tiles,
    const int32_t* __restrict__ group_mode,
    void* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    int max_tiles) {
  const int max_hit_tiles = min(max_tiles, ceil_div_i32(max(p.M, 1), p.tile_tokens));
  const int total = p.N * p.Hkv * max_hit_tiles;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  __shared__ __nv_bfloat16 sh_k[kHeadDim];
  __shared__ __nv_bfloat16 sh_v[kHeadDim];

  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int tile = task % max_hit_tiles;
    int64_t go = static_cast<int64_t>(task / max_hit_tiles);
    if (group_mode[go] != kGroupModeMacHit) continue;
    int hkv = static_cast<int>(go % p.Hkv);
    int n = static_cast<int>(go / p.Hkv);
    int tiles = group_tiles[go];
    if (tile >= tiles) continue;

    int begin = group_begin[go] + tile * p.tile_tokens;
    int end = min(begin + p.tile_tokens, group_end[go]);
    if (end <= begin) continue;
    int req = req_ids[n];
    int past_len = past_lens[n];
    int hq = hkv * p.group_size + warp;
    bool tile_overlaps_head = false;
    if (warp < p.group_size) {
      int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
      tile_overlaps_head = end > head_rect_start[ho] && begin < head_new_end[ho];
    }

    float m = -kInf;
    float denom = 0.0f;
    bool has_value = false;
    float q_vals[4];
    float o_vals[4];
    int dims[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = lane + i * 32;
      dims[i] = d;
      o_vals[i] = 0.0f;
      q_vals[i] = 0.0f;
      if (warp < p.group_size && d < p.D) {
        q_vals[i] = bf16_to_float(q_post[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d]);
      }
    }

    for (int pos = begin; pos < end; ++pos) {
      int phys = physical_token(
          n, req, pos, past_len, req_to_token, p.req_to_token_stride,
          out_cache_loc_i32, out_cache_loc_i64, p.out_cache_loc_is_i64);
      for (int d = threadIdx.x; d < p.D; d += blockDim.x) {
        int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + d;
        sh_k[d] = k_buffer[off];
        sh_v[d] = v_buffer[off];
      }
      __syncthreads();

      bool contributes = false;
      if (warp < p.group_size) {
        int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
        contributes = pos >= head_rect_start[ho] && pos < head_new_end[ho];
      }
      if (contributes) {
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int d = dims[i];
          if (d < p.D) {
            dot = fmaf(q_vals[i], bf16_to_float(sh_k[d]), dot);
          }
        }
        dot = warp_sum(dot);
        float score = __shfl_sync(0xffffffffu, dot, 0) * p.sm_scale;
        float new_m = fmaxf(m, score);
        float alpha = has_value ? expf(m - new_m) : 0.0f;
        float beta = expf(score - new_m);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int d = dims[i];
          if (d < p.D) {
            o_vals[i] = o_vals[i] * alpha + beta * bf16_to_float(sh_v[d]);
          }
        }
        denom = denom * alpha + beta;
        m = new_m;
        has_value = true;
      }
      __syncthreads();
    }

    if (warp < p.group_size && tile_overlaps_head) {
      int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * max_tiles + tile) *
                         p.group_size + warp);
      float out_lse = denom > 0.0f ? (m + logf(denom)) : -kInf;
      if (lane == 0) {
        partial_lse[lse_off] = out_lse;
      }
      int64_t o_base = lse_off * p.D;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int d = dims[i];
        if (d < p.D) {
          store_partial_o(partial_o, o_base + d,
                          denom > 0.0f ? (o_vals[i] / denom) : 0.0f,
                          p.partial_o_bf16);
        }
      }
    }
    __syncthreads();
  }
}

__device__ void phase_tail_attention_head_direct(
    const Params& p,
    const __nv_bfloat16* __restrict__ q_post,
    const __nv_bfloat16* __restrict__ k_buffer,
    const __nv_bfloat16* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    const int32_t* __restrict__ group_tail_begin,
    const int32_t* __restrict__ group_tail_end,
    const int32_t* __restrict__ hit_tail_task_ho,
    const int32_t* __restrict__ hit_tail_task_tile,
    const int32_t* __restrict__ task_counts,
    void* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    int max_tiles) {
  const int total = task_counts[2];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;

  for (int task = blockIdx.x * warps + warp; task < total; task += gridDim.x * warps) {
    int ho_i = hit_tail_task_ho[task];
    int tile = hit_tail_task_tile[task];
    int hq = ho_i % p.Hq;
    int n = ho_i / p.Hq;
    int hkv = hq / p.group_size;
    int lane_id = hq - hkv * p.group_size;
    int64_t go = static_cast<int64_t>(n) * p.Hkv + hkv;
    int begin = group_tail_begin[go] + tile * p.tile_tokens;
    int end = min(begin + p.tile_tokens, group_tail_end[go]);
    if (end <= begin) continue;

    int req = req_ids[n];
    int past_len = past_lens[n];
    float m = -kInf;
    float denom = 0.0f;
    bool has_value = false;
    float q_vals[4];
    float o_vals[4];
    int dims[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = lane + i * 32;
      dims[i] = d;
      o_vals[i] = 0.0f;
      q_vals[i] = 0.0f;
      if (d < p.D) {
        q_vals[i] = bf16_to_float(q_post[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d]);
      }
    }

    for (int pos = begin; pos < end; ++pos) {
      int phys = physical_token(
          n, req, pos, past_len, req_to_token, p.req_to_token_stride,
          out_cache_loc_i32, out_cache_loc_i64, p.out_cache_loc_is_i64);
      float dot = 0.0f;
      float v_vals[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int d = dims[i];
        v_vals[i] = 0.0f;
        if (d < p.D) {
          int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + d;
          dot = fmaf(q_vals[i], bf16_to_float(k_buffer[off]), dot);
          v_vals[i] = bf16_to_float(v_buffer[off]);
        }
      }
      dot = warp_sum(dot);
      float score = __shfl_sync(0xffffffffu, dot, 0) * p.sm_scale;
      float new_m = fmaxf(m, score);
      float alpha = has_value ? expf(m - new_m) : 0.0f;
      float beta = expf(score - new_m);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int d = dims[i];
        if (d < p.D) {
          o_vals[i] = o_vals[i] * alpha + beta * v_vals[i];
        }
      }
      denom = denom * alpha + beta;
      m = new_m;
      has_value = true;
    }

    int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * max_tiles + tile) *
                       p.group_size + lane_id);
    float out_lse = denom > 0.0f ? (m + logf(denom)) : -kInf;
    if (lane == 0) {
      partial_lse[lse_off] = out_lse;
    }
    int64_t o_base = lse_off * p.D;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = dims[i];
      if (d < p.D) {
        store_partial_o(partial_o, o_base + d,
                        denom > 0.0f ? (o_vals[i] / denom) : 0.0f,
                        p.partial_o_bf16);
      }
    }
  }
}

__device__ void phase_all_hit_tail_direct(
    const Params& p,
    const __nv_bfloat16* __restrict__ q_post,
    const __nv_bfloat16* __restrict__ k_buffer,
    const __nv_bfloat16* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    const int32_t* __restrict__ group_tail_begin,
    const int32_t* __restrict__ group_tail_end,
    const int32_t* __restrict__ group_tail_tiles,
    const int32_t* __restrict__ group_mode,
    void* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    int max_tiles) {
  if ((threadIdx.x >> 5) != 0) return;
  const int total = p.N * p.Hq * max_tiles;
  const int lane = threadIdx.x & 31;

  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int tile = task % max_tiles;
    int ho_i = task / max_tiles;
    int hq = ho_i % p.Hq;
    int n = ho_i / p.Hq;
    int hkv = hq / p.group_size;
    int lane_id = hq - hkv * p.group_size;
    int64_t go = static_cast<int64_t>(n) * p.Hkv + hkv;
    if (group_mode[go] != kGroupModeMacHit) continue;
    int tail_tiles = group_tail_tiles[go];
    if (tile >= tail_tiles) continue;

    int begin = group_tail_begin[go] + tile * p.tile_tokens;
    int end = min(begin + p.tile_tokens, group_tail_end[go]);
    if (end <= begin) continue;

    int req = req_ids[n];
    int past_len = past_lens[n];
    float m = -kInf;
    float denom = 0.0f;
    bool has_value = false;
    float q_vals[4];
    float o_vals[4];
    int dims[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = lane + i * 32;
      dims[i] = d;
      o_vals[i] = 0.0f;
      q_vals[i] = 0.0f;
      if (d < p.D) {
        q_vals[i] = bf16_to_float(q_post[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d]);
      }
    }

    for (int pos = begin; pos < end; ++pos) {
      int phys = physical_token(
          n, req, pos, past_len, req_to_token, p.req_to_token_stride,
          out_cache_loc_i32, out_cache_loc_i64, p.out_cache_loc_is_i64);
      float dot = 0.0f;
      float v_vals[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int d = dims[i];
        v_vals[i] = 0.0f;
        if (d < p.D) {
          int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + d;
          dot = fmaf(q_vals[i], bf16_to_float(k_buffer[off]), dot);
          v_vals[i] = bf16_to_float(v_buffer[off]);
        }
      }
      dot = warp_sum(dot);
      float score = __shfl_sync(0xffffffffu, dot, 0) * p.sm_scale;
      float new_m = fmaxf(m, score);
      float alpha = has_value ? expf(m - new_m) : 0.0f;
      float beta = expf(score - new_m);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int d = dims[i];
        if (d < p.D) {
          o_vals[i] = o_vals[i] * alpha + beta * v_vals[i];
        }
      }
      denom = denom * alpha + beta;
      m = new_m;
      has_value = true;
    }

    int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * max_tiles + tile) *
                       p.group_size + lane_id);
    float out_lse = denom > 0.0f ? (m + logf(denom)) : -kInf;
    if (lane == 0) {
      partial_lse[lse_off] = out_lse;
    }
    int64_t o_base = lse_off * p.D;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = dims[i];
      if (d < p.D) {
        store_partial_o(partial_o, o_base + d,
                        denom > 0.0f ? (o_vals[i] / denom) : 0.0f,
                        p.partial_o_bf16);
      }
    }
  }
}

__device__ void phase_reduce_full_fallback_rect(
    const Params& p,
    const int32_t* __restrict__ group_mode,
    const int32_t* __restrict__ group_rect_begin,
    const int32_t* __restrict__ group_rect_end,
    const int32_t* __restrict__ head_rect_start,
    const int32_t* __restrict__ head_new_end,
    int32_t* __restrict__ group_rect_tiles,
    const int32_t* __restrict__ task_counts,
    const void* __restrict__ partial_rect_o,
    const float* __restrict__ partial_rect_lse,
    void* __restrict__ partial_rect_reduced_o,
    float* __restrict__ partial_rect_reduced_lse) {
  const int total = p.N * p.Hq * p.max_tiles_reduce;
  __shared__ float sh_lse;
  __shared__ bool sh_valid;
  __shared__ float sh_w_state;
  __shared__ float sh_w_other;
  __shared__ bool sh_take_other;
  __shared__ bool sh_other_valid;

  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int chunk = task % p.max_tiles_reduce;
    int ho_i = task / p.max_tiles_reduce;
    int hq = ho_i % p.Hq;
    int n = ho_i / p.Hq;
    int hkv = hq / p.group_size;
    int lane_id = hq - hkv * p.group_size;
    int64_t go = static_cast<int64_t>(n) * p.Hkv + hkv;
    int rect_tiles_marker = group_rect_tiles[go];
    bool per_head_rect = rect_tiles_marker < 0;
    if (!per_head_rect && group_mode[go] == kGroupModeMacHit) continue;
    if (!per_head_rect && group_mode[go] == kGroupModeFullFallback &&
        full_fallback_warp_reduce_enabled(p)) {
      continue;
    }

    int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
    int head_begin = head_rect_start[ho];
    int head_end = head_new_end[ho];
    int group_tokens = group_rect_end[go] - group_rect_begin[go];
    int tile_tokens = per_head_rect
                          ? p.tile_tokens
                          : producer_rect_tile_tokens_for(p, task_counts, group_mode[go],
                                                          group_tokens);
    int rect_tokens = per_head_rect ? (head_end - head_begin)
                                    : (group_rect_end[go] - group_rect_begin[go]);
    int orig_tiles = rect_tokens > 0 ? ceil_div_i32(rect_tokens, tile_tokens) : 0;
    if (orig_tiles <= rect_reduce_threshold_for_counts(p, group_mode[go], task_counts)) continue;
    int reduce_chunk = clamp_reduce_chunk_to_workspace(
        balanced_rect_reduce_chunk_for_mode(p, group_mode[go], task_counts), orig_tiles,
        p.max_tiles_reduce);
    int reduced_tiles = ceil_div_i32(orig_tiles, reduce_chunk);
    if (chunk >= reduced_tiles) continue;
    if (!per_head_rect && chunk == 0 && threadIdx.x == 0) {
      group_rect_tiles[go] = reduced_tiles;
    }

    int tile_begin = chunk * reduce_chunk;
    int tile_end = min(tile_begin + reduce_chunk, orig_tiles);
    int group_begin = per_head_rect ? head_begin : group_rect_begin[go];
    int dim = threadIdx.x;
    bool owns_dim = dim < p.D;
    float acc = 0.0f;
    bool state_valid = false;
    float state_lse = -kInf;

    if (threadIdx.x == 0) {
      sh_lse = -kInf;
      sh_valid = false;
    }
    __syncthreads();

    for (int tile = tile_begin; tile < tile_end; ++tile) {
      int begin = group_begin + tile * tile_tokens;
      int end = min(begin + tile_tokens, group_rect_end[go]);
      if (end <= head_begin || begin >= head_end) {
        continue;
      }
      int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context + tile) *
                         p.group_size + lane_id);
      if (threadIdx.x == 0) {
        float w_state = 1.0f, w_other = 0.0f;
        bool take_other = false;
        float other_lse = partial_rect_lse[lse_off];
        bool other_valid = isfinite(other_lse);
        merge_scalar(state_lse, state_valid, other_lse, w_state, w_other, take_other);
        sh_lse = state_lse;
        sh_valid = state_valid;
        sh_w_state = w_state;
        sh_w_other = w_other;
        sh_take_other = take_other;
        sh_other_valid = other_valid;
      }
      __syncthreads();
      state_lse = sh_lse;
      state_valid = sh_valid;
      if (owns_dim && (sh_take_other || sh_other_valid)) {
        float other = load_partial_o(partial_rect_o, lse_off * p.D + dim, p.partial_o_bf16);
        if (sh_take_other) {
          acc = other;
        } else {
          acc = sh_w_state * acc + sh_w_other * other;
        }
      }
      __syncthreads();
    }

    int64_t reduced_lse_off =
        (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_reduce + chunk) *
         p.group_size + lane_id);
    if (threadIdx.x == 0) {
      partial_rect_reduced_lse[reduced_lse_off] = state_valid ? state_lse : -kInf;
    }
    if (owns_dim) {
      store_partial_o(partial_rect_reduced_o, reduced_lse_off * p.D + dim,
                      state_valid ? acc : 0.0f, p.partial_o_bf16);
    }
    __syncthreads();
  }
}

__device__ void phase_reduce_full_fallback_group_warp(
    const Params& p,
    const int32_t* __restrict__ group_mode,
    const int32_t* __restrict__ group_rect_begin,
    const int32_t* __restrict__ group_rect_end,
    int32_t* __restrict__ group_rect_tiles,
    const int32_t* __restrict__ task_counts,
    const void* __restrict__ partial_rect_o,
    const float* __restrict__ partial_rect_lse,
    void* __restrict__ partial_rect_reduced_o,
    float* __restrict__ partial_rect_reduced_lse) {
  if (!full_fallback_warp_reduce_enabled(p)) return;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= p.group_size) return;
  const unsigned mask = 0xffffffffu;
  const int max_reduce_chunks =
      max_rect_reduce_chunks_for_mode(p, kGroupModeFullFallback, task_counts);
  const int total = p.N * p.Hkv * max_reduce_chunks;

  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int chunk = task % max_reduce_chunks;
    int go_i = task / max_reduce_chunks;
    int64_t go = static_cast<int64_t>(go_i);
    if (group_mode[go] != kGroupModeFullFallback) continue;

    int hkv = go_i % p.Hkv;
    int n = go_i / p.Hkv;
    int group_tokens = group_rect_end[go] - group_rect_begin[go];
    int tile_tokens = producer_rect_tile_tokens_for(p, task_counts, group_mode[go], group_tokens);
    int orig_tiles = group_tokens > 0 ? ceil_div_i32(group_tokens, tile_tokens) : 0;
    if (orig_tiles <= rect_reduce_threshold_for_counts(p, group_mode[go], task_counts)) continue;
    int reduce_chunk = clamp_reduce_chunk_to_workspace(
        balanced_rect_reduce_chunk_for_mode(p, group_mode[go], task_counts), orig_tiles,
        p.max_tiles_reduce);
    int reduced_tiles = ceil_div_i32(orig_tiles, reduce_chunk);
    if (chunk >= reduced_tiles) continue;
    if (chunk == 0 && threadIdx.x == 0) {
      group_rect_tiles[go] = reduced_tiles;
    }

    int tile_begin = chunk * reduce_chunk;
    int tile_end = min(tile_begin + reduce_chunk, orig_tiles);
    int dims[4];
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      dims[i] = lane + i * 32;
      acc[i] = 0.0f;
    }
    float state_m_log2 = -kInf;
    float state_d = 0.0f;
    bool state_valid = false;

    for (int tile = tile_begin; tile < tile_end; ++tile) {
      int64_t lse_off =
          (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context + tile) *
           p.group_size + warp);
      int other_valid_i = 0;
      float scale_state = 1.0f;
      float scale_other = 0.0f;
      if (lane == 0) {
        float other_lse = partial_rect_lse[lse_off];
        bool other_valid = isfinite(other_lse);
        other_valid_i = other_valid ? 1 : 0;
        if (other_valid) {
          float other_m = other_lse;
          if (!state_valid) {
            state_m_log2 = other_m;
            state_d = 1.0f;
            state_valid = true;
            scale_state = 0.0f;
            scale_other = 1.0f;
          } else {
            float new_m = fmaxf(state_m_log2, other_m);
            scale_state = ptx_exp2_approx(state_m_log2 - new_m);
            scale_other = ptx_exp2_approx(other_m - new_m);
            state_d = state_d * scale_state + scale_other;
            state_m_log2 = new_m;
          }
        }
      }
      state_m_log2 = __shfl_sync(mask, state_m_log2, 0);
      state_d = __shfl_sync(mask, state_d, 0);
      state_valid = __shfl_sync(mask, state_valid ? 1 : 0, 0) != 0;
      scale_state = __shfl_sync(mask, scale_state, 0);
      scale_other = __shfl_sync(mask, scale_other, 0);
      other_valid_i = __shfl_sync(mask, other_valid_i, 0);
      if (other_valid_i) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int d = dims[i];
          float other = load_partial_o(partial_rect_o, lse_off * p.D + d, p.partial_o_bf16);
          acc[i] = scale_state * acc[i] + scale_other * other;
        }
      }
    }

    int64_t reduced_lse_off =
        (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_reduce + chunk) *
         p.group_size + warp);
    if (lane == 0) {
      partial_rect_reduced_lse[reduced_lse_off] =
          state_valid ? (state_m_log2 + log2f(state_d)) * kLn2 : -kInf;
    }
    float inv_d = state_valid ? ptx_rcp_approx(state_d) : 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = dims[i];
      store_partial_o(partial_rect_reduced_o, reduced_lse_off * p.D + d,
                      state_valid ? acc[i] * inv_d : 0.0f, p.partial_o_bf16);
    }
  }
}

__device__ void phase_reduce_full_fallback_head_vec(
    const Params& p,
    const int32_t* __restrict__ group_mode,
    const int32_t* __restrict__ group_rect_begin,
    const int32_t* __restrict__ group_rect_end,
    int32_t* __restrict__ group_rect_tiles,
    const int32_t* __restrict__ task_counts,
    const void* __restrict__ partial_rect_o,
    const float* __restrict__ partial_rect_lse,
    void* __restrict__ partial_rect_reduced_o,
    float* __restrict__ partial_rect_reduced_lse) {
  if (!full_fallback_head_reduce_enabled(p)) return;

  constexpr int kBdx = 16;
  constexpr int kBdy = 8;
  constexpr int kVec = 8;
  const int tx = threadIdx.x & (kBdx - 1);
  const int ty = threadIdx.x >> 4;
  if (ty >= kBdy) return;

  __shared__ float sh_o[kBdy * kHeadDim];
  __shared__ float sh_m[kBdy];
  __shared__ float sh_d[kBdy];

  const int max_reduce_chunks =
      max_rect_reduce_chunks_for_mode(p, kGroupModeFullFallback, task_counts);
  const int total = p.N * p.Hq * max_reduce_chunks;
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int chunk = task % max_reduce_chunks;
    int ho_i = task / max_reduce_chunks;
    int hq = ho_i % p.Hq;
    int n = ho_i / p.Hq;
    int hkv = hq / p.group_size;
    int lane_id = hq - hkv * p.group_size;
    int64_t go = static_cast<int64_t>(n) * p.Hkv + hkv;
    if (group_mode[go] != kGroupModeFullFallback) continue;

    int group_tokens = group_rect_end[go] - group_rect_begin[go];
    int tile_tokens = producer_rect_tile_tokens_for(p, task_counts, group_mode[go], group_tokens);
    int orig_tiles = group_tokens > 0 ? ceil_div_i32(group_tokens, tile_tokens) : 0;
    if (orig_tiles <= rect_reduce_threshold_for_counts(p, group_mode[go], task_counts)) continue;
    int reduce_chunk = clamp_reduce_chunk_to_workspace(
        balanced_rect_reduce_chunk_for_mode(p, group_mode[go], task_counts), orig_tiles,
        p.max_tiles_reduce);
    int reduced_tiles = ceil_div_i32(orig_tiles, reduce_chunk);
    if (chunk >= reduced_tiles) continue;
    if (chunk == 0 && lane_id == 0 && threadIdx.x == 0) {
      group_rect_tiles[go] = reduced_tiles;
    }

    int tile_begin = chunk * reduce_chunk;
    int tile_end = min(tile_begin + reduce_chunk, orig_tiles);
    int dim_base = tx * kVec;

    float local_o[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      local_o[i] = 0.0f;
    }
    float local_m = -kInf;
    float local_d = 0.0f;
    bool local_valid = false;

    for (int tile = tile_begin + ty; tile < tile_end; tile += kBdy) {
      int64_t lse_off =
          (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context + tile) *
           p.group_size + lane_id);
      float other_lse = partial_rect_lse[lse_off];
      if (!isfinite(other_lse)) continue;
      float other_m = other_lse;
      float new_m = local_valid ? fmaxf(local_m, other_m) : other_m;
      float scale_state = local_valid ? ptx_exp2_approx(local_m - new_m) : 0.0f;
      float scale_other = ptx_exp2_approx(other_m - new_m);
      int64_t o_base = lse_off * p.D + dim_base;
      if (p.partial_o_bf16) {
        Bf16x8 other_pack =
            load_bf16x8(reinterpret_cast<const __nv_bfloat16*>(partial_rect_o) + o_base);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          float other = bf16_to_float(other_pack.v[i]);
          local_o[i] = scale_state * local_o[i] + scale_other * other;
        }
      } else {
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          int d = dim_base + i;
          float other = reinterpret_cast<const float*>(partial_rect_o)[lse_off * p.D + d];
          local_o[i] = scale_state * local_o[i] + scale_other * other;
        }
      }
      local_d = local_d * scale_state + scale_other;
      local_m = new_m;
      local_valid = true;
    }

    if (tx == 0) {
      sh_m[ty] = local_valid ? local_m : -kInf;
      sh_d[ty] = local_valid ? local_d : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      int d = dim_base + i;
      sh_o[ty * kHeadDim + d] = local_valid ? local_o[i] : 0.0f;
    }
    __syncthreads();

    if (ty == 0) {
      float acc[kVec];
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        acc[i] = 0.0f;
      }
      float state_m = -kInf;
      float state_d = 0.0f;
      bool state_valid = false;

#pragma unroll
      for (int part = 0; part < kBdy; ++part) {
        float other_m = sh_m[part];
        float other_d = sh_d[part];
        if (!(isfinite(other_m) && other_d > 0.0f)) continue;
        float new_m = state_valid ? fmaxf(state_m, other_m) : other_m;
        float scale_state = state_valid ? ptx_exp2_approx(state_m - new_m) : 0.0f;
        float scale_other = ptx_exp2_approx(other_m - new_m);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          int d = dim_base + i;
          float other = sh_o[part * kHeadDim + d];
          acc[i] = scale_state * acc[i] + scale_other * other;
        }
        state_d = state_d * scale_state + other_d * scale_other;
        state_m = new_m;
        state_valid = true;
      }

      int64_t reduced_lse_off =
          (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_reduce + chunk) *
           p.group_size + lane_id);
      if (tx == 0) {
        partial_rect_reduced_lse[reduced_lse_off] =
            state_valid ? (state_m + log2f(state_d)) * kLn2 : -kInf;
      }
      float inv_d = state_valid ? ptx_rcp_approx(state_d) : 0.0f;
      int64_t o_base = reduced_lse_off * p.D + dim_base;
      if (p.partial_o_bf16) {
        Bf16x8 out_pack;
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          out_pack.v[i] = float_to_bf16(state_valid ? acc[i] * inv_d : 0.0f);
        }
        store_bf16x8(reinterpret_cast<__nv_bfloat16*>(partial_rect_reduced_o) + o_base,
                     out_pack);
      } else {
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          int d = dim_base + i;
          reinterpret_cast<float*>(partial_rect_reduced_o)[reduced_lse_off * p.D + d] =
              state_valid ? acc[i] * inv_d : 0.0f;
        }
      }
    }
    __syncthreads();
  }
}

__device__ void phase_reduce_mixed_rect_head_vec(
    const Params& p,
    const int32_t* __restrict__ group_mode,
    const int32_t* __restrict__ group_rect_begin,
    const int32_t* __restrict__ group_rect_end,
    const int32_t* __restrict__ head_rect_start,
    const int32_t* __restrict__ head_new_end,
    int32_t* __restrict__ group_rect_tiles,
    const int32_t* __restrict__ task_counts,
    const void* __restrict__ partial_rect_o,
    const float* __restrict__ partial_rect_lse,
    void* __restrict__ partial_rect_reduced_o,
    float* __restrict__ partial_rect_reduced_lse) {
  if (!mixed_head_reduce_enabled(p)) return;

  constexpr int kBdx = 16;
  constexpr int kBdy = 8;
  constexpr int kVec = 8;
  const int tx = threadIdx.x & (kBdx - 1);
  const int ty = threadIdx.x >> 4;
  if (ty >= kBdy) return;

  __shared__ float sh_o[kBdy * kHeadDim];
  __shared__ float sh_m[kBdy];
  __shared__ float sh_d[kBdy];

  const int max_reduce_chunks =
      (task_counts != nullptr && task_counts[8] > 0) ? task_counts[8] : 0;
  if (max_reduce_chunks <= 0) return;
  const int total = p.N * p.Hq * max_reduce_chunks;
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int chunk = task % max_reduce_chunks;
    int ho_i = task / max_reduce_chunks;
    int hq = ho_i % p.Hq;
    int n = ho_i / p.Hq;
    int hkv = hq / p.group_size;
    int lane_id = hq - hkv * p.group_size;
    int64_t go = static_cast<int64_t>(n) * p.Hkv + hkv;
    if (group_mode[go] != kGroupModeMixedFallback) continue;

    int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
    int head_begin = head_rect_start[ho];
    int head_end = head_new_end[ho];
    if (head_end <= head_begin) continue;

    int group_begin = group_rect_begin[go];
    int group_end = group_rect_end[go];
    int group_tokens = group_end - group_begin;
    int tile_tokens = producer_rect_tile_tokens_for(p, task_counts, group_mode[go], group_tokens);
    int orig_tiles = group_tokens > 0 ? ceil_div_i32(group_tokens, tile_tokens) : 0;
    if (orig_tiles <= rect_reduce_threshold_for_counts(p, group_mode[go], task_counts)) continue;
    int reduce_chunk = clamp_reduce_chunk_to_workspace(
        balanced_rect_reduce_chunk_for_mode(p, group_mode[go], task_counts), orig_tiles,
        p.max_tiles_reduce);
    int reduced_tiles = ceil_div_i32(orig_tiles, reduce_chunk);
    if (chunk >= reduced_tiles) continue;
    if (chunk == 0 && lane_id == 0 && threadIdx.x == 0) {
      group_rect_tiles[go] = reduced_tiles;
    }

    int tile_begin = chunk * reduce_chunk;
    int tile_end = min(tile_begin + reduce_chunk, orig_tiles);
    int chunk_begin_pos = group_begin + tile_begin * tile_tokens;
    int chunk_end_pos = min(group_begin + tile_end * tile_tokens, group_end);
    if (chunk_end_pos <= head_begin || chunk_begin_pos >= head_end) continue;

    int dim_base = tx * kVec;
    float local_o[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      local_o[i] = 0.0f;
    }
    float local_m_log2 = -kInf;
    float local_d = 0.0f;
    bool local_valid = false;

    for (int tile = tile_begin + ty; tile < tile_end; tile += kBdy) {
      int begin = group_begin + tile * tile_tokens;
      int end = min(begin + tile_tokens, group_end);
      if (end <= head_begin || begin >= head_end) continue;
      int64_t lse_off =
          (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context + tile) *
           p.group_size + lane_id);
      float other_lse = partial_rect_lse[lse_off];
      if (!isfinite(other_lse)) continue;
      float other_m_log2 = other_lse * kLog2e;
      float new_m_log2 = local_valid ? fmaxf(local_m_log2, other_m_log2) : other_m_log2;
      float scale_state = local_valid ? ptx_exp2_approx(local_m_log2 - new_m_log2) : 0.0f;
      float scale_other = ptx_exp2_approx(other_m_log2 - new_m_log2);
      int64_t o_base = lse_off * p.D + dim_base;
      if (p.partial_o_bf16) {
        Bf16x8 other_pack =
            load_bf16x8(reinterpret_cast<const __nv_bfloat16*>(partial_rect_o) + o_base);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          float other = bf16_to_float(other_pack.v[i]);
          local_o[i] = scale_state * local_o[i] + scale_other * other;
        }
      } else {
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          int d = dim_base + i;
          float other = reinterpret_cast<const float*>(partial_rect_o)[lse_off * p.D + d];
          local_o[i] = scale_state * local_o[i] + scale_other * other;
        }
      }
      local_d = local_d * scale_state + scale_other;
      local_m_log2 = new_m_log2;
      local_valid = true;
    }

    if (tx == 0) {
      sh_m[ty] = local_valid ? local_m_log2 : -kInf;
      sh_d[ty] = local_valid ? local_d : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      int d = dim_base + i;
      sh_o[ty * kHeadDim + d] = local_valid ? local_o[i] : 0.0f;
    }
    __syncthreads();

    if (ty == 0) {
      float acc[kVec];
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        acc[i] = 0.0f;
      }
      float state_m_log2 = -kInf;
      float state_d = 0.0f;
      bool state_valid = false;

#pragma unroll
      for (int part = 0; part < kBdy; ++part) {
        float other_m_log2 = sh_m[part];
        float other_d = sh_d[part];
        if (!(isfinite(other_m_log2) && other_d > 0.0f)) continue;
        float new_m_log2 = state_valid ? fmaxf(state_m_log2, other_m_log2) : other_m_log2;
        float scale_state = state_valid ? ptx_exp2_approx(state_m_log2 - new_m_log2) : 0.0f;
        float scale_other = ptx_exp2_approx(other_m_log2 - new_m_log2);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          int d = dim_base + i;
          float other = sh_o[part * kHeadDim + d];
          acc[i] = scale_state * acc[i] + scale_other * other;
        }
        state_d = state_d * scale_state + other_d * scale_other;
        state_m_log2 = new_m_log2;
        state_valid = true;
      }

      int64_t reduced_lse_off =
          (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_reduce + chunk) *
           p.group_size + lane_id);
      if (tx == 0) {
        partial_rect_reduced_lse[reduced_lse_off] =
            state_valid ? (state_m_log2 + log2f(state_d)) * kLn2 : -kInf;
      }
      float inv_d = state_valid ? ptx_rcp_approx(state_d) : 0.0f;
      int64_t o_base = reduced_lse_off * p.D + dim_base;
      if (p.partial_o_bf16) {
        Bf16x8 out_pack;
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          out_pack.v[i] = float_to_bf16(state_valid ? acc[i] * inv_d : 0.0f);
        }
        store_bf16x8(reinterpret_cast<__nv_bfloat16*>(partial_rect_reduced_o) + o_base,
                     out_pack);
      } else {
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          int d = dim_base + i;
          reinterpret_cast<float*>(partial_rect_reduced_o)[reduced_lse_off * p.D + d] =
              state_valid ? acc[i] * inv_d : 0.0f;
        }
      }
    }
    __syncthreads();
  }
}

__device__ void phase_merge_full_fallback_group(
    const Params& p,
    const __nv_bfloat16* __restrict__ q_post,
    const __nv_bfloat16* __restrict__ q_pre,
    const __nv_bfloat16* __restrict__ k_buffer,
    const __nv_bfloat16* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    __nv_bfloat16* __restrict__ query_cache,
    __nv_bfloat16* __restrict__ attn_cache,
    float* __restrict__ lse_cache,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ optional_lse,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    const int32_t* __restrict__ group_rect_begin,
    const int32_t* __restrict__ group_rect_end,
    const int32_t* __restrict__ group_rect_tiles,
    const int32_t* __restrict__ group_mode,
    const int32_t* __restrict__ group_tail_begin,
    const int32_t* __restrict__ group_tail_end,
    const int32_t* __restrict__ group_tail_tiles,
    const int32_t* __restrict__ task_counts,
    const void* __restrict__ partial_rect_o,
    const float* __restrict__ partial_rect_lse,
    const void* __restrict__ partial_rect_reduced_o,
    const float* __restrict__ partial_rect_reduced_lse) {
  if (p.full_fallback_group_merge == 0 || p.group_size != 4 || blockDim.x < 128 ||
      p.fuse_fallback_tail_in_merge == 0) {
    return;
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= 4) return;
  const unsigned mask = 0xffffffffu;
  const int total = p.N * p.Hkv;

  for (int go_i = blockIdx.x; go_i < total; go_i += gridDim.x) {
    int64_t go = static_cast<int64_t>(go_i);
    if (group_mode[go] != kGroupModeFullFallback) continue;
    int tail_tiles = group_tail_tiles[go];
    if (tail_tiles > kMaxFuseTailTilesInMerge) continue;

    int hkv = go_i % p.Hkv;
    int n = go_i / p.Hkv;
    int hq = hkv * p.group_size + warp;
    int lane_id = warp;
    int req = req_ids[n];
    int past_len = past_lens[n];
    int dest_slot = past_len % p.M;
    int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;

    int dims[4];
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      dims[i] = lane + i * 32;
      acc[i] = 0.0f;
    }
    float state_lse = -kInf;
    bool state_valid = false;

    int rect_tiles = group_rect_tiles[go];
    int original_rect_tokens = group_rect_end[go] - group_rect_begin[go];
    int rect_tile_tokens = producer_rect_tile_tokens_for(p, task_counts, group_mode[go], original_rect_tokens);
    int original_rect_tiles =
        original_rect_tokens > 0 ? ceil_div_i32(original_rect_tokens, rect_tile_tokens) : 0;
    int reduce_chunk = clamp_reduce_chunk_to_workspace(
        balanced_rect_reduce_chunk_for_mode(p, group_mode[go], task_counts), original_rect_tiles,
        p.max_tiles_reduce);
    bool use_reduced_rect =
        original_rect_tiles > rect_reduce_threshold_for_counts(p, group_mode[go], task_counts);
    int merge_tiles = use_reduced_rect ? ceil_div_i32(original_rect_tiles, reduce_chunk)
                                       : original_rect_tiles;
    if (use_reduced_rect && rect_tiles > 0 && merge_tiles > rect_tiles) {
      merge_tiles = rect_tiles;
    }

    for (int tile = 0; tile < merge_tiles; ++tile) {
      int64_t lse_off =
          use_reduced_rect
              ? (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_reduce + tile) *
                 p.group_size + lane_id)
              : (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context + tile) *
                 p.group_size + lane_id);
      float w_state = 1.0f;
      float w_other = 0.0f;
      int take_other_i = 0;
      int other_valid_i = 0;
      if (lane == 0) {
        float other_lse =
            use_reduced_rect ? partial_rect_reduced_lse[lse_off] : partial_rect_lse[lse_off];
        bool other_valid = isfinite(other_lse);
        bool take_other = false;
        merge_scalar(state_lse, state_valid, other_lse, w_state, w_other, take_other);
        take_other_i = take_other ? 1 : 0;
        other_valid_i = other_valid ? 1 : 0;
      }
      state_lse = __shfl_sync(mask, state_lse, 0);
      state_valid = __shfl_sync(mask, state_valid ? 1 : 0, 0) != 0;
      w_state = __shfl_sync(mask, w_state, 0);
      w_other = __shfl_sync(mask, w_other, 0);
      take_other_i = __shfl_sync(mask, take_other_i, 0);
      other_valid_i = __shfl_sync(mask, other_valid_i, 0);
      if (take_other_i || other_valid_i) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int d = dims[i];
          if (d < p.D) {
            float other = use_reduced_rect
                              ? load_partial_o(partial_rect_reduced_o, lse_off * p.D + d,
                                               p.partial_o_bf16)
                              : load_partial_o(partial_rect_o, lse_off * p.D + d,
                                               p.partial_o_bf16);
            acc[i] = take_other_i ? other : (w_state * acc[i] + w_other * other);
          }
        }
      }
    }

    float cache_lse = state_lse;
    bool cache_valid = state_valid;
    int64_t q_cache_base = (((static_cast<int64_t>(req) * p.M + dest_slot) * p.Hq + hq) * p.D);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = dims[i];
      if (d < p.D) {
        query_cache[q_cache_base + d] =
            q_pre[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d];
        attn_cache[q_cache_base + d] = cache_valid ? float_to_bf16(acc[i]) : float_to_bf16(0.0f);
      }
    }
    if (lane == 0) {
      lse_cache[(static_cast<int64_t>(req) * p.M + dest_slot) * p.Hq + hq] =
          cache_valid ? cache_lse : -kInf;
    }

    float q_vals[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = dims[i];
      q_vals[i] = 0.0f;
      if (d < p.D) {
        q_vals[i] = bf16_to_float(q_post[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d]);
      }
    }

    int tail_begin_base = group_tail_begin[go];
    int tail_end_all = group_tail_end[go];
    for (int tile = 0; tile < tail_tiles; ++tile) {
        int begin = tail_begin_base + tile * p.tile_tokens;
        int end = min(begin + p.tile_tokens, tail_end_all);
        float tile_m = -kInf;
        float tile_denom = 0.0f;
        bool tile_valid = false;
        float tile_vals[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          tile_vals[i] = 0.0f;
        }
        for (int pos = begin; pos < end; ++pos) {
          int phys = physical_token(n, req, pos, past_len, req_to_token, p.req_to_token_stride,
                                    out_cache_loc_i32, out_cache_loc_i64,
                                    p.out_cache_loc_is_i64);
          float dot = 0.0f;
          float v_vals[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int d = dims[i];
            v_vals[i] = 0.0f;
            if (d < p.D) {
              int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + d;
              dot = fmaf(q_vals[i], bf16_to_float(k_buffer[off]), dot);
              v_vals[i] = bf16_to_float(v_buffer[off]);
            }
          }
          dot = warp_sum(dot);
          float score = __shfl_sync(mask, dot, 0) * p.sm_scale;
          float new_m = fmaxf(tile_m, score);
          float alpha = tile_valid ? expf(tile_m - new_m) : 0.0f;
          float beta = expf(score - new_m);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int d = dims[i];
            if (d < p.D) {
              tile_vals[i] = tile_vals[i] * alpha + beta * v_vals[i];
            }
          }
          tile_denom = tile_denom * alpha + beta;
          tile_m = new_m;
          tile_valid = true;
        }

        float other_lse = tile_denom > 0.0f ? (tile_m + logf(tile_denom)) : -kInf;
        float inv = tile_denom > 0.0f ? (1.0f / tile_denom) : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          tile_vals[i] *= inv;
        }
        float w_state = 1.0f;
        float w_other = 0.0f;
        bool take_other = false;
        bool other_valid = isfinite(other_lse);
        if (lane == 0) {
          merge_scalar(state_lse, state_valid, other_lse, w_state, w_other, take_other);
        }
        state_lse = __shfl_sync(mask, state_lse, 0);
        state_valid = __shfl_sync(mask, state_valid ? 1 : 0, 0) != 0;
        w_state = __shfl_sync(mask, w_state, 0);
        w_other = __shfl_sync(mask, w_other, 0);
        int take_other_i = __shfl_sync(mask, take_other ? 1 : 0, 0);
        int other_valid_i = __shfl_sync(mask, other_valid ? 1 : 0, 0);
        if (take_other_i || other_valid_i) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int d = dims[i];
            if (d < p.D) {
              acc[i] = take_other_i ? tile_vals[i] : (w_state * acc[i] + w_other * tile_vals[i]);
            }
          }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = dims[i];
      if (d < p.D) {
        out[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d] =
            state_valid ? float_to_bf16(acc[i]) : float_to_bf16(0.0f);
      }
    }
    if (optional_lse != nullptr && lane == 0) {
      optional_lse[ho] = state_valid ? state_lse : -kInf;
    }
  }
}

__device__ void phase4_merge_write(
    const Params& p,
    const __nv_bfloat16* __restrict__ q_post,
    const __nv_bfloat16* __restrict__ q_pre,
    const __nv_bfloat16* __restrict__ k_buffer,
    const __nv_bfloat16* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    __nv_bfloat16* __restrict__ query_cache,
    __nv_bfloat16* __restrict__ attn_cache,
    float* __restrict__ lse_cache,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ optional_lse,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    const int32_t* __restrict__ head_hit,
    const int32_t* __restrict__ head_match_slot,
    const int32_t* __restrict__ head_rect_start,
    const int32_t* __restrict__ head_new_end,
    const int32_t* __restrict__ group_rect_begin,
    const int32_t* __restrict__ group_rect_end,
    const int32_t* __restrict__ group_rect_tiles,
    const int32_t* __restrict__ group_mode,
    const int32_t* __restrict__ group_tail_begin,
    const int32_t* __restrict__ group_tail_end,
    const int32_t* __restrict__ group_tail_tiles,
    const int32_t* __restrict__ task_counts,
    const void* __restrict__ partial_rect_o,
    const float* __restrict__ partial_rect_lse,
    const void* __restrict__ partial_rect_reduced_o,
    const float* __restrict__ partial_rect_reduced_lse,
    const void* __restrict__ partial_tail_o,
    const float* __restrict__ partial_tail_lse) {
  const int total = p.N * p.Hq;
  __shared__ float sh_lse;
  __shared__ bool sh_valid;
  __shared__ float sh_w_state;
  __shared__ float sh_w_other;
  __shared__ bool sh_take_other;
  __shared__ bool sh_other_valid;
  __shared__ float sh_fused_tail_lse[kMaxFuseTailTilesInMerge];
  __shared__ float sh_fused_tail_o[kMaxFuseTailTilesInMerge * kHeadDim];

  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int hq = task % p.Hq;
    int n = task / p.Hq;
    int hkv = hq / p.group_size;
    int lane_id = hq - hkv * p.group_size;
    int req = req_ids[n];
    int dest_slot = past_lens[n] % p.M;
    int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
    int64_t go = static_cast<int64_t>(n) * p.Hkv + hkv;
    int64_t q_cache_base =
        (((static_cast<int64_t>(req) * p.M + dest_slot) * p.Hq + hq) * p.D);
    if (p.full_fallback_group_merge != 0 && p.fuse_fallback_tail_in_merge != 0 &&
        group_mode[go] == kGroupModeFullFallback &&
        group_tail_tiles[go] <= kMaxFuseTailTilesInMerge) {
      continue;
    }
    int dim = threadIdx.x;
    bool owns_dim = dim < p.D;
    float acc = 0.0f;
    bool hit = head_hit[ho] != 0;
    int match_slot = head_match_slot[ho];
    if (threadIdx.x == 0) {
      sh_lse = -kInf;
      sh_valid = false;
      if (hit && match_slot >= 0) {
        float prefix_lse = lse_cache[(static_cast<int64_t>(req) * p.M + match_slot) * p.Hq + hq];
        if (isfinite(prefix_lse)) {
          sh_lse = prefix_lse;
          sh_valid = true;
        }
      }
    }
    __syncthreads();

    bool state_valid = sh_valid;
    float state_lse = sh_lse;
    if (owns_dim && hit && match_slot >= 0 && state_valid) {
      int64_t off = (((static_cast<int64_t>(req) * p.M + match_slot) * p.Hq + hq) * p.D + dim);
      acc = bf16_to_float(attn_cache[off]);
    }
    __syncthreads();

    int rect_tiles = group_rect_tiles[go];
    bool per_head_rect = rect_tiles < 0;
    int rect_tile_tokens = p.tile_tokens;
    int original_rect_tokens = per_head_rect ? 0 : (group_rect_end[go] - group_rect_begin[go]);
    if (!per_head_rect) {
      rect_tile_tokens =
          producer_rect_tile_tokens_for(p, task_counts, group_mode[go], original_rect_tokens);
    }
    int original_rect_tiles =
        original_rect_tokens > 0 ? ceil_div_i32(original_rect_tokens, rect_tile_tokens) : 0;
    int reduce_chunk = clamp_reduce_chunk_to_workspace(
        balanced_rect_reduce_chunk_for_mode(p, group_mode[go], task_counts), original_rect_tiles,
        p.max_tiles_reduce);
    bool use_reduced_rect =
        !per_head_rect && group_mode[go] != kGroupModeMacHit &&
        original_rect_tiles > rect_reduce_threshold_for_counts(p, group_mode[go], task_counts);
    int rect_tile_begin = 0;
    int rect_tile_end = rect_tiles;
    if (per_head_rect) {
      int head_begin = head_rect_start[ho];
      int head_end = head_new_end[ho];
      int head_tiles = (head_end > head_begin) ? ceil_div_i32(head_end - head_begin, p.tile_tokens) : 0;
      use_reduced_rect = head_tiles > kRectReduceChunk;
      rect_tile_begin = 0;
      rect_tile_end = use_reduced_rect ? ceil_div_i32(head_tiles, reduce_chunk) : head_tiles;
    } else if (rect_tiles > 0) {
      int group_begin = group_rect_begin[go];
      int head_begin = head_rect_start[ho];
      int head_end = head_new_end[ho];
      if (head_end <= head_begin) {
        rect_tile_end = 0;
      } else {
        int rel_begin = head_begin - group_begin;
        int rel_end = head_end - group_begin;
        if (rel_begin < 0) rel_begin = 0;
        if (rel_end < 0) rel_end = 0;
        int orig_tile_begin = rel_begin / rect_tile_tokens;
        int orig_tile_end = ceil_div_i32(rel_end, rect_tile_tokens);
        if (use_reduced_rect) {
          rect_tile_begin = orig_tile_begin / reduce_chunk;
          rect_tile_end = ceil_div_i32(orig_tile_end, reduce_chunk);
        } else {
          rect_tile_begin = orig_tile_begin;
          rect_tile_end = orig_tile_end;
        }
        if (rect_tile_begin < 0) rect_tile_begin = 0;
        if (rect_tile_end > rect_tiles) rect_tile_end = rect_tiles;
      }
    }
    for (int tile = rect_tile_begin; tile < rect_tile_end; ++tile) {
      int64_t lse_off =
          use_reduced_rect
              ? (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_reduce + tile) *
                 p.group_size + lane_id)
              : (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context + tile) *
                 p.group_size + lane_id);
      if (threadIdx.x == 0) {
        float w_state = 1.0f, w_other = 0.0f;
        bool take_other = false;
        float other_lse =
            use_reduced_rect ? partial_rect_reduced_lse[lse_off] : partial_rect_lse[lse_off];
        bool other_valid = isfinite(other_lse);
        merge_scalar(state_lse, state_valid, other_lse, w_state, w_other, take_other);
        sh_lse = state_lse;
        sh_valid = state_valid;
        sh_w_state = w_state;
        sh_w_other = w_other;
        sh_take_other = take_other;
        sh_other_valid = other_valid;
      }
      __syncthreads();
      state_lse = sh_lse;
      state_valid = sh_valid;
      if (owns_dim && (sh_take_other || sh_other_valid)) {
        float other =
            use_reduced_rect
                ? load_partial_o(partial_rect_reduced_o, lse_off * p.D + dim, p.partial_o_bf16)
                : load_partial_o(partial_rect_o, lse_off * p.D + dim, p.partial_o_bf16);
        if (sh_take_other) {
          acc = other;
        } else {
          acc = sh_w_state * acc + sh_w_other * other;
        }
      }
      __syncthreads();
    }

    float cache_lse = state_lse;
    bool cache_valid = state_valid;
    if (owns_dim) {
      query_cache[q_cache_base + dim] = q_pre[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + dim];
      attn_cache[q_cache_base + dim] = cache_valid ? float_to_bf16(acc) : float_to_bf16(0.0f);
    }
    if (threadIdx.x == 0) {
      lse_cache[(static_cast<int64_t>(req) * p.M + dest_slot) * p.Hq + hq] =
          cache_valid ? cache_lse : -kInf;
    }
    __syncthreads();

    int tail_tiles = group_tail_tiles[go];
    bool group_full_fallback = group_mode[go] == kGroupModeFullFallback;
    bool group_mixed_fallback = group_mode[go] == kGroupModeMixedFallback;
    bool unfuse_sparse_tail = sparse_fallback_unfuse_tail_active(p, task_counts);
    bool fuse_mixed_tail =
        group_mixed_fallback && mixed_fallback_tail_fuse_active(p, tail_tiles);
    bool fuse_tail =
        !unfuse_sparse_tail && tail_tiles > 0 && tail_tiles <= kMaxFuseTailTilesInMerge &&
        ((hit && p.fuse_hit_tail_in_merge) ||
         (!hit && group_full_fallback && p.fuse_fallback_tail_in_merge) ||
         fuse_mixed_tail);
    if (fuse_tail) {
      int warp = threadIdx.x >> 5;
      int lane = threadIdx.x & 31;
      int warps = blockDim.x >> 5;
      float q_vals[4];
      int dims[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int d = lane + i * 32;
        dims[i] = d;
        q_vals[i] = 0.0f;
        if (d < p.D) {
          q_vals[i] = bf16_to_float(q_post[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d]);
        }
      }

      int tail_begin_base = group_tail_begin[go];
      int tail_end_all = group_tail_end[go];
      for (int tile = warp; tile < tail_tiles; tile += warps) {
        int begin = tail_begin_base + tile * p.tile_tokens;
        int end = min(begin + p.tile_tokens, tail_end_all);
        float tile_m = -kInf;
        float tile_denom = 0.0f;
        bool tile_valid = false;
        float tile_vals[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          tile_vals[i] = 0.0f;
        }
        for (int pos = begin; pos < end; ++pos) {
          int phys = physical_token(
              n, req, pos, past_lens[n], req_to_token, p.req_to_token_stride,
              out_cache_loc_i32, out_cache_loc_i64, p.out_cache_loc_is_i64);
          float dot = 0.0f;
          float v_vals[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int d = dims[i];
            v_vals[i] = 0.0f;
            if (d < p.D) {
              int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + d;
              dot = fmaf(q_vals[i], bf16_to_float(k_buffer[off]), dot);
              v_vals[i] = bf16_to_float(v_buffer[off]);
            }
          }
          dot = warp_sum(dot);
          float score = __shfl_sync(0xffffffffu, dot, 0) * p.sm_scale;
          float new_m = fmaxf(tile_m, score);
          float alpha = tile_valid ? expf(tile_m - new_m) : 0.0f;
          float beta = expf(score - new_m);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int d = dims[i];
            if (d < p.D) {
              tile_vals[i] = tile_vals[i] * alpha + beta * v_vals[i];
            }
          }
          tile_denom = tile_denom * alpha + beta;
          tile_m = new_m;
          tile_valid = true;
        }

        if (lane == 0) {
          sh_fused_tail_lse[tile] = tile_denom > 0.0f ? (tile_m + logf(tile_denom)) : -kInf;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int d = dims[i];
          if (d < p.D) {
            sh_fused_tail_o[tile * p.D + d] =
                tile_denom > 0.0f ? (tile_vals[i] / tile_denom) : 0.0f;
          }
        }
      }
      __syncthreads();

      for (int tile = 0; tile < tail_tiles; ++tile) {
        if (threadIdx.x == 0) {
          float w_state = 1.0f, w_other = 0.0f;
          bool take_other = false;
          float other_lse = sh_fused_tail_lse[tile];
          bool other_valid = isfinite(other_lse);
          merge_scalar(state_lse, state_valid, other_lse, w_state, w_other, take_other);
          sh_lse = state_lse;
          sh_valid = state_valid;
          sh_w_state = w_state;
          sh_w_other = w_other;
          sh_take_other = take_other;
          sh_other_valid = other_valid;
        }
        __syncthreads();
        state_lse = sh_lse;
        state_valid = sh_valid;
        if (owns_dim && (sh_take_other || sh_other_valid)) {
          float other = sh_fused_tail_o[tile * p.D + dim];
          if (sh_take_other) {
            acc = other;
          } else {
            acc = sh_w_state * acc + sh_w_other * other;
          }
        }
        __syncthreads();
      }

      if (owns_dim) {
        out[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + dim] =
            state_valid ? float_to_bf16(acc) : float_to_bf16(0.0f);
      }
      if (optional_lse != nullptr && threadIdx.x == 0) {
        optional_lse[ho] = state_valid ? state_lse : -kInf;
      }
    } else {
      for (int tile = 0; tile < tail_tiles; ++tile) {
        int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_tail + tile) *
                           p.group_size + lane_id);
        if (threadIdx.x == 0) {
          float w_state = 1.0f, w_other = 0.0f;
          bool take_other = false;
          float other_lse = partial_tail_lse[lse_off];
          bool other_valid = isfinite(other_lse);
          merge_scalar(state_lse, state_valid, other_lse, w_state, w_other, take_other);
          sh_lse = state_lse;
          sh_valid = state_valid;
          sh_w_state = w_state;
          sh_w_other = w_other;
          sh_take_other = take_other;
          sh_other_valid = other_valid;
        }
        __syncthreads();
        state_lse = sh_lse;
        state_valid = sh_valid;
        if (owns_dim && (sh_take_other || sh_other_valid)) {
          float other = load_partial_o(partial_tail_o, lse_off * p.D + dim, p.partial_o_bf16);
          if (sh_take_other) {
            acc = other;
          } else {
            acc = sh_w_state * acc + sh_w_other * other;
          }
        }
        __syncthreads();
      }

      if (owns_dim) {
        out[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + dim] =
            state_valid ? float_to_bf16(acc) : float_to_bf16(0.0f);
      }
      if (optional_lse != nullptr && threadIdx.x == 0) {
        optional_lse[ho] = state_valid ? state_lse : -kInf;
      }
    }
    __syncthreads();
  }
}

__global__ void mac_persistent_decode_kernel(
    Params p,
    const __nv_bfloat16* __restrict__ q_post,
    const __nv_bfloat16* __restrict__ q_pre,
    const __nv_bfloat16* __restrict__ k_buffer,
    const __nv_bfloat16* __restrict__ v_buffer,
    const int32_t* __restrict__ req_to_token,
    const int32_t* __restrict__ req_ids,
    const int32_t* __restrict__ past_lens,
    const int32_t* __restrict__ out_cache_loc_i32,
    const int64_t* __restrict__ out_cache_loc_i64,
    __nv_bfloat16* __restrict__ query_cache,
    __nv_bfloat16* __restrict__ attn_cache,
    float* __restrict__ lse_cache,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ optional_lse,
    float* __restrict__ match_dist,
    int32_t* __restrict__ match_pos,
    int32_t* __restrict__ match_slot,
    int32_t* __restrict__ head_hit,
    int32_t* __restrict__ head_match_slot,
    int32_t* __restrict__ head_match_pos,
    int32_t* __restrict__ head_prefix_end,
    int32_t* __restrict__ head_rect_start,
    int32_t* __restrict__ head_new_end,
    int32_t* __restrict__ group_rect_begin,
    int32_t* __restrict__ group_rect_end,
    int32_t* __restrict__ group_rect_tiles,
    int32_t* __restrict__ group_tail_begin,
    int32_t* __restrict__ group_tail_end,
    int32_t* __restrict__ group_tail_tiles,
    int32_t* __restrict__ group_mode,
    int32_t* __restrict__ complete_task_go,
    int32_t* __restrict__ complete_task_tile,
    int32_t* __restrict__ tail_task_go,
    int32_t* __restrict__ tail_task_tile,
    int32_t* __restrict__ hit_tail_task_ho,
    int32_t* __restrict__ hit_tail_task_tile,
    int32_t* __restrict__ task_counts,
    void* __restrict__ partial_rect_o,
    float* __restrict__ partial_rect_lse,
    void* __restrict__ partial_rect_reduced_o,
    float* __restrict__ partial_rect_reduced_lse,
    void* __restrict__ partial_tail_o,
    float* __restrict__ partial_tail_lse,
    int64_t* __restrict__ phase_cycles) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ uint4 dyn_smem_u4[];
  uint8_t* dyn_smem = reinterpret_cast<uint8_t*>(dyn_smem_u4);
  const bool record_phase_cycles = p.debug_enabled >= 2 && phase_cycles != nullptr;
  if (record_phase_cycles && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_cycles[0] = static_cast<int64_t>(clock64());
  }
  phase1_match_scan(p, q_pre, query_cache, req_ids, past_lens, match_dist, match_pos, match_slot);
  grid.sync();
  if (record_phase_cycles && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_cycles[1] = static_cast<int64_t>(clock64());
  }
  phase2_reduce_schedule(p, past_lens, req_ids, lse_cache, match_dist, match_pos, match_slot,
                         head_hit, head_match_slot, head_match_pos, head_prefix_end,
                         head_rect_start, head_new_end, group_rect_begin, group_rect_end,
                         group_rect_tiles, group_tail_begin, group_tail_end, group_tail_tiles,
                         group_mode, complete_task_go, complete_task_tile, tail_task_go,
                         tail_task_tile, hit_tail_task_ho, hit_tail_task_tile, task_counts, grid);
  grid.sync();
  if (record_phase_cycles && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_cycles[2] = static_cast<int64_t>(clock64());
  }
  if (task_counts[6] > 0 || mixed_group_direct_z2_enabled(p)) {
    phase_complete_attention_group_direct_z2(
        p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens, out_cache_loc_i32,
        out_cache_loc_i64, group_rect_begin, group_rect_end, group_rect_tiles, group_mode,
        head_hit, head_rect_start, head_new_end, complete_task_go, complete_task_tile, task_counts,
        partial_rect_o, partial_rect_lse, p.max_tiles_context, dyn_smem);
  }
  phase_attention_tiles_compact<false>(p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens,
                                       out_cache_loc_i32, out_cache_loc_i64,
                                       head_rect_start, head_new_end,
                                       group_rect_begin, group_rect_end, group_rect_tiles,
                                       group_mode,
                                       complete_task_go, complete_task_tile, task_counts,
                                       partial_rect_o, partial_rect_lse, p.max_tiles_context,
                                       false);
  phase_complete_attention_head_direct(p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens,
                                       out_cache_loc_i32, out_cache_loc_i64,
                                       head_rect_start, head_new_end,
                                       group_rect_begin, group_rect_end, group_rect_tiles,
                                       group_mode,
                                       complete_task_go, complete_task_tile, task_counts,
                                       partial_rect_o, partial_rect_lse, p.max_tiles_context);
  if (all_hit_direct_active(p, task_counts)) {
    phase_all_hit_complete_direct(p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens,
                                  out_cache_loc_i32, out_cache_loc_i64, head_rect_start, head_new_end,
                                  group_rect_begin, group_rect_end, group_rect_tiles, group_mode,
                                  partial_rect_o, partial_rect_lse, p.max_tiles_context);
  }
  if (record_phase_cycles && p.phase_cycles_count > 6 && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_cycles[6] = static_cast<int64_t>(clock64());
  }

  if (record_phase_cycles && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_cycles[3] = static_cast<int64_t>(clock64());
  }
  const bool tail_group_direct_z2_on = tail_group_direct_z2_active(p, task_counts);
  if (tail_group_direct_z2_on) {
    phase_tail_attention_group_direct_z2(p, q_post, k_buffer, v_buffer, req_to_token,
                                         req_ids, past_lens, out_cache_loc_i32,
                                         out_cache_loc_i64, group_tail_begin,
                                         group_tail_end, group_tail_tiles, group_mode,
                                         tail_task_go, tail_task_tile, task_counts,
                                         partial_tail_o, partial_tail_lse,
                                         p.max_tiles_tail, dyn_smem,
                                         tail_group_direct_z2_on);
  }
  phase_attention_tiles_compact<true>(p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens,
                                      out_cache_loc_i32, out_cache_loc_i64,
                                      head_rect_start, head_new_end,
                                      group_tail_begin, group_tail_end, group_tail_tiles,
                                      group_mode,
                                      tail_task_go, tail_task_tile, task_counts,
                                      partial_tail_o, partial_tail_lse, p.max_tiles_tail,
                                      tail_group_direct_z2_on);
  phase_tail_attention_head_direct(p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens,
                                   out_cache_loc_i32, out_cache_loc_i64, group_tail_begin,
                                   group_tail_end, hit_tail_task_ho, hit_tail_task_tile,
                                   task_counts, partial_tail_o, partial_tail_lse,
                                   p.max_tiles_tail);
  if (all_hit_direct_active(p, task_counts)) {
    phase_all_hit_tail_direct(p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens,
                              out_cache_loc_i32, out_cache_loc_i64, group_tail_begin,
                              group_tail_end, group_tail_tiles, group_mode,
                              partial_tail_o, partial_tail_lse, p.max_tiles_tail);
  }
  grid.sync();
  if (record_phase_cycles && blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t tail_done = static_cast<int64_t>(clock64());
    if (p.phase_cycles_count > 13) {
      phase_cycles[13] = tail_done;
    } else if (p.phase_cycles_count > 8) {
      phase_cycles[8] = tail_done;
    }
  }

  if (task_counts[3] > 0) {
    if (record_phase_cycles && p.phase_cycles_count > 12 && blockIdx.x == 0 && threadIdx.x == 0) {
      phase_cycles[12] = static_cast<int64_t>(clock64());
    }
    bool use_head_vec_reduce = full_fallback_head_reduce_enabled(p);
    bool use_mixed_head_vec_reduce = mixed_head_reduce_enabled(p);
    if (task_counts[6] > 0 && use_head_vec_reduce) {
      phase_reduce_full_fallback_head_vec(
          p, group_mode, group_rect_begin, group_rect_end, group_rect_tiles, task_counts,
          partial_rect_o, partial_rect_lse, partial_rect_reduced_o, partial_rect_reduced_lse);
    } else if (task_counts[6] > 0) {
      phase_reduce_full_fallback_group_warp(
          p, group_mode, group_rect_begin, group_rect_end, group_rect_tiles, task_counts,
          partial_rect_o, partial_rect_lse, partial_rect_reduced_o, partial_rect_reduced_lse);
    }
    if (use_mixed_head_vec_reduce) {
      phase_reduce_mixed_rect_head_vec(
          p, group_mode, group_rect_begin, group_rect_end, head_rect_start, head_new_end,
          group_rect_tiles, task_counts, partial_rect_o, partial_rect_lse,
          partial_rect_reduced_o, partial_rect_reduced_lse);
    }
    bool generic_needs_full_groups =
        task_counts[6] > 0 && !full_fallback_warp_reduce_enabled(p);
    bool generic_needs_mixed_groups =
        task_counts[3] > task_counts[6] && !use_mixed_head_vec_reduce;
    if (generic_needs_full_groups || generic_needs_mixed_groups) {
      phase_reduce_full_fallback_rect(p, group_mode, group_rect_begin, group_rect_end,
                                      head_rect_start, head_new_end,
                                      group_rect_tiles, task_counts, partial_rect_o,
                                      partial_rect_lse, partial_rect_reduced_o,
                                      partial_rect_reduced_lse);
    }
    grid.sync();
  }
  if (record_phase_cycles && p.phase_cycles_count > 7 && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_cycles[7] = static_cast<int64_t>(clock64());
  }
  if (record_phase_cycles && p.phase_cycles_count > 8 && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_cycles[8] = static_cast<int64_t>(clock64());
  }
  phase_merge_full_fallback_group(p, q_post, q_pre, k_buffer, v_buffer, req_to_token,
                                  query_cache, attn_cache, lse_cache, out, optional_lse,
                                  req_ids, past_lens, out_cache_loc_i32, out_cache_loc_i64,
                                  group_rect_begin, group_rect_end, group_rect_tiles, group_mode,
                                  group_tail_begin, group_tail_end, group_tail_tiles,
                                  task_counts, partial_rect_o, partial_rect_lse, partial_rect_reduced_o,
                                  partial_rect_reduced_lse);
  if (record_phase_cycles && p.phase_cycles_count > 9 && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_cycles[9] = static_cast<int64_t>(clock64());
  }
  if (record_phase_cycles && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_cycles[4] = static_cast<int64_t>(clock64());
  }
  phase4_merge_write(p, q_post, q_pre, k_buffer, v_buffer, req_to_token, query_cache, attn_cache,
                     lse_cache, out, optional_lse, req_ids, past_lens, out_cache_loc_i32,
                     out_cache_loc_i64, head_hit, head_match_slot, head_rect_start, head_new_end,
                     group_rect_begin, group_rect_end, group_rect_tiles, group_mode,
                     group_tail_begin, group_tail_end, group_tail_tiles, task_counts, partial_rect_o,
                     partial_rect_lse, partial_rect_reduced_o, partial_rect_reduced_lse, partial_tail_o,
                     partial_tail_lse);
  if (record_phase_cycles && p.phase_cycles_count > 10 && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_cycles[10] = static_cast<int64_t>(clock64());
  }
  if (record_phase_cycles) {
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      if (p.phase_cycles_count > 11) {
        phase_cycles[11] = static_cast<int64_t>(clock64());
      }
      phase_cycles[5] = static_cast<int64_t>(clock64());
    }
  }
}

void check_tensor(const Tensor& t, const char* name, at::ScalarType dtype, int dim) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dtype() == dtype, name, " has wrong dtype");
  TORCH_CHECK(t.dim() == dim, name, " has wrong rank");
}

void check_partial_o_tensor(const Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16,
              name, " must be float32 or bfloat16");
  TORCH_CHECK(t.dim() == 5, name, " has wrong rank");
}

int env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (!raw) return fallback;
  int v = std::atoi(raw);
  return v > 0 ? v : fallback;
}

int env_flag(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (!raw) return fallback;
  return std::atoi(raw) != 0 ? 1 : 0;
}

}  // namespace

void mac_persistent_decode_bf16(
    Tensor q_post,
    Tensor q_pre,
    Tensor k_buffer,
    Tensor v_buffer,
    Tensor req_to_token,
    Tensor req_ids,
    Tensor past_lens,
    Tensor out_cache_loc,
    Tensor query_cache,
    Tensor attn_cache,
    Tensor lse_cache,
    Tensor out,
    Tensor optional_lse,
    Tensor match_dist,
    Tensor match_pos,
    Tensor match_slot,
    Tensor head_hit,
    Tensor head_match_slot,
    Tensor head_match_pos,
    Tensor head_prefix_end,
    Tensor head_rect_start,
    Tensor head_new_end,
    Tensor group_rect_begin,
    Tensor group_rect_end,
    Tensor group_rect_tiles,
    Tensor group_tail_begin,
    Tensor group_tail_end,
    Tensor group_tail_tiles,
    Tensor group_mode,
    Tensor complete_task_go,
    Tensor complete_task_tile,
    Tensor tail_task_go,
    Tensor tail_task_tile,
    Tensor hit_tail_task_ho,
    Tensor hit_tail_task_tile,
    Tensor task_counts,
    Tensor partial_rect_o,
    Tensor partial_rect_lse,
    Tensor partial_rect_reduced_o,
    Tensor partial_rect_reduced_lse,
    Tensor partial_tail_o,
    Tensor partial_tail_lse,
    Tensor phase_cycles,
    int64_t max_context,
    int64_t tile_tokens,
    int64_t match_tile_slots,
    int64_t semantic_pos_ahead,
    int64_t gen_min_limit,
    int64_t lookback_right,
    int64_t candidate_mode,
    double threshold,
    double sm_scale,
    int64_t debug_enabled,
    int64_t bench_mode,
    double bench_hit_rate,
    double bench_hit_rate_std,
    double bench_skip_ratio,
    double bench_skip_ratio_std,
    double bench_match_lag_mean,
    double bench_match_lag_std,
    int64_t bench_seed,
    int64_t bench_miss_mask,
    int64_t bench_layer_id) {
  check_tensor(q_post, "q_post", at::kBFloat16, 3);
  check_tensor(q_pre, "q_pre", at::kBFloat16, 3);
  check_tensor(k_buffer, "k_buffer", at::kBFloat16, 3);
  check_tensor(v_buffer, "v_buffer", at::kBFloat16, 3);
  check_tensor(req_to_token, "req_to_token", at::kInt, 2);
  check_tensor(req_ids, "req_ids", at::kInt, 1);
  check_tensor(past_lens, "past_lens", at::kInt, 1);
  CHECK_CUDA(out_cache_loc);
  CHECK_CONTIG(out_cache_loc);
  TORCH_CHECK(out_cache_loc.dim() == 1, "out_cache_loc has wrong rank");
  TORCH_CHECK(
      out_cache_loc.scalar_type() == at::kInt || out_cache_loc.scalar_type() == at::kLong,
      "out_cache_loc must be int32 or int64");
  check_tensor(query_cache, "query_cache", at::kBFloat16, 4);
  check_tensor(attn_cache, "attn_cache", at::kBFloat16, 4);
  check_tensor(lse_cache, "lse_cache", at::kFloat, 3);
  check_tensor(out, "out", at::kBFloat16, 3);
  CHECK_CUDA(optional_lse);
  CHECK_CONTIG(optional_lse);
  TORCH_CHECK(optional_lse.numel() == 0 || optional_lse.scalar_type() == at::kFloat,
              "optional_lse must be empty or float32");
  CHECK_CUDA(phase_cycles);
  CHECK_CONTIG(phase_cycles);
  TORCH_CHECK(phase_cycles.scalar_type() == at::kLong, "phase_cycles must be int64");
  TORCH_CHECK(phase_cycles.numel() >= 6, "phase_cycles must have at least 6 elements");
  check_tensor(group_mode, "group_mode", at::kInt, 2);
  check_tensor(complete_task_go, "complete_task_go", at::kInt, 1);
  check_tensor(complete_task_tile, "complete_task_tile", at::kInt, 1);
  check_tensor(tail_task_go, "tail_task_go", at::kInt, 1);
  check_tensor(tail_task_tile, "tail_task_tile", at::kInt, 1);
  check_tensor(hit_tail_task_ho, "hit_tail_task_ho", at::kInt, 1);
  check_tensor(hit_tail_task_tile, "hit_tail_task_tile", at::kInt, 1);
  check_tensor(task_counts, "task_counts", at::kInt, 1);
  check_partial_o_tensor(partial_rect_o, "partial_rect_o");
  check_tensor(partial_rect_lse, "partial_rect_lse", at::kFloat, 4);
  check_partial_o_tensor(partial_rect_reduced_o, "partial_rect_reduced_o");
  check_tensor(partial_rect_reduced_lse, "partial_rect_reduced_lse", at::kFloat, 4);
  check_partial_o_tensor(partial_tail_o, "partial_tail_o");
  check_tensor(partial_tail_lse, "partial_tail_lse", at::kFloat, 4);
  at::ScalarType partial_o_dtype = partial_rect_o.scalar_type();
  TORCH_CHECK(partial_rect_reduced_o.scalar_type() == partial_o_dtype &&
                  partial_tail_o.scalar_type() == partial_o_dtype,
              "all partial output workspaces must use the same dtype");

  int N = (int)q_post.size(0);
  int Hq = (int)q_post.size(1);
  int D = (int)q_post.size(2);
  int Hkv = (int)k_buffer.size(1);
  TORCH_CHECK(D == kHeadDim, "persistent decode v1 supports D=128 only");
  TORCH_CHECK(q_pre.sizes() == q_post.sizes(), "q_pre/q_post shape mismatch");
  TORCH_CHECK(v_buffer.sizes() == k_buffer.sizes(), "k/v shape mismatch");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be divisible by Hkv");
  int group_size = Hq / Hkv;
  TORCH_CHECK(group_size > 0 && group_size <= kMaxWarps, "unsupported GQA group size");
  TORCH_CHECK(req_ids.size(0) >= N && past_lens.size(0) >= N && out_cache_loc.size(0) >= N,
              "batch metadata too small");
  TORCH_CHECK(out.sizes() == q_post.sizes(), "out shape mismatch");

  int M = (int)query_cache.size(1);
  int R = (int)query_cache.size(0);
  TORCH_CHECK(query_cache.size(2) == Hq && query_cache.size(3) == D, "query_cache shape mismatch");
  TORCH_CHECK(attn_cache.sizes() == query_cache.sizes(), "attn_cache shape mismatch");
  TORCH_CHECK(lse_cache.size(0) == R && lse_cache.size(1) == M && lse_cache.size(2) == Hq,
              "lse_cache shape mismatch");

  int max_match_tiles = (int)match_dist.size(2);
  int max_tiles_context = (int)partial_rect_lse.size(2);
  int max_tiles_reduce = (int)partial_rect_reduced_lse.size(2);
  int max_tiles_tail = (int)partial_tail_lse.size(2);
  int max_complete_tasks = (int)complete_task_go.numel();
  int max_tail_tasks = (int)tail_task_go.numel();
  int max_hit_tail_tasks = (int)hit_tail_task_ho.numel();
  TORCH_CHECK(max_match_tiles >= (M + (int)match_tile_slots - 1) / (int)match_tile_slots,
              "match workspace too small");
  TORCH_CHECK(max_tiles_context >= ((int)max_context + (int)tile_tokens - 1) / (int)tile_tokens,
              "rect workspace too small");
  TORCH_CHECK(max_tiles_reduce >= ((int)max_context + (int)tile_tokens * kRectReduceChunk - 1) /
                                      ((int)tile_tokens * kRectReduceChunk),
              "reduced rect workspace too small");
  TORCH_CHECK(max_tiles_tail >= ((int)semantic_pos_ahead + (int)tile_tokens - 1) / (int)tile_tokens,
              "tail workspace too small");
  TORCH_CHECK(partial_rect_o.size(0) >= N && partial_rect_o.size(1) == Hkv &&
                  partial_rect_o.size(2) == max_tiles_context &&
                  partial_rect_o.size(3) == group_size &&
                  partial_rect_o.size(4) == D,
              "rect output workspace has wrong shape");
  TORCH_CHECK(partial_rect_lse.size(0) >= N && partial_rect_lse.size(1) == Hkv &&
                  partial_rect_lse.size(2) == max_tiles_context &&
                  partial_rect_lse.size(3) == group_size,
              "rect lse workspace has wrong shape");
  TORCH_CHECK(partial_rect_reduced_o.size(0) >= N && partial_rect_reduced_o.size(1) == Hkv &&
                  partial_rect_reduced_o.size(2) == max_tiles_reduce &&
                  partial_rect_reduced_o.size(3) == group_size &&
                  partial_rect_reduced_o.size(4) == D,
              "reduced rect output workspace has wrong shape");
  TORCH_CHECK(partial_rect_reduced_lse.size(0) >= N && partial_rect_reduced_lse.size(1) == Hkv &&
                  partial_rect_reduced_lse.size(2) == max_tiles_reduce &&
                  partial_rect_reduced_lse.size(3) == group_size,
              "reduced rect lse workspace has wrong shape");
  TORCH_CHECK(partial_tail_o.size(0) >= N && partial_tail_o.size(1) == Hkv &&
                  partial_tail_o.size(2) == max_tiles_tail &&
                  partial_tail_o.size(3) == group_size &&
                  partial_tail_o.size(4) == D,
              "tail output workspace has wrong shape");
  TORCH_CHECK(partial_tail_lse.size(0) >= N && partial_tail_lse.size(1) == Hkv &&
                  partial_tail_lse.size(2) == max_tiles_tail &&
                  partial_tail_lse.size(3) == group_size,
              "tail lse workspace has wrong shape");
  TORCH_CHECK(group_mode.size(0) >= N && group_mode.size(1) >= Hkv, "group_mode workspace too small");
  TORCH_CHECK(complete_task_tile.numel() == complete_task_go.numel(), "complete task shape mismatch");
  TORCH_CHECK(tail_task_tile.numel() == tail_task_go.numel(), "tail task shape mismatch");
  TORCH_CHECK(hit_tail_task_tile.numel() == hit_tail_task_ho.numel(), "hit tail task shape mismatch");
  TORCH_CHECK(task_counts.numel() >= 9, "task_counts must have at least nine elements");
  TORCH_CHECK(bench_mode == 0 || bench_mode == 1 || bench_mode == 2,
              "unsupported synthetic MAC bench mode");
  TORCH_CHECK(max_complete_tasks >= N * Hq * max_tiles_context, "complete task workspace too small");
  TORCH_CHECK(max_tail_tasks >= N * Hkv * max_tiles_tail, "tail task workspace too small");
  TORCH_CHECK(max_hit_tail_tasks >= N * Hq * max_tiles_tail, "hit tail task workspace too small");

  c10::cuda::CUDAGuard guard(q_post.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q_post.get_device()).stream();

  Params p{};
  p.N = N;
  p.Hq = Hq;
  p.Hkv = Hkv;
  p.D = D;
  p.group_size = group_size;
  p.M = M;
  p.R = R;
  p.req_to_token_stride = (int)req_to_token.size(1);
  p.max_context = (int)max_context;
  p.max_match_tiles = max_match_tiles;
  p.max_tiles_context = max_tiles_context;
  p.max_tiles_reduce = max_tiles_reduce;
  p.max_tiles_tail = max_tiles_tail;
  p.tile_tokens = (int)tile_tokens;
  p.stage_tokens = env_int("MAC_PERSISTENT_STAGE_TOKENS", 8);
  if (p.stage_tokens < 1) p.stage_tokens = 1;
  if (p.stage_tokens > kMaxStageTokens) p.stage_tokens = kMaxStageTokens;
  p.match_tile_slots = (int)match_tile_slots;
  p.match_early_exit = env_flag("MAC_PERSISTENT_MATCH_EARLY_EXIT", 1);
  p.semantic_pos_ahead = (int)semantic_pos_ahead;
  p.gen_min_limit = (int)gen_min_limit;
  p.lookback_right = (int)lookback_right;
  p.candidate_mode = (int)candidate_mode;
  p.debug_enabled = (int)debug_enabled;
  p.phase_cycles_count = (int)phase_cycles.numel();
  p.out_cache_loc_is_i64 = out_cache_loc.scalar_type() == at::kLong ? 1 : 0;
  p.group_rect_max_spread = env_int("MAC_PERSISTENT_GROUP_RECT_MAX_SPREAD", 1);
  p.fuse_hit_tail_in_merge = env_flag("MAC_FUSE_HIT_TAIL_IN_MERGE", 0);
  p.fuse_fallback_tail_in_merge = env_flag("MAC_FUSE_FALLBACK_TAIL_IN_MERGE", 0);
  p.fuse_mixed_fallback_tail_in_merge =
      env_flag("MAC_FUSE_MIXED_FALLBACK_TAIL_IN_MERGE", 1);
  p.mixed_group_fallback = env_flag("MAC_PERSISTENT_MIXED_GROUP_FALLBACK", 1);
  p.hit_tail_group = env_flag("MAC_PERSISTENT_HIT_TAIL_GROUP", 1);
  p.all_hit_direct = env_flag("MAC_PERSISTENT_ALL_HIT_DIRECT", 0);
  p.hit_complete_head_direct = env_flag("MAC_PERSISTENT_HIT_COMPLETE_HEAD_DIRECT", 1);
  p.full_fallback_group_direct = env_flag("MAC_PERSISTENT_FULL_FALLBACK_GROUP_DIRECT", 1);
  p.full_fallback_head_direct = env_flag("MAC_PERSISTENT_FULL_FALLBACK_HEAD_DIRECT", 1);
  p.full_fallback_group_merge = env_flag("MAC_PERSISTENT_FULL_FALLBACK_GROUP_MERGE", 0);
  p.full_fallback_head_reduce =
      env_flag("MAC_PERSISTENT_FULL_FALLBACK_HEAD_REDUCE", 1);
  p.full_fallback_warp_reduce =
      env_flag("MAC_PERSISTENT_FULL_FALLBACK_WARP_REDUCE", 1);
  p.mixed_head_reduce = env_flag("MAC_PERSISTENT_MIXED_HEAD_REDUCE", 1);
  p.full_fallback_dense_mixed_no_reduce =
      env_flag("MAC_PERSISTENT_FULL_FALLBACK_DENSE_MIXED_NO_REDUCE", 0);
  p.full_fallback_wide_mixed_no_reduce =
      env_flag("MAC_PERSISTENT_FULL_FALLBACK_WIDE_MIXED_NO_REDUCE", 0);
  p.mixed_group_direct_z2 = env_flag("MAC_PERSISTENT_MIXED_GROUP_DIRECT_Z2", 1);
  p.tail_group_direct_z2 = env_flag("MAC_PERSISTENT_TAIL_GROUP_DIRECT_Z2", 1);
  p.parallel_z2_schedule = env_flag("MAC_PERSISTENT_PARALLEL_Z2_SCHEDULE", 1);
  p.mixed_misspack_z2 = env_flag("MAC_PERSISTENT_MIXED_MISSPACK_Z2", 1);
  p.mixed_early_miss_direct = env_flag("MAC_PERSISTENT_MIXED_EARLY_MISS_DIRECT", 1);
  p.partial_o_bf16 = partial_o_dtype == at::kBFloat16 ? 1 : 0;
  p.long_rect_tile_tokens = env_int("MAC_PERSISTENT_LONG_RECT_TILE_TOKENS", 384);
  if (p.long_rect_tile_tokens < p.tile_tokens) p.long_rect_tile_tokens = p.tile_tokens;
  p.long_rect_min_tokens = env_int("MAC_PERSISTENT_LONG_RECT_MIN_TOKENS", 4096);
  p.full_fallback_min_chunk_tokens =
      env_int("MAC_PERSISTENT_FULL_FALLBACK_MIN_CHUNK_TOKENS", 256);
  p.full_fallback_target_ctas = env_int("MAC_PERSISTENT_FULL_FALLBACK_TARGET_CTAS", 512);
  p.full_fallback_dense_target_ctas =
      env_int("MAC_PERSISTENT_FULL_FALLBACK_DENSE_TARGET_CTAS", 512);
  p.full_fallback_per_mode_tiles =
      env_flag("MAC_PERSISTENT_FULL_FALLBACK_PER_MODE_TILES", 1);
  p.full_fallback_producer_coarsen =
      env_int("MAC_PERSISTENT_FULL_FALLBACK_PRODUCER_COARSEN", 1);
  if (p.full_fallback_producer_coarsen < 1) p.full_fallback_producer_coarsen = 1;
  if (p.full_fallback_producer_coarsen > 8) p.full_fallback_producer_coarsen = 8;
  p.mixed_fallback_heavy_target_ctas =
      env_int("MAC_PERSISTENT_MIXED_FALLBACK_HEAVY_TARGET_CTAS", 416);
  p.mixed_head_direct_target_ctas =
      env_int("MAC_PERSISTENT_MIXED_HEAD_DIRECT_TARGET_CTAS", 2048);
  p.mixed_head_broad_target_ctas =
      env_int("MAC_PERSISTENT_MIXED_HEAD_BROAD_TARGET_CTAS", 2048);
  p.mixed_head_dense_full_target_ctas =
      env_int("MAC_PERSISTENT_MIXED_HEAD_DENSE_FULL_TARGET_CTAS", 6144);
  p.full_fallback_partial_reduce_chunk =
      env_int("MAC_PERSISTENT_FULL_FALLBACK_PARTIAL_REDUCE_CHUNK",
              p.partial_o_bf16 ? kDenseFullFallbackReduceChunk
                                : kDenseFullFallbackReduceChunkFp32);
  if (p.full_fallback_partial_reduce_chunk < 1) p.full_fallback_partial_reduce_chunk = 1;
  if (p.full_fallback_partial_reduce_chunk > kFullFallbackReduceChunkMax) {
    p.full_fallback_partial_reduce_chunk = kFullFallbackReduceChunkMax;
  }
  p.full_fallback_multi_request_reduce_chunk =
      env_int("MAC_PERSISTENT_FULL_FALLBACK_MULTI_REQUEST_REDUCE_CHUNK",
              kDenseFullFallbackReduceChunkMultiRequest);
  if (p.full_fallback_multi_request_reduce_chunk < 1) {
    p.full_fallback_multi_request_reduce_chunk = 1;
  }
  if (p.full_fallback_multi_request_reduce_chunk > kFullFallbackReduceChunkMax) {
    p.full_fallback_multi_request_reduce_chunk = kFullFallbackReduceChunkMax;
  }
  p.mixed_reduce_chunk = env_int("MAC_PERSISTENT_MIXED_REDUCE_CHUNK", 64);
  if (p.mixed_reduce_chunk < 1) p.mixed_reduce_chunk = 1;
  if (p.mixed_reduce_chunk > kFullFallbackReduceChunkMax) {
    p.mixed_reduce_chunk = kFullFallbackReduceChunkMax;
  }
  p.mixed_no_full_reduce_chunk =
      env_int("MAC_PERSISTENT_MIXED_NO_FULL_REDUCE_CHUNK", 32);
  if (p.mixed_no_full_reduce_chunk < 1) p.mixed_no_full_reduce_chunk = 1;
  if (p.mixed_no_full_reduce_chunk > kFullFallbackReduceChunkMax) {
    p.mixed_no_full_reduce_chunk = kFullFallbackReduceChunkMax;
  }
  p.full_fallback_reduce_threshold =
      env_int("MAC_PERSISTENT_FULL_FALLBACK_REDUCE_THRESHOLD", 1);
  if (p.full_fallback_reduce_threshold < 1) p.full_fallback_reduce_threshold = 1;
  p.sparse_fallback_unfuse_tail =
      env_flag("MAC_PERSISTENT_SPARSE_FALLBACK_UNFUSE_TAIL", 1);
  int mixed_group_direct_default_min =
      (p.mixed_group_direct_z2 != 0 && p.group_size == 4 && p.D == kHeadDim) ? 1 : p.group_size;
  p.mixed_group_direct_min_miss_heads =
      env_int("MAC_PERSISTENT_MIXED_GROUP_DIRECT_MIN_MISS_HEADS", mixed_group_direct_default_min);
  p.bench_mode = (int)bench_mode;
  p.bench_exact_quota = env_flag("MAC_BENCH_EXACT_QUOTA", 1);
  p.bench_seed = (int)bench_seed;
  p.bench_layer_id = (int)bench_layer_id;
  p.bench_miss_mask = (int)bench_miss_mask;
  p.threshold = (float)threshold;
  p.threshold_distance = 2.0f * (float)D * (1.0f - p.threshold) * (1.0f - p.threshold);
  p.sm_scale = (float)sm_scale;
  p.bench_hit_rate = (float)bench_hit_rate;
  p.bench_hit_rate_std = (float)bench_hit_rate_std;
  p.bench_skip_ratio = (float)bench_skip_ratio;
  p.bench_skip_ratio_std = (float)bench_skip_ratio_std;
  p.bench_match_lag_mean = (float)bench_match_lag_mean;
  p.bench_match_lag_std = (float)bench_match_lag_std;

  int block_threads = env_int("MAC_PERSISTENT_BLOCK_THREADS", group_size > 4 ? 256 : 128);
  if (block_threads < group_size * 32) block_threads = group_size * 32;
  if (block_threads != 128 && block_threads != 256) block_threads = group_size > 4 ? 256 : 128;
  TORCH_CHECK(block_threads >= D, "persistent decode v1 merge requires at least one thread per head dim");
  bool needs_direct_z2_smem =
      p.group_size == 4 && p.D == kHeadDim && block_threads == 128 &&
      (p.full_fallback_group_direct != 0 || p.tail_group_direct_z2 != 0);
  int coop_dynamic_smem = needs_direct_z2_smem ? kDirectZ2SmemBytes : 0;
  if (coop_dynamic_smem > 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        mac_persistent_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        coop_dynamic_smem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        mac_persistent_decode_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  }

  int device = q_post.get_device();
  int coop = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(coop != 0, "device does not support cooperative launch");
  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int active_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_per_sm, mac_persistent_decode_kernel, block_threads, coop_dynamic_smem));
  TORCH_CHECK(active_per_sm > 0, "persistent kernel has zero occupancy");
  int max_grid = active_per_sm * sm_count;
  int default_cap = 1024 * (p.N > 0 ? p.N : 1);
  if (default_cap > max_grid) default_cap = max_grid;
  int cap = env_int("MAC_PERSISTENT_COOP_CTAS", default_cap);
  int grid = cap < max_grid ? cap : max_grid;
  if (grid < 1) grid = 1;

  const __nv_bfloat16* q_post_ptr = reinterpret_cast<const __nv_bfloat16*>(q_post.data_ptr<at::BFloat16>());
  const __nv_bfloat16* q_pre_ptr = reinterpret_cast<const __nv_bfloat16*>(q_pre.data_ptr<at::BFloat16>());
  const __nv_bfloat16* k_ptr = reinterpret_cast<const __nv_bfloat16*>(k_buffer.data_ptr<at::BFloat16>());
  const __nv_bfloat16* v_ptr = reinterpret_cast<const __nv_bfloat16*>(v_buffer.data_ptr<at::BFloat16>());
  const int32_t* req_to_token_ptr = req_to_token.data_ptr<int32_t>();
  const int32_t* req_ids_ptr = req_ids.data_ptr<int32_t>();
  const int32_t* past_lens_ptr = past_lens.data_ptr<int32_t>();
  const int32_t* out_cache_loc_i32_ptr =
      p.out_cache_loc_is_i64 ? nullptr : out_cache_loc.data_ptr<int32_t>();
  const int64_t* out_cache_loc_i64_ptr =
      p.out_cache_loc_is_i64 ? out_cache_loc.data_ptr<int64_t>() : nullptr;
  __nv_bfloat16* q_cache_ptr = reinterpret_cast<__nv_bfloat16*>(query_cache.data_ptr<at::BFloat16>());
  __nv_bfloat16* a_cache_ptr = reinterpret_cast<__nv_bfloat16*>(attn_cache.data_ptr<at::BFloat16>());
  float* lse_cache_ptr = lse_cache.data_ptr<float>();
  __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  float* optional_lse_ptr = optional_lse.numel() == 0 ? nullptr : optional_lse.data_ptr<float>();
  float* match_dist_ptr = match_dist.data_ptr<float>();
  int32_t* match_pos_ptr = match_pos.data_ptr<int32_t>();
  int32_t* match_slot_ptr = match_slot.data_ptr<int32_t>();
  int32_t* head_hit_ptr = head_hit.data_ptr<int32_t>();
  int32_t* head_match_slot_ptr = head_match_slot.data_ptr<int32_t>();
  int32_t* head_match_pos_ptr = head_match_pos.data_ptr<int32_t>();
  int32_t* head_prefix_end_ptr = head_prefix_end.data_ptr<int32_t>();
  int32_t* head_rect_start_ptr = head_rect_start.data_ptr<int32_t>();
  int32_t* head_new_end_ptr = head_new_end.data_ptr<int32_t>();
  int32_t* group_rect_begin_ptr = group_rect_begin.data_ptr<int32_t>();
  int32_t* group_rect_end_ptr = group_rect_end.data_ptr<int32_t>();
  int32_t* group_rect_tiles_ptr = group_rect_tiles.data_ptr<int32_t>();
  int32_t* group_tail_begin_ptr = group_tail_begin.data_ptr<int32_t>();
  int32_t* group_tail_end_ptr = group_tail_end.data_ptr<int32_t>();
  int32_t* group_tail_tiles_ptr = group_tail_tiles.data_ptr<int32_t>();
  int32_t* group_mode_ptr = group_mode.data_ptr<int32_t>();
  int32_t* complete_task_go_ptr = complete_task_go.data_ptr<int32_t>();
  int32_t* complete_task_tile_ptr = complete_task_tile.data_ptr<int32_t>();
  int32_t* tail_task_go_ptr = tail_task_go.data_ptr<int32_t>();
  int32_t* tail_task_tile_ptr = tail_task_tile.data_ptr<int32_t>();
  int32_t* hit_tail_task_ho_ptr = hit_tail_task_ho.data_ptr<int32_t>();
  int32_t* hit_tail_task_tile_ptr = hit_tail_task_tile.data_ptr<int32_t>();
  int32_t* task_counts_ptr = task_counts.data_ptr<int32_t>();
  void* partial_rect_o_ptr = partial_rect_o.data_ptr();
  float* partial_rect_lse_ptr = partial_rect_lse.data_ptr<float>();
  void* partial_rect_reduced_o_ptr = partial_rect_reduced_o.data_ptr();
  float* partial_rect_reduced_lse_ptr = partial_rect_reduced_lse.data_ptr<float>();
  void* partial_tail_o_ptr = partial_tail_o.data_ptr();
  float* partial_tail_lse_ptr = partial_tail_lse.data_ptr<float>();
  int64_t* phase_cycles_ptr = phase_cycles.data_ptr<int64_t>();

  void* kernel_args[] = {
      &p,
      &q_post_ptr,
      &q_pre_ptr,
      &k_ptr,
      &v_ptr,
      &req_to_token_ptr,
      &req_ids_ptr,
      &past_lens_ptr,
      &out_cache_loc_i32_ptr,
      &out_cache_loc_i64_ptr,
      &q_cache_ptr,
      &a_cache_ptr,
      &lse_cache_ptr,
      &out_ptr,
      &optional_lse_ptr,
      &match_dist_ptr,
      &match_pos_ptr,
      &match_slot_ptr,
      &head_hit_ptr,
      &head_match_slot_ptr,
      &head_match_pos_ptr,
      &head_prefix_end_ptr,
      &head_rect_start_ptr,
      &head_new_end_ptr,
      &group_rect_begin_ptr,
      &group_rect_end_ptr,
      &group_rect_tiles_ptr,
      &group_tail_begin_ptr,
      &group_tail_end_ptr,
      &group_tail_tiles_ptr,
      &group_mode_ptr,
      &complete_task_go_ptr,
      &complete_task_tile_ptr,
      &tail_task_go_ptr,
      &tail_task_tile_ptr,
      &hit_tail_task_ho_ptr,
      &hit_tail_task_tile_ptr,
      &task_counts_ptr,
      &partial_rect_o_ptr,
      &partial_rect_lse_ptr,
      &partial_rect_reduced_o_ptr,
      &partial_rect_reduced_lse_ptr,
      &partial_tail_o_ptr,
      &partial_tail_lse_ptr,
      &phase_cycles_ptr,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(mac_persistent_decode_kernel),
      dim3(grid),
      dim3(block_threads),
      kernel_args,
      coop_dynamic_smem,
      stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mac_persistent_decode_bf16", &mac_persistent_decode_bf16,
        "MAC-Attention persistent decode v1 (BF16, D=128, cooperative)");
}
