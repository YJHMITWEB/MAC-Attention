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
#include <map>
#include <mutex>
#include <utility>

using torch::Tensor;
namespace cg = cooperative_groups;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).dtype() == (dt), #x " has wrong dtype")

namespace {

constexpr int kHeadDim = 128;
constexpr int kMaxWarps = 8;
constexpr int kMaxGroupSize = 16;
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
// z2 group-direct producer layout: the block splits into 16-lane subwarps,
// one per (head-in-group, z-part). kStageTokens = kGroup * kZParts =
// blockDim / 16 tokens are staged per iteration and each z-part consumes its
// own kGroup-token half, so the stage barrier only spans one z-part.
// The (block_threads, group_size) -> kZParts support table lives in
// z2_parts_for_config(); scheduler emission, producer dispatch and the
// log2-domain fast reducers must all consult it (a mismatch silently drops
// tasks or mixes LSE domains).
__host__ __device__ __forceinline__ int z2_parts_for_config(int block_threads,
                                                            int group_size,
                                                            int head_dim) {
  if (head_dim != kHeadDim) return 0;
  if (block_threads == 128 && group_size == 4) return 2;
  if (block_threads == 256 && group_size == 8) return 2;
  // group_size > kMaxWarps is rejected at the launcher (per-head phases map
  // one warp per head); such models take the FlashInfer fallback in python.
  return 0;
}

// cp_async ring depth for the z2 producers. Two stages (8-16KB in flight per
// CTA) cannot cover HBM latency at 2 blocks/SM (block 256 measured ~30% of
// FlashInfer's streaming rate on full-fallback groups); four stages keep
// 32KB in flight. Prefetch depth only — the compute order and math are
// untouched.
// Per-variant: block 256 runs only 2 blocks/SM, so per-CTA in-flight depth
// carries the whole SM's memory parallelism (8 stages = 64KB in flight);
// block 128 runs 5 blocks/SM and a deeper ring would cost that occupancy
// (5 -> 4 at 8 stages), so it keeps 4.
constexpr int z2_pipeline_bufs(int stage_tokens) {
  // 8 deep for both block variants. Block 128 sits at 4 blocks/SM either way
  // (register-file granularity, not shared memory, is the occupancy limiter:
  // shrinking the pool to 16.1KB did NOT restore 5/SM but halving the ring
  // cost ~5-9% on fallback streaming), so keep the full ring depth.
  return 8;
}

constexpr int direct_z2_smem_bytes(int stage_tokens) {
  return z2_pipeline_bufs(stage_tokens) * 2 * stage_tokens * kHeadDim *
             static_cast<int>(sizeof(__nv_bfloat16)) +
         stage_tokens * kHeadDim * static_cast<int>(sizeof(float)) +
         stage_tokens * static_cast<int>(sizeof(float));
}

constexpr int cmax_i(int a, int b) { return a > b ? a : b; }

// One dynamic shared pool time-sliced by every phase that needs bulk shared
// memory. Declaring these arrays statically per phase made ptxas SUM them
// (~30KB static + 12KB dynamic per block), capping occupancy at 4 blocks/SM;
// pooling drops per-block shared memory to ~13KB. Phases touching the pool
// must be separated by grid.sync() or start with __syncthreads().
// Compact producer: 2 buffers (double-buffered cp_async) x {K, V} stages.
constexpr int kCompactStageSmemBytes =
    2 * 2 * kMaxStageTokens * kHeadDim * static_cast<int>(sizeof(__nv_bfloat16));
// Head-vec reducers stage blockDim/16 rows of one head vector each.
constexpr int head_vec_reduce_smem_bytes(int rows) {
  return (rows * kHeadDim + rows + rows) * static_cast<int>(sizeof(float));
}
constexpr int kMergeFusedTailSmemBytes =
    (kMaxFuseTailTilesInMerge * kHeadDim + kMaxFuseTailTilesInMerge) *
    static_cast<int>(sizeof(float));
// The z2 stage footprint and the head-vec row count both scale with
// blockDim/16, so the pool is sized per block variant (12.3KB z2 / 16.4KB
// compact at 128 threads; 24.6KB z2 at 256 threads).
constexpr int smem_pool_bytes(int block_threads) {
  return cmax_i(direct_z2_smem_bytes(block_threads / 16),
                cmax_i(kCompactStageSmemBytes,
                       cmax_i(head_vec_reduce_smem_bytes(block_threads / 16),
                              kMergeFusedTailSmemBytes)));
}

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
  // Tail tiles use their own (larger) granularity: the tail span is a fixed
  // ~semantic_pos_ahead tokens per group, and slicing it into tile_tokens
  // pieces made hundreds of tiny tasks whose fixed costs dominated.
  int32_t tail_tile_tokens;
  // Tail prepass: tail bounds depend only on past_len (never on match
  // results), and tail partial writes are idempotent stores, so the whole
  // tail phase can run alongside the latency-bound match scan in phase 1.
  // The scheduler then emits no tail tasks and phase 4 merges the partials.
  int32_t tail_prepass;
  int32_t stage_tokens;
  int32_t match_tile_slots;
  int32_t match_early_exit;
  int32_t semantic_pos_ahead;
  // Front-sink exclusion (v2): ring entries written with prefix end E cover
  // exactly [min(front_sink_tokens, E), E). The excluded sink region is
  // recomputed under the CURRENT query at every reuse (like the band) and
  // merged into the output only — never stored, never subtracted.
  int32_t front_sink_tokens;
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
  int32_t task_overhead_tokens;
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
  // Device-side readiness gate (CUDA-graph replay safe). Python cannot gate
  // the persistent path per step once the launch is baked into a graph, so
  // the kernel itself must decide per request: a request whose MAC cache is
  // not ready (prefill offload pending, epoch mismatch after slot reuse, or
  // low-hit ejection) runs as pure full-fallback attention and must neither
  // read match state nor write any cache row (the offload may be writing the
  // same rows concurrently). Null pointers mean "all requests ready" (bench
  // and check harnesses).
  const uint8_t* cache_ready_ptr;
  const int64_t* cache_epoch_ptr;
  const int64_t* request_epoch_ptr;
};

__device__ __forceinline__ bool request_persistent_ready(const Params& p, int req) {
  if (p.cache_ready_ptr == nullptr) return true;
  if (req < 0) return false;
  if (p.cache_ready_ptr[req] == 0) return false;
  if (p.cache_epoch_ptr != nullptr && p.request_epoch_ptr != nullptr &&
      p.cache_epoch_ptr[req] != p.request_epoch_ptr[req]) {
    return false;
  }
  return true;
}

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

// Stage barrier for one z-part of the z2 producers: z-part `tz` is the
// kGroup*16 contiguous threads that both load and consume stage slots
// [tz*kGroup, (tz+1)*kGroup), so only they need to converge between stages.
// Named barriers 1..kZParts (0 is __syncthreads); kZParts == 1 degenerates to
// a full block barrier.
template <int kGroup, int kZParts>
__device__ __forceinline__ void z2_stage_part_sync(int tz) {
#if defined(__CUDA_ARCH__)
  if constexpr (kZParts == 1) {
    __syncthreads();
    (void)tz;
  } else {
    constexpr int kPartThreads = kGroup * 16;
    static_assert(kZParts <= 8, "named-barrier budget");
    asm volatile("bar.sync %0, %1;\n" ::"r"(tz + 1), "n"(kPartThreads) : "memory");
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

// True when this launch has a z2 group-direct producer instantiation for
// p.group_size (see z2_parts_for_config). Every specialized full-fallback
// path hangs off this: the z2 producer emits full-fallback band partials in
// LOG2 domain when the warp/head-vec reducers are active, so the gates below
// must stay in lockstep with producer dispatch.
__device__ __forceinline__ bool group_direct_z2_supported(const Params& p) {
  return z2_parts_for_config(static_cast<int>(blockDim.x), p.group_size, p.D) > 0;
}

__device__ __forceinline__ bool tail_prepass_active(const Params& p) {
  return p.tail_prepass != 0 && p.tail_group_direct_z2 != 0 &&
         group_direct_z2_supported(p);
}

__device__ __forceinline__ bool full_fallback_warp_reduce_enabled(const Params& p) {
  return p.full_fallback_warp_reduce != 0 && p.full_fallback_group_direct != 0 &&
         group_direct_z2_supported(p) && blockDim.x >= 128;
}

__device__ __forceinline__ bool full_fallback_head_reduce_enabled(const Params& p) {
  return p.full_fallback_head_reduce != 0 && full_fallback_warp_reduce_enabled(p) &&
         (blockDim.x == 128 || blockDim.x == 256);
}

__device__ __forceinline__ bool mixed_head_reduce_enabled(const Params& p) {
  return p.mixed_head_reduce != 0 && p.D == kHeadDim &&
         (blockDim.x == 128 || blockDim.x == 256);
}

__device__ __forceinline__ bool mixed_group_direct_z2_enabled(const Params& p) {
  return p.mixed_group_direct_z2 != 0 && p.full_fallback_group_direct != 0 &&
         group_direct_z2_supported(p);
}

__device__ __forceinline__ bool mixed_misspack_z2_enabled(const Params& p) {
  return p.mixed_misspack_z2 != 0 && mixed_group_direct_z2_enabled(p);
}

__device__ __forceinline__ bool tail_group_direct_z2_enabled(const Params& p) {
  return p.tail_group_direct_z2 != 0 && group_direct_z2_supported(p);
}

__device__ __forceinline__ bool complete_group_direct_z2_selected(
    const Params& p, int mode, int miss_heads) {
  if (mode == kGroupModeFullFallback) {
    return p.full_fallback_group_direct != 0 && group_direct_z2_supported(p);
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
  // No mixed groups: the phase2 balancer wave-quantizes task_counts[4]
  // against the grid; this ramp-based override would undo that.
  if (task_counts[5] == task_counts[6]) {
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
  // Under the tail prepass the partials already exist; fusing would just
  // recompute the same tokens inline in the merge.
  return p.fuse_mixed_fallback_tail_in_merge != 0 &&
         p.fuse_hit_tail_in_merge != 0 && !tail_prepass_active(p) &&
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
  // No mixed groups: the wave-quantized balancer chunk is already sized to
  // the grid; coarsening would break its wave alignment.
  if (task_counts[5] == task_counts[6]) {
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

__device__ __forceinline__ int clamp_front_sink_end(const Params& p, int rect_end) {
  int end = p.front_sink_tokens;
  if (end < 0) end = 0;
  if (end > rect_end) end = rect_end;
  return end;
}

__device__ __forceinline__ int head_front_sink_end(const Params& p,
                                                   int head_rect_start,
                                                   int head_rect_end) {
  int end = clamp_front_sink_end(p, head_rect_end);
  if (end > head_rect_start) end = head_rect_start;
  return end > 0 ? end : 0;
}

__device__ __forceinline__ bool rect_pos_contributes_to_head(
    const Params& p, int pos, int head_rect_start, int head_rect_end) {
  int sink_end = head_front_sink_end(p, head_rect_start, head_rect_end);
  return (pos >= 0 && pos < sink_end) ||
         (pos >= head_rect_start && pos < head_rect_end);
}

__device__ __forceinline__ bool rect_tile_overlaps_head(
    const Params& p, int begin, int end, int head_rect_start, int head_rect_end) {
  if (end <= begin) return false;
  int sink_end = head_front_sink_end(p, head_rect_start, head_rect_end);
  bool sink_overlap = sink_end > 0 && begin < sink_end && end > 0;
  bool band_overlap = end > head_rect_start && begin < head_rect_end;
  return sink_overlap || band_overlap;
}

__device__ __forceinline__ int front_sink_tiles_for_group(
    const Params& p, int mode, int rect_end) {
  // Exclusion design: every group mode emits sink tiles [0, min(F, rect_end))
  // ahead of its band tiles; hit heads use them to complement the sink-excluded
  // reused entry, miss heads use them as the head of their full recompute.
  (void)mode;
  if (p.front_sink_tokens <= 0) return 0;
  int sink_end = clamp_front_sink_end(p, rect_end);
  return sink_end > 0 ? ceil_div_i32(sink_end, p.tile_tokens) : 0;
}

__device__ __forceinline__ int complete_tiles_for_group(
    const Params& p,
    const int32_t* __restrict__ task_counts,
    int mode,
    int group_begin,
    int group_end) {
  int rect_tokens = group_end - group_begin;
  int tile_tokens = producer_rect_tile_tokens_for(p, task_counts, mode, rect_tokens);
  int band_tiles = rect_tokens > 0 ? ceil_div_i32(rect_tokens, tile_tokens) : 0;
  return front_sink_tiles_for_group(p, mode, group_end) + band_tiles;
}

__device__ __forceinline__ bool complete_tile_bounds(
    const Params& p,
    const int32_t* __restrict__ task_counts,
    int mode,
    int group_begin,
    int group_end,
    int tile,
    int& begin,
    int& end,
    bool& is_front_sink_tile) {
  int sink_tiles = front_sink_tiles_for_group(p, mode, group_end);
  if (tile < sink_tiles) {
    begin = tile * p.tile_tokens;
    end = min(begin + p.tile_tokens, clamp_front_sink_end(p, group_end));
    is_front_sink_tile = true;
    return end > begin;
  }
  int rect_tokens = group_end - group_begin;
  int tile_tokens = producer_rect_tile_tokens_for(p, task_counts, mode, rect_tokens);
  int band_tile = tile - sink_tiles;
  begin = group_begin + band_tile * tile_tokens;
  end = min(begin + tile_tokens, group_end);
  is_front_sink_tile = false;
  return end > begin;
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

__device__ __forceinline__ void merge_partial_rect_tile_into_state(
    const Params& p,
    int64_t lse_off,
    bool use_reduced_rect,
    const void* __restrict__ partial_rect_o,
    const float* __restrict__ partial_rect_lse,
    const void* __restrict__ partial_rect_reduced_o,
    const float* __restrict__ partial_rect_reduced_lse,
    int dim,
    bool owns_dim,
    float& acc,
    float& state_lse,
    bool& state_valid,
    float* sh_lse,
    bool* sh_valid,
    float* sh_w_state,
    float* sh_w_other,
    bool* sh_take_other,
    bool* sh_other_valid) {
  if (threadIdx.x == 0) {
    float w_state = 1.0f, w_other = 0.0f;
    bool take_other = false;
    float other_lse =
        use_reduced_rect ? partial_rect_reduced_lse[lse_off] : partial_rect_lse[lse_off];
    bool other_valid = isfinite(other_lse);
    merge_scalar(state_lse, state_valid, other_lse, w_state, w_other, take_other);
    *sh_lse = state_lse;
    *sh_valid = state_valid;
    *sh_w_state = w_state;
    *sh_w_other = w_other;
    *sh_take_other = take_other;
    *sh_other_valid = other_valid;
  }
  __syncthreads();
  state_lse = *sh_lse;
  state_valid = *sh_valid;
  if (owns_dim && (*sh_take_other || *sh_other_valid)) {
    float other =
        use_reduced_rect
            ? load_partial_o(partial_rect_reduced_o, lse_off * p.D + dim, p.partial_o_bf16)
            : load_partial_o(partial_rect_o, lse_off * p.D + dim, p.partial_o_bf16);
    if (*sh_take_other) {
      acc = other;
    } else {
      acc = *sh_w_state * acc + *sh_w_other * other;
    }
  }
  __syncthreads();
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
  // v2: one warp scans two slots per iteration (one per 16-thread subwarp)
  // with fully vectorized 16B loads. When match_early_exit is set, a cheap
  // probe over the first 32 dims filters slots first: the partial sum of
  // squares is a lower bound on the full distance, so "probe > threshold"
  // can never discard a hit, and every surviving slot gets its exact full
  // distance. Unlike the old per-32-dim early exit, the probe loads carry no
  // serial dependency between slots, so the memory pipeline stays saturated.
  const int total = p.N * p.Hq * p.max_match_tiles;
  const int warps = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int half = lane >> 4;  // which slot of the pair this subwarp owns
  const int tx = lane & 15;
  const unsigned sub_mask = (lane & 16) ? 0xffff0000u : 0x0000ffffu;

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
    int req = req_ids[n];
    bool eligible = past_len >= p.gen_min_limit && candidate_end > candidate_begin &&
                    request_persistent_ready(p, req);

    float best_d = kInf;
    int best_p = INT_MAX;
    int best_s = -1;
    int slot_begin = tile * p.match_tile_slots;
    int slot_end = min(p.M, slot_begin + p.match_tile_slots);
    const int64_t q_base = (static_cast<int64_t>(n) * p.Hq + hq) * p.D;
    // Full-precision lane view: dims [tx*8, tx*8+8) (D == kHeadDim == 128).
    float q_full[8];
    {
      Bf16x8 qv = load_bf16x8(&q_pre[q_base + tx * 8]);
#pragma unroll
      for (int i = 0; i < 8; ++i) q_full[i] = bf16_to_float(qv.v[i]);
    }
    // Probe lane view: dims [tx*2, tx*2+2) covering [0, 32), plus a second
    // stage over [32, 64) for slots the first stage cannot separate from the
    // threshold. Both stages are partial sums of squared differences, i.e.
    // exact lower bounds on the full distance — pruning never drops a match
    // and accepted candidates still get the identical full evaluation.
    float q_probe[2];
    q_probe[0] = bf16_to_float(q_pre[q_base + tx * 2]);
    q_probe[1] = bf16_to_float(q_pre[q_base + tx * 2 + 1]);
    float q_probe2[2];
    q_probe2[0] = bf16_to_float(q_pre[q_base + 32 + tx * 2]);
    q_probe2[1] = bf16_to_float(q_pre[q_base + 32 + tx * 2 + 1]);
    const bool use_probe = p.match_early_exit != 0;
    const float thr = p.threshold_distance;
    const int64_t cache_base = (static_cast<int64_t>(req) * p.M) * p.Hq + hq;

    // Four slot pairs per iteration: all eight 4B probe loads (64 probe dims
    // per slot) are issued before any reduction, so the global-memory latency
    // overlaps the shuffle chains instead of serializing them — the match
    // phase is latency-bound, not bandwidth-bound. The probe remains an exact
    // lower bound and surviving slots take the identical full evaluation.
    for (int s0 = slot_begin; s0 < slot_end; s0 += 8) {
      uint32_t raw_a[4];
      uint32_t raw_b[4];
      bool pair_valid[4];
      int pair_pos[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        int slot = s0 + 2 * u + half;
        int logical_pos = ring_slot_to_pos(past_len, p.M, slot);
        bool valid = slot < slot_end && eligible && logical_pos >= candidate_begin &&
                     logical_pos < candidate_end;
        pair_valid[u] = valid;
        pair_pos[u] = logical_pos;
        raw_a[u] = 0u;
        raw_b[u] = 0u;
        if (use_probe && valid) {
          const int64_t c_off = (cache_base + static_cast<int64_t>(slot) * p.Hq) * p.D;
          raw_a[u] = *reinterpret_cast<const uint32_t*>(&query_cache[c_off + tx * 2]);
          raw_b[u] = *reinterpret_cast<const uint32_t*>(&query_cache[c_off + 32 + tx * 2]);
        }
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        int slot = s0 + 2 * u + half;
        bool valid = pair_valid[u];
        int logical_pos = pair_pos[u];
        const int64_t c_off = (cache_base + static_cast<int64_t>(slot) * p.Hq) * p.D;
        bool evaluate = valid;
        if (use_probe) {
          float probe = 0.0f;
          if (valid) {
            __nv_bfloat162 pa = *reinterpret_cast<const __nv_bfloat162*>(&raw_a[u]);
            __nv_bfloat162 pb = *reinterpret_cast<const __nv_bfloat162*>(&raw_b[u]);
            float d0 = q_probe[0] - bf16_to_float(pa.x);
            float d1 = q_probe[1] - bf16_to_float(pa.y);
            float e0 = q_probe2[0] - bf16_to_float(pb.x);
            float e1 = q_probe2[1] - bf16_to_float(pb.y);
            probe = fmaf(d0, d0, d1 * d1);
            probe = fmaf(e0, e0, probe);
            probe = fmaf(e1, e1, probe);
          }
          probe = subwarp16_xor_sum(probe, sub_mask);
          evaluate = valid && probe <= thr;
        }
        unsigned eval_mask = __ballot_sync(0xffffffffu, evaluate);
        if (eval_mask & sub_mask) {
          float acc = 0.0f;
          Bf16x8 cv = load_bf16x8(&query_cache[c_off + tx * 8]);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float diff = q_full[i] - bf16_to_float(cv.v[i]);
            acc = fmaf(diff, diff, acc);
          }
          acc = subwarp16_xor_sum(acc, sub_mask);
          // xor butterfly leaves the total on every lane of the subwarp, so
          // all lanes track an identical running best for their slot stream.
          if (valid && better_pair(acc, logical_pos, best_d, best_p)) {
            best_d = acc;
            best_p = logical_pos;
            best_s = slot;
          }
        }
      }
    }

    // Combine the two subwarp bests; every lane of a subwarp holds the same
    // value, so lane 16's copy represents the odd-slot stream.
    float other_d = __shfl_sync(0xffffffffu, best_d, 16);
    int other_p = __shfl_sync(0xffffffffu, best_p, 16);
    int other_s = __shfl_sync(0xffffffffu, best_s, 16);
    if (lane == 0) {
      if (better_pair(other_d, other_p, best_d, best_p)) {
        best_d = other_d;
        best_p = other_p;
        best_s = other_s;
      }
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
      bool eligible = past_len >= p.gen_min_limit && candidate_end > candidate_begin &&
                      request_persistent_ready(p, req_ids[n]);
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
      // Front-sink exclusion: an entry with prefix_end < F covers nothing (its
      // whole span lies inside the excluded sink region), and reusing it would
      // break the write-back coverage invariant [min(F, E), E). Demote to miss;
      // unreachable under production configs (matches sit at prefix_end >=
      // gen_min_limit - M - r, far above any practical F).
      if (hit && p.front_sink_tokens > 0 && prefix_end < p.front_sink_tokens) {
        hit = false;
        mpos = -1;
        mslot = -1;
        prefix_end = 0;
        rect_start = 0;
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
    // Front-sink exclusion: miss-head recompute is split into sink tiles
    // [0, fs_begin) plus band tiles [fs_begin, new_end); the sink part is
    // output-only so the written entry covers exactly [fs_begin, new_end).
    // fs_begin == 0 when front_sink_tokens == 0 (legacy behavior).
    int fs_begin = clamp_front_sink_end(p, new_end);
    if (mode == kGroupModeFullFallback) {
      complete_begin = fs_begin;
      for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
        int hq = hq0 + lane_i;
        int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
        head_hit[ho] = 0;
        head_match_slot[ho] = -1;
        head_match_pos[ho] = -1;
        head_prefix_end[ho] = 0;
        head_rect_start[ho] = fs_begin;
        head_new_end[ho] = rect_end;
      }
    } else if (mode == kGroupModeMixedFallback) {
      complete_begin = fs_begin;
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
        head_rect_start[ho] = fs_begin;
        head_new_end[ho] = new_end;
      }
    }

    complete_begin = max(0, min(complete_begin, rect_end));
    int rect_tokens = rect_end - complete_begin;
    int tail_tiles =
        (kv_len > tail_begin) ? ceil_div_i32(kv_len - tail_begin, p.tail_tile_tokens) : 0;
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
                   !(p.full_fallback_group_direct != 0 && group_direct_z2_supported(p)) &&
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
        // Never emit fewer chunks than the persistent grid can run at once:
        // the env-tuned CTA targets (416/512) predate the higher-occupancy
        // grid (570 at 5 blocks/SM), and a target below gridDim.x leaves CTAs
        // idle while each busy CTA serially walks a ~30k-token chunk. Two
        // chunks per CTA also shortens the straggler tail.
        int min_parallel_ctas = 2 * static_cast<int>(gridDim.x);
        if (target_ctas < min_parallel_ctas) target_ctas = min_parallel_ctas;
        long long raw = (weighted_tokens + static_cast<long long>(target_ctas) - 1LL) /
                        static_cast<long long>(target_ctas);
        if (mixed_groups == 0 && task_counts[6] > 0) {
          // Wave-quantized chunking for fallback-dominated schedules: a
          // token-count-derived chunk lands on arbitrary task counts (e.g.
          // 512 tasks on a 228-CTA grid = 2.25 waves, so the CTAs holding a
          // third tile finish 1.5x after the rest). The producer's finish
          // time is waves(k) * chunk(k) for k tiles per fallback group —
          // pick k by direct search (single thread, k <= 64: negligible).
          long long units = task_counts[6];
          if (units < 1) units = 1;
          long long grid_ctas = static_cast<long long>(gridDim.x);
          // Each producer task pays a fixed cost (pipeline fill, per-task
          // setup, partial write + later reduce/merge read) on top of its
          // streaming time; charge it as token-equivalents so the search
          // prefers few large chunks over many perfectly wave-aligned tiny
          // ones (k=57 "exact 16 waves" measured slower than k=7's 2 waves).
          const long long kTaskOverheadTokens = p.task_overhead_tokens;
          long long best_cost = 0;
          long long best_chunk = raw;
          for (int k = 1; k <= 64; ++k) {
            long long chunk = (static_cast<long long>(max_tokens) + k - 1) / k;
            if (chunk < min_chunk) break;  // larger k only shrinks the chunk
            long long waves = (units * k + grid_ctas - 1) / grid_ctas;
            long long cost = waves * (chunk + kTaskOverheadTokens);
            if (best_cost == 0 || cost < best_cost) {
              best_cost = cost;
              best_chunk = chunk;
            }
          }
          raw = best_chunk;
        }
        if (raw < min_chunk) raw = min_chunk;
        if (raw > max_tokens) raw = max_tokens;
        chosen = static_cast<int>(raw);
        if (chosen < p.tile_tokens) chosen = p.tile_tokens;
      }
    }
    task_counts[4] = chosen;
  }

  grid.sync();

  // Emit all z2 group tiles FIRST (contiguous low task indices) so the
  // round-robin task loop deals every CTA an equal share of the big chunks.
  // Serial per-group emission interleaves big fallback tiles with tiny hit
  // tiles in atomic order, and a CTA that draws two big chunks becomes the
  // straggler every other CTA waits on at the next grid sync (measured as a
  // spurious ~2x "tail phase" at hit=0.5). Covers mixed-only schedules
  // (original) and fallback+hit schedules with no mixed groups.
  if (p.parallel_z2_schedule != 0 && task_counts[5] > 0 &&
      (task_counts[6] == 0
           ? mixed_group_direct_z2_enabled(p)
           : (task_counts[5] == task_counts[6] &&
              p.full_fallback_group_direct != 0 && group_direct_z2_supported(p)))) {
    __shared__ int sh_z2_base;
    for (int go_i = blockIdx.x; go_i < group_total; go_i += gridDim.x) {
      int64_t go = static_cast<int64_t>(go_i);
      int mode = group_mode[go];
      int n = go_i / p.Hkv;
      int hkv = go_i - n * p.Hkv;
      int hq0 = hkv * p.group_size;
      int miss_heads = 0;
      if (mode == kGroupModeMixedFallback) {
        for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
          int hq = hq0 + lane_i;
          int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
          miss_heads += head_hit[ho] == 0 ? 1 : 0;
        }
      } else if (mode == kGroupModeFullFallback) {
        miss_heads = p.group_size;
      }

      int rect_tokens = group_rect_end[go] - group_rect_begin[go];
      int rect_tile_tokens = producer_rect_tile_tokens_for(p, task_counts, mode, rect_tokens);
      int rect_tiles =
          complete_tiles_for_group(p, task_counts, mode, group_rect_begin[go], group_rect_end[go]);
      bool z2_selected = rect_tiles > 0 && complete_group_direct_z2_selected(p, mode, miss_heads);
      if (z2_selected) {
        int misspack_tiles = 0;
        bool use_misspack_prefix = false;
        if (mode == kGroupModeMixedFallback && mixed_misspack_z2_enabled(p) && miss_heads > 0 &&
            miss_heads * 2 <= p.group_size) {
          int valid_begin = group_rect_end[go];
          for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
            int hq = hq0 + lane_i;
            int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
            if (head_hit[ho] != 0) {
              valid_begin = min(valid_begin, head_rect_start[ho]);
            }
          }
          int sink_tiles = front_sink_tiles_for_group(p, mode, group_rect_end[go]);
          int direct_until = max(group_rect_begin[go], min(valid_begin, group_rect_end[go]));
          misspack_tiles = sink_tiles + (direct_until - group_rect_begin[go]) / rect_tile_tokens;
          if (misspack_tiles < 0) misspack_tiles = 0;
          if (misspack_tiles > rect_tiles) misspack_tiles = rect_tiles;
          use_misspack_prefix = misspack_tiles > 0;
        }
        if (threadIdx.x == 0) {
          group_rect_tiles[go] = rect_tiles;
          sh_z2_base = atomicAdd(&task_counts[0], rect_tiles);
          // Reduction operates on band tiles only (sink tiles are merged raw
          // after the cache write), so threshold/chunk math counts band tiles.
          int z2_band_tiles =
              rect_tiles - front_sink_tiles_for_group(p, mode, group_rect_end[go]);
          if (z2_band_tiles > rect_reduce_threshold_for_counts(p, mode, task_counts)) {
            atomicAdd(&task_counts[3], 1);
            int reduce_chunk = clamp_reduce_chunk_to_workspace(
                balanced_rect_reduce_chunk_for_mode(p, mode, task_counts), z2_band_tiles,
                p.max_tiles_reduce);
            int reduced_tiles = ceil_div_i32(z2_band_tiles, reduce_chunk);
            if (mode == kGroupModeFullFallback) {
              atomicMax(&task_counts[7], reduced_tiles);
            } else {
              atomicMax(&task_counts[8], reduced_tiles);
            }
          }
        }
        __syncthreads();
        int sink_tiles_for_tag =
            front_sink_tiles_for_group(p, mode, group_rect_end[go]);
        for (int tile = threadIdx.x; tile < rect_tiles; tile += blockDim.x) {
          int32_t task_go_value = static_cast<int32_t>(go);
          if (mode == kGroupModeMixedFallback) {
            // Sink tiles contribute to every head (hit heads merge them
            // output-only), so they must never take the miss-heads-only
            // misspack tag.
            task_go_value =
                (use_misspack_prefix && tile >= sink_tiles_for_tag &&
                 tile < misspack_tiles)
                    ? static_cast<int32_t>(go + group_total * 2)
                    : static_cast<int32_t>(go + group_total);
          }
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
    int hit_miss_heads = 0;  // head_hit-based count, mirrors the parallel-z2 emitter
    bool valid_heads[kMaxGroupSize];
    for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
      int hq = hq0 + lane_i;
      int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
      int head_begin = head_rect_start[ho];
      int head_end = head_new_end[ho];
      int slot = head_match_slot[ho];
      hit_miss_heads += head_hit[ho] == 0 ? 1 : 0;
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
    int rect_tiles =
        complete_tiles_for_group(p, task_counts, mode, complete_begin, rect_end);
    int tail_begin = group_tail_begin[go];
    int tail_tiles = group_tail_tiles[go];
    group_rect_tiles[go] = rect_tiles;
    int miss_heads = p.group_size - valid_count;
    // Must reproduce the parallel-z2 emitter's decision EXACTLY (it counts
    // misses via head_hit, not via the stricter valid_heads criterion) —
    // disagreement here either double-emits or drops a group's tasks.
    bool z2_complete_emitted_parallel =
        p.parallel_z2_schedule != 0 &&
        (task_counts[6] == 0 || task_counts[5] == task_counts[6]) &&
        complete_group_direct_z2_selected(p, mode, hit_miss_heads);

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
        int sink_tiles = front_sink_tiles_for_group(p, mode, rect_end);
        int direct_tiles = sink_tiles +
                           (direct_until > complete_begin
                                ? (direct_until - complete_begin) / rect_tile_tokens
                                : 0);
        if (direct_tiles > rect_tiles) direct_tiles = rect_tiles;
        bool use_mixed_z2_task =
            complete_group_direct_z2_selected(p, mode, miss_heads);
        bool use_misspack_z2_task =
            use_mixed_z2_task && mixed_misspack_z2_enabled(p) && miss_heads > 0 &&
            miss_heads < p.group_size;
        if (direct_tiles > 0 && miss_heads > 0) {
          // Sink tiles contribute to every head (hit heads merge them
          // output-only), so they are always emitted as full-group tasks and
          // never per-miss-head or with the miss-heads-only misspack tag.
          int sink_emit = min(sink_tiles, direct_tiles);
          if (sink_emit > 0) {
            int base = atomicAdd(&task_counts[0], sink_emit);
            int32_t sink_go_value = use_mixed_z2_task
                                        ? static_cast<int32_t>(go + group_total)
                                        : static_cast<int32_t>(go);
            for (int tile = 0; tile < sink_emit; ++tile) {
              int idx = base + tile;
              complete_task_go[idx] = sink_go_value;
              complete_task_tile[idx] = tile;
            }
          }
          int band_direct = direct_tiles - sink_emit;
          if (band_direct > 0 && p.mixed_group_direct_min_miss_heads > 0 &&
              miss_heads >= p.mixed_group_direct_min_miss_heads) {
            int base = atomicAdd(&task_counts[0], band_direct);
            int32_t task_go_value = static_cast<int32_t>(go);
            if (use_misspack_z2_task) {
              task_go_value = static_cast<int32_t>(go + group_total * 2);
            } else if (use_mixed_z2_task) {
              task_go_value = static_cast<int32_t>(go + group_total);
            }
            for (int tile = sink_emit; tile < direct_tiles; ++tile) {
              int idx = base + (tile - sink_emit);
              complete_task_go[idx] = task_go_value;
              complete_task_tile[idx] = tile;
            }
          } else if (band_direct > 0) {
            int base = atomicAdd(&task_counts[0], miss_heads * band_direct);
            int out_lane = 0;
            for (int lane_i = 0; lane_i < p.group_size; ++lane_i) {
              if (valid_heads[lane_i]) continue;
              int hq = hq0 + lane_i;
              int32_t ho = static_cast<int32_t>(n * p.Hq + hq);
              for (int tile = sink_emit; tile < direct_tiles; ++tile) {
                int idx = base + out_lane * band_direct + (tile - sink_emit);
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
                 group_direct_z2_supported(p)) {
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
      // Reduction operates on band tiles only; the few sink tiles are merged
      // raw (after the cache write) so the stored entry can exclude them.
      int sched_band_tiles =
          rect_tiles - front_sink_tiles_for_group(p, mode, rect_end);
      if (mode != kGroupModeMacHit &&
          sched_band_tiles > rect_reduce_threshold_for_counts(p, mode, task_counts)) {
        atomicAdd(&task_counts[3], 1);
        if (mode == kGroupModeFullFallback) {
          int reduce_chunk = clamp_reduce_chunk_to_workspace(
              balanced_rect_reduce_chunk_for_mode(p, mode, task_counts), sched_band_tiles,
              p.max_tiles_reduce);
          int reduced_tiles = ceil_div_i32(sched_band_tiles, reduce_chunk);
          atomicMax(&task_counts[7], reduced_tiles);
        } else if (mode == kGroupModeMixedFallback) {
          int reduce_chunk = clamp_reduce_chunk_to_workspace(
              balanced_rect_reduce_chunk_for_mode(p, mode, task_counts), sched_band_tiles,
              p.max_tiles_reduce);
          int reduced_tiles = ceil_div_i32(sched_band_tiles, reduce_chunk);
          atomicMax(&task_counts[8], reduced_tiles);
        }
      }
    }

    if (tail_tiles > 0 && !tail_prepass_active(p)) {
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
    bool tail_group_direct_z2_on,
    uint8_t* __restrict__ dyn_smem) {
  const int total = task_counts[IsTail ? 1 : 0];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // Carved from the shared pool (double-buffered; the preceding z2 producer
  // used the same bytes, so re-converge the block before the first write).
  __nv_bfloat16* sh_k_stage = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
  __nv_bfloat16* sh_v_stage = sh_k_stage + 2 * kMaxStageTokens * kHeadDim;
  __syncthreads();

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
          group_direct_z2_supported(p)) {
        continue;
      }
    }
    int tiles = group_tiles[go];
    if (tile >= tiles) continue;

    int begin = 0;
    int end = 0;
    bool is_front_sink_tile = false;
    if constexpr (IsTail) {
      begin = group_begin[go] + tile * p.tail_tile_tokens;
      end = min(begin + p.tail_tile_tokens, group_end[go]);
    } else {
      if (!complete_tile_bounds(p, task_counts, group_mode[go], group_begin[go],
                                group_end[go], tile, begin, end,
                                is_front_sink_tile)) {
        continue;
      }
    }
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
        tile_overlaps_head =
            rect_tile_overlaps_head(p, begin, end, head_rect_start[ho], head_new_end[ho]);
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

    // Double-buffered cp_async staging: the copy of stage i+1 overlaps the
    // softmax of stage i, so the per-stage global-memory latency is hidden
    // instead of serialized behind two __syncthreads (this producer is the
    // generic path for every GQA size, and the only staged one for GQA-8).
    // Each thread copies whole 16B chunks: chunk c -> token s = c / 16,
    // dims [(c % 16) * 8, +8); the issuing thread reads req_to_token itself
    // (L1-resident), so no shared phys list or extra barrier is needed.
    const int chunks_per_token = p.D / 8;  // 16 for D=128
    auto issue_stage = [&](int stage_begin_i, int stage_count_i, int buf) {
      int total_chunks = stage_count_i * chunks_per_token;
      __nv_bfloat16* k_dst = sh_k_stage + buf * kMaxStageTokens * p.D;
      __nv_bfloat16* v_dst = sh_v_stage + buf * kMaxStageTokens * p.D;
      for (int c = threadIdx.x; c < total_chunks; c += blockDim.x) {
        int s = c / chunks_per_token;
        int d = (c - s * chunks_per_token) * 8;
        int pos = stage_begin_i + s;
        int phys = physical_token(
            n, req, pos, past_len, req_to_token, p.req_to_token_stride,
            out_cache_loc_i32, out_cache_loc_i64, p.out_cache_loc_is_i64);
        int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + d;
        cp_async_cg_16(&k_dst[s * p.D + d], &k_buffer[off]);
        cp_async_cg_16(&v_dst[s * p.D + d], &v_buffer[off]);
      }
      cp_async_commit();
    };

    int cur_buf = 0;
    issue_stage(begin, min(p.stage_tokens, end - begin), cur_buf);

    for (int stage_begin = begin; stage_begin < end; stage_begin += p.stage_tokens) {
      int stage_count = min(p.stage_tokens, end - stage_begin);
      int next_begin = stage_begin + p.stage_tokens;
      // Wait for the current buffer, and (via the barrier) for every warp to
      // be done with the buffer the next issue will overwrite.
      cp_async_wait_group<0>();
      __syncthreads();
      if (next_begin < end) {
        issue_stage(next_begin, min(p.stage_tokens, end - next_begin), cur_buf ^ 1);
      }
      const int stage_off_base = cur_buf * kMaxStageTokens * p.D;

      for (int s = 0; s < stage_count; ++s) {
        int pos = stage_begin + s;
        bool contributes = warp < p.group_size;
        if constexpr (!IsTail) {
          if (contributes) {
            int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
            contributes =
                rect_pos_contributes_to_head(p, pos, head_rect_start[ho], head_new_end[ho]);
          }
        }

        if (contributes) {
          const int stage_off = stage_off_base + s * p.D;
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
      cur_buf ^= 1;
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

template <int kGroup, int kZParts>
__device__ void phase_complete_attention_group_direct_z2_impl(
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
  constexpr int kStageTokens = kGroup * kZParts;
  constexpr int kBufs = z2_pipeline_bufs(kStageTokens);
  const int tx = threadIdx.x & 15;
  const int local = threadIdx.x >> 4;  // linearized (ty + kGroup * tz)
  const int ty = local % kGroup;
  const int tz = local / kGroup;
  constexpr bool active_z = true;
  const unsigned subwarp_mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
  const int dim_base = tx * 8;

  __nv_bfloat16* sh_k = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
  __nv_bfloat16* sh_v = sh_k + kBufs * kStageTokens * kHeadDim;
  float* sh_o = reinterpret_cast<float*>(sh_v + kBufs * kStageTokens * kHeadDim);
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
    // The reducer consumes band tiles only, so the threshold must count band
    // tiles (matching the scheduler's sched_band_tiles), not sink + band.
    bool keep_log2_for_fallback_reduce =
        mode == kGroupModeFullFallback && full_fallback_warp_reduce_enabled(p) &&
        (tiles - front_sink_tiles_for_group(p, mode, group_end[go])) >
            rect_reduce_threshold_for_counts(p, group_mode[go], task_counts);

    int begin = 0;
    int end = 0;
    bool is_front_sink_tile = false;
    if (!complete_tile_bounds(p, task_counts, group_mode[go], group_begin[go],
                              group_end[go], tile, begin, end, is_front_sink_tile)) {
      continue;
    }
    // Sink tiles bypass the reducer and are merged raw (natural-log domain) by
    // the phase4 output-only sink merge; never store them in log2 domain.
    keep_log2_for_fallback_reduce = keep_log2_for_fallback_reduce && !is_front_sink_tile;

    int req = req_ids[n];
    int past_len = past_lens[n];
    int64_t req_token_base = static_cast<int64_t>(req) * p.req_to_token_stride;
    int miss_lanes[kGroup];
    int miss_heads = 0;
    int lanes_per_miss_head = 1;
    int miss_slot = ty;
    bool misspack_lane_active = true;
    int out_lane = ty;
    int local_miss_part = 0;
    if (misspack_z2) {
#pragma unroll
      for (int lane_i = 0; lane_i < kGroup; ++lane_i) {
        int hq_i = hkv * p.group_size + lane_i;
        int64_t ho_i = static_cast<int64_t>(n) * p.Hq + hq_i;
        if (lane_i < p.group_size && head_hit[ho_i] == 0) {
          miss_lanes[miss_heads++] = lane_i;
        }
      }
      lanes_per_miss_head = miss_heads > 0 ? max(1, kGroup / miss_heads) : 1;
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
        (!mixed_z2 || rect_tile_overlaps_head(p, begin, end, head_begin, head_end)) &&
        misspack_lane_active;
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

    // cp_async ring: keep kBufs-1 stages in flight (one commit
    // group per stage; threads with nothing to load still commit an empty
    // group so per-thread group counts stay uniform for wait_group<N>).
    const int load_slot = tz * kGroup + ty;
    for (int i = 0; i < kBufs - 1; ++i) {
      int sb = begin + i * kStageTokens;
      int cnt = sb < end ? min(kStageTokens, end - sb) : 0;
      if (active_z && load_slot < cnt) {
        int load_pos = sb + load_slot;
        int phys = req_to_token[req_token_base + load_pos];
        int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + dim_base;
        int sh_off = i * kStageTokens * p.D + load_slot * p.D + dim_base;
        cp_async_cg_16(&sh_k[sh_off], &k_buffer[off]);
        cp_async_cg_16(&sh_v[sh_off], &v_buffer[off]);
      }
      cp_async_commit();
    }
    cp_async_wait_group<kBufs - 2>();
    z2_stage_part_sync<kGroup, kZParts>(tz);

    int cur_buf = 0;
    int stage_count = 0;
    for (int stage_begin = begin; stage_begin < end; stage_begin += kStageTokens) {
      stage_count = min(kStageTokens, end - stage_begin);
      int pre_begin = stage_begin + (kBufs - 1) * kStageTokens;
      int pre_count = pre_begin < end ? min(kStageTokens, end - pre_begin) : 0;
      int pre_buf = cur_buf + (kBufs - 1);
      if (pre_buf >= kBufs) pre_buf -= kBufs;
      // pre_buf was computed in the PREVIOUS iteration; the part sync at its
      // end orders that compute before this overwrite.
      if (active_z && load_slot < pre_count) {
        int load_pos = pre_begin + load_slot;
        int phys = req_to_token[req_token_base + load_pos];
        int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + dim_base;
        int sh_off = pre_buf * kStageTokens * p.D + load_slot * p.D + dim_base;
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
          bool contributes =
              tile_overlaps_head && rect_pos_contributes_to_head(p, pos, head_begin, head_end);
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
              tile_overlaps_head && slot < stage_count &&
              rect_pos_contributes_to_head(p, pos, head_begin, head_end);
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
              tile_overlaps_head &&
              (!mixed_z2 || rect_pos_contributes_to_head(p, pos, head_begin, head_end));
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
              (!mixed_z2 || rect_pos_contributes_to_head(p, pos, head_begin, head_end));
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
      cp_async_wait_group<kBufs - 2>();
      z2_stage_part_sync<kGroup, kZParts>(tz);
      cur_buf += 1;
      if (cur_buf >= kBufs) cur_buf = 0;
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

template <int BlockThreads>
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
  if (p.full_fallback_group_direct == 0 || dyn_smem == nullptr) return;
  if (z2_parts_for_config(BlockThreads, p.group_size, p.D) == 0) return;
#define MAC_Z2_COMPLETE_ARGS                                                      \
  p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens,                \
      out_cache_loc_i32, out_cache_loc_i64, group_begin, group_end, group_tiles,  \
      group_mode, head_hit, head_rect_start, head_new_end, task_go, task_tile,    \
      task_counts, partial_o, partial_lse, max_tiles, dyn_smem
  if constexpr (BlockThreads == 128) {
    if (p.group_size == 4) {
      phase_complete_attention_group_direct_z2_impl<4, 2>(MAC_Z2_COMPLETE_ARGS);
    }
  } else {
    if (p.group_size == 8) {
      phase_complete_attention_group_direct_z2_impl<8, 2>(MAC_Z2_COMPLETE_ARGS);
    }
  }
#undef MAC_Z2_COMPLETE_ARGS
}

template <int kGroup, int kZParts>
__device__ void phase_tail_attention_group_direct_z2_impl(
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
    bool prepass) {
  constexpr int kStageTokens = kGroup * kZParts;
  constexpr int kBufs = z2_pipeline_bufs(kStageTokens);
  const int tx = threadIdx.x & 15;
  const int local = threadIdx.x >> 4;
  const int ty = local % kGroup;
  const int tz = local / kGroup;
  const unsigned subwarp_mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
  const int dim_base = tx * 8;

  __nv_bfloat16* sh_k = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
  __nv_bfloat16* sh_v = sh_k + kBufs * kStageTokens * kHeadDim;
  float* sh_o = reinterpret_cast<float*>(sh_v + kBufs * kStageTokens * kHeadDim);
  float* sh_lse = sh_o + kGroup * kZParts * kHeadDim;

  const int group_total = p.N * p.Hkv;
  // Prepass mode: schedule-free flat (group, tile) index space; the bounds
  // math matches group_tail_begin/end exactly (both derive from past_len).
  int tiles_pg = 1;
  if (prepass) {
    tiles_pg = p.semantic_pos_ahead > 0
                   ? ceil_div_i32(p.semantic_pos_ahead, p.tail_tile_tokens)
                   : 1;
    if (tiles_pg < 1) tiles_pg = 1;
  }
  const int total = prepass ? group_total * tiles_pg : task_counts[1];
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int64_t go;
    int tile;
    int begin;
    int end;
    int hkv;
    int n;
    if (prepass) {
      go = task / tiles_pg;
      tile = task % tiles_pg;
      hkv = static_cast<int>(go % p.Hkv);
      n = static_cast<int>(go / p.Hkv);
      int past_len_i = past_lens[n];
      int kv_len = past_len_i + 1;
      int tail_begin = cache_end_for_pos(past_len_i, p.semantic_pos_ahead);
      begin = tail_begin + tile * p.tail_tile_tokens;
      end = min(begin + p.tail_tile_tokens, kv_len);
    } else {
      int encoded_task = task_go[task];
      if (encoded_task < 0 || encoded_task >= group_total) continue;
      go = static_cast<int64_t>(encoded_task);
      tile = task_tile[task];
      int tiles = group_tiles[go];
      if (tile >= tiles) continue;
      hkv = static_cast<int>(go % p.Hkv);
      n = static_cast<int>(go / p.Hkv);
      begin = group_begin[go] + tile * p.tail_tile_tokens;
      end = min(begin + p.tail_tile_tokens, group_end[go]);
    }
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

    // cp_async ring, same structure as the complete producer (see there).
    const int load_slot = tz * kGroup + ty;
    for (int i = 0; i < kBufs - 1; ++i) {
      int sb = begin + i * kStageTokens;
      int cnt = sb < end ? min(kStageTokens, end - sb) : 0;
      if (load_slot < cnt) {
        int load_pos = sb + load_slot;
        int phys = physical_token(n, req, load_pos, past_len, req_to_token,
                                  p.req_to_token_stride, out_cache_loc_i32,
                                  out_cache_loc_i64, p.out_cache_loc_is_i64);
        int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + dim_base;
        int sh_off = i * kStageTokens * p.D + load_slot * p.D + dim_base;
        cp_async_cg_16(&sh_k[sh_off], &k_buffer[off]);
        cp_async_cg_16(&sh_v[sh_off], &v_buffer[off]);
      }
      cp_async_commit();
    }
    cp_async_wait_group<kBufs - 2>();
    z2_stage_part_sync<kGroup, kZParts>(tz);

    int cur_buf = 0;
    int stage_count = 0;
    for (int stage_begin = begin; stage_begin < end; stage_begin += kStageTokens) {
      stage_count = min(kStageTokens, end - stage_begin);
      int pre_begin = stage_begin + (kBufs - 1) * kStageTokens;
      int pre_count = pre_begin < end ? min(kStageTokens, end - pre_begin) : 0;
      int pre_buf = cur_buf + (kBufs - 1);
      if (pre_buf >= kBufs) pre_buf -= kBufs;
      if (load_slot < pre_count) {
        int load_pos = pre_begin + load_slot;
        int phys = physical_token(n, req, load_pos, past_len, req_to_token,
                                  p.req_to_token_stride, out_cache_loc_i32,
                                  out_cache_loc_i64, p.out_cache_loc_is_i64);
        int64_t off = (static_cast<int64_t>(phys) * p.Hkv + hkv) * p.D + dim_base;
        int sh_off = pre_buf * kStageTokens * p.D + load_slot * p.D + dim_base;
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
      cp_async_wait_group<kBufs - 2>();
      z2_stage_part_sync<kGroup, kZParts>(tz);
      cur_buf += 1;
      if (cur_buf >= kBufs) cur_buf = 0;
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

template <int BlockThreads>
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
    bool tail_group_direct_z2_on,
    bool prepass) {
  if (dyn_smem == nullptr) return;
  if (prepass) {
    if (!tail_prepass_active(p)) return;
  } else if (!tail_group_direct_z2_on) {
    return;
  }
  if (z2_parts_for_config(BlockThreads, p.group_size, p.D) == 0) return;
#define MAC_Z2_TAIL_ARGS                                                          \
  p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens,                \
      out_cache_loc_i32, out_cache_loc_i64, group_begin, group_end, group_tiles,  \
      group_mode, task_go, task_tile, task_counts, partial_o, partial_lse,        \
      max_tiles, dyn_smem, prepass
  if constexpr (BlockThreads == 128) {
    if (p.group_size == 4) {
      phase_tail_attention_group_direct_z2_impl<4, 2>(MAC_Z2_TAIL_ARGS);
    }
  } else {
    if (p.group_size == 8) {
      phase_tail_attention_group_direct_z2_impl<8, 2>(MAC_Z2_TAIL_ARGS);
    }
  }
#undef MAC_Z2_TAIL_ARGS
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

    int begin = 0;
    int end = 0;
    bool is_front_sink_tile = false;
    if (!complete_tile_bounds(p, task_counts, group_mode[go], group_begin[go],
                              group_end[go], tile, begin, end, is_front_sink_tile)) {
      continue;
    }
    int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
    if (is_front_sink_tile) {
      end = min(end, head_front_sink_end(p, head_rect_start[ho], head_new_end[ho]));
    } else {
      begin = max(begin, head_rect_start[ho]);
      end = min(end, head_new_end[ho]);
    }
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
  const int max_hit_tiles =
      min(max_tiles, ceil_div_i32(max(p.M, 1), p.tile_tokens) +
                         ceil_div_i32(max(p.front_sink_tokens, 0), p.tile_tokens));
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

    int begin = 0;
    int end = 0;
    bool is_front_sink_tile = false;
    if (!complete_tile_bounds(p, nullptr, group_mode[go], group_begin[go],
                              group_end[go], tile, begin, end, is_front_sink_tile)) {
      continue;
    }
    int req = req_ids[n];
    int past_len = past_lens[n];
    int hq = hkv * p.group_size + warp;
    bool tile_overlaps_head = false;
    if (warp < p.group_size) {
      int64_t ho = static_cast<int64_t>(n) * p.Hq + hq;
      tile_overlaps_head =
          rect_tile_overlaps_head(p, begin, end, head_rect_start[ho], head_new_end[ho]);
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
        contributes =
            rect_pos_contributes_to_head(p, pos, head_rect_start[ho], head_new_end[ho]);
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
    int begin = group_tail_begin[go] + tile * p.tail_tile_tokens;
    int end = min(begin + p.tail_tile_tokens, group_tail_end[go]);
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

    int begin = group_tail_begin[go] + tile * p.tail_tile_tokens;
    int end = min(begin + p.tail_tile_tokens, group_tail_end[go]);
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
    // Band tiles sit physically after the group's front-sink tiles; the sink
    // tiles are excluded from reduction (merged raw after the cache write).
    int sink_tiles_total =
        per_head_rect ? 0
                      : front_sink_tiles_for_group(p, group_mode[go], group_rect_end[go]);
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
      if (!rect_tile_overlaps_head(p, begin, end, head_begin, head_end)) {
        continue;
      }
      int64_t lse_off = (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context +
                          (sink_tiles_total + tile)) *
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
  const int warps = static_cast<int>(blockDim.x) >> 5;
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
    // Band tiles sit physically after the group's front-sink tiles; sink tiles
    // are excluded from reduction (merged raw after the cache write).
    int sink_tiles_total =
        front_sink_tiles_for_group(p, group_mode[go], group_rect_end[go]);
    // One warp per head lane; group sizes above the warp count wrap around
    // (e.g. group 16 on a 256-thread block reduces two heads per warp).
    for (int lane_hd = warp; lane_hd < p.group_size; lane_hd += warps) {
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
          (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context +
            (sink_tiles_total + tile)) *
           p.group_size + lane_hd);
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
         p.group_size + lane_hd);
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
}

template <int BlockThreads>
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
    float* __restrict__ partial_rect_reduced_lse,
    uint8_t* __restrict__ dyn_smem) {
  if (!full_fallback_head_reduce_enabled(p)) return;

  constexpr int kBdx = 16;
  // Row count covers the whole block so every thread reaches the per-task
  // __syncthreads() below (an early return here would desync the block from
  // the next pool-using phase).
  constexpr int kBdy = BlockThreads / 16;
  constexpr int kVec = 8;
  const int tx = threadIdx.x & (kBdx - 1);
  const int ty = threadIdx.x >> 4;

  float* sh_o = reinterpret_cast<float*>(dyn_smem);
  float* sh_m = sh_o + kBdy * kHeadDim;
  float* sh_d = sh_m + kBdy;

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
    // Band tiles sit physically after the group's front-sink tiles; sink tiles
    // are excluded from reduction (merged raw after the cache write).
    int sink_tiles_total =
        front_sink_tiles_for_group(p, group_mode[go], group_rect_end[go]);
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
          (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context +
            (sink_tiles_total + tile)) *
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

template <int BlockThreads>
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
    float* __restrict__ partial_rect_reduced_lse,
    uint8_t* __restrict__ dyn_smem) {
  if (!mixed_head_reduce_enabled(p)) return;

  constexpr int kBdx = 16;
  // Whole-block row coverage; see phase_reduce_full_fallback_head_vec.
  constexpr int kBdy = BlockThreads / 16;
  constexpr int kVec = 8;
  const int tx = threadIdx.x & (kBdx - 1);
  const int ty = threadIdx.x >> 4;

  float* sh_o = reinterpret_cast<float*>(dyn_smem);
  float* sh_m = sh_o + kBdy * kHeadDim;
  float* sh_d = sh_m + kBdy;

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
    if (!rect_tile_overlaps_head(p, chunk_begin_pos, chunk_end_pos, head_begin, head_end)) {
      continue;
    }

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
      if (!rect_tile_overlaps_head(p, begin, end, head_begin, head_end)) continue;
      // Physical band tiles are offset by the group's front-sink tiles.
      int64_t lse_off =
          (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context +
            (front_sink_tiles_for_group(p, group_mode[go], group_end) + tile)) *
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
    // Not-ready requests (graph replay while the prefill offload is still
    // populating this request's rows) must not touch any cache row.
    const bool cache_row_writable = request_persistent_ready(p, req);
    int64_t q_cache_base = (((static_cast<int64_t>(req) * p.M + dest_slot) * p.Hq + hq) * p.D);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = dims[i];
      if (d < p.D && cache_row_writable) {
        query_cache[q_cache_base + d] =
            q_pre[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + d];
        attn_cache[q_cache_base + d] = cache_valid ? float_to_bf16(acc[i]) : float_to_bf16(0.0f);
      }
    }
    if (lane == 0 && cache_row_writable) {
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
        int begin = tail_begin_base + tile * p.tail_tile_tokens;
        int end = min(begin + p.tail_tile_tokens, tail_end_all);
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
    const float* __restrict__ partial_tail_lse,
    uint8_t* __restrict__ dyn_smem) {
  const int total = p.N * p.Hq;
  __shared__ float sh_lse;
  __shared__ bool sh_valid;
  __shared__ float sh_w_state;
  __shared__ float sh_w_other;
  __shared__ bool sh_take_other;
  __shared__ bool sh_other_valid;
  float* sh_fused_tail_o = reinterpret_cast<float*>(dyn_smem);
  float* sh_fused_tail_lse = sh_fused_tail_o + kMaxFuseTailTilesInMerge * kHeadDim;

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
    int head_begin = head_rect_start[ho];
    int head_end = head_new_end[ho];
    int rect_tile_begin = 0;
    int rect_tile_end = rect_tiles;
    if (per_head_rect) {
      int head_tiles = (head_end > head_begin) ? ceil_div_i32(head_end - head_begin, p.tile_tokens) : 0;
      use_reduced_rect = head_tiles > kRectReduceChunk;
      rect_tile_begin = 0;
      rect_tile_end = use_reduced_rect ? ceil_div_i32(head_tiles, reduce_chunk) : head_tiles;
      for (int tile = rect_tile_begin; tile < rect_tile_end; ++tile) {
        int64_t lse_off =
            use_reduced_rect
                ? (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_reduce + tile) *
                   p.group_size + lane_id)
                : (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context + tile) *
                   p.group_size + lane_id);
        merge_partial_rect_tile_into_state(
            p, lse_off, use_reduced_rect, partial_rect_o, partial_rect_lse,
            partial_rect_reduced_o, partial_rect_reduced_lse, dim, owns_dim, acc,
            state_lse, state_valid, &sh_lse, &sh_valid, &sh_w_state, &sh_w_other,
            &sh_take_other, &sh_other_valid);
      }
    } else if (rect_tiles > 0) {
      int group_begin = group_rect_begin[go];
      int group_end = group_rect_end[go];
      int mode = group_mode[go];
      int sink_tiles_total = front_sink_tiles_for_group(p, mode, group_end);

      // Only band tiles are merged into the state here; the front-sink tiles
      // are merged into the OUTPUT after the cache write below, so stored
      // entries exclude [0, min(F, prefix_end)) by construction.
      if (mode == kGroupModeMacHit) {
        if (head_end > head_begin) {
          int rel_begin = head_begin - group_begin;
          int rel_end = head_end - group_begin;
          if (rel_begin < 0) rel_begin = 0;
          if (rel_end < 0) rel_end = 0;
          int orig_tile_begin = rel_begin / rect_tile_tokens;
          int orig_tile_end = ceil_div_i32(rel_end, rect_tile_tokens);
          for (int tile = orig_tile_begin; tile < orig_tile_end; ++tile) {
            int physical_tile = sink_tiles_total + tile;
            if (physical_tile < 0 || physical_tile >= rect_tiles) continue;
            int64_t lse_off =
                (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context +
                  physical_tile) *
                 p.group_size + lane_id);
            merge_partial_rect_tile_into_state(
                p, lse_off, false, partial_rect_o, partial_rect_lse,
                partial_rect_reduced_o, partial_rect_reduced_lse, dim, owns_dim, acc,
                state_lse, state_valid, &sh_lse, &sh_valid, &sh_w_state, &sh_w_other,
                &sh_take_other, &sh_other_valid);
          }
        }
      } else {
        // Fallback/mixed band region already starts after the group's sink
        // (group_begin == min(F, new_end)); merge the band tiles overlapping
        // this head. Raw band tiles sit physically after the sink tiles;
        // reduced band tiles are indexed from 0 (the reducer skips the sink).
        int band_orig_tile_begin = 0;
        int band_orig_tile_end = 0;
        if (head_end > head_begin) {
          int rel_begin = head_begin - group_begin;
          int rel_end = head_end - group_begin;
          if (rel_begin < 0) rel_begin = 0;
          if (rel_end < 0) rel_end = 0;
          band_orig_tile_begin = rel_begin / rect_tile_tokens;
          band_orig_tile_end = ceil_div_i32(rel_end, rect_tile_tokens);
        }

        int band_tile_begin = band_orig_tile_begin;
        int band_tile_end = band_orig_tile_end;
        if (use_reduced_rect) {
          band_tile_begin = band_orig_tile_begin / reduce_chunk;
          band_tile_end = ceil_div_i32(band_orig_tile_end, reduce_chunk);
        }
        if (band_tile_end > rect_tiles) band_tile_end = rect_tiles;

        for (int tile = band_tile_begin; tile < band_tile_end; ++tile) {
          int64_t physical_tile =
              use_reduced_rect ? static_cast<int64_t>(tile)
                               : static_cast<int64_t>(sink_tiles_total + tile);
          int64_t lse_off =
              use_reduced_rect
                  ? (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_reduce +
                      physical_tile) *
                     p.group_size + lane_id)
                  : (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context +
                      physical_tile) *
                     p.group_size + lane_id);
          merge_partial_rect_tile_into_state(
              p, lse_off, use_reduced_rect, partial_rect_o, partial_rect_lse,
              partial_rect_reduced_o, partial_rect_reduced_lse, dim, owns_dim, acc,
              state_lse, state_valid, &sh_lse, &sh_valid, &sh_w_state, &sh_w_other,
              &sh_take_other, &sh_other_valid);
        }
      }
    }

    // Cache write BEFORE the sink merge: the stored entry covers exactly
    // [min(F, new_end), new_end) — hit heads: reused [F, rect_start) plus band;
    // miss heads: band from min(F, new_end). Exclusion by construction; no
    // sink metadata is needed and no subtraction ever happens.
    float cache_lse = state_lse;
    bool cache_valid = state_valid;
    // Not-ready requests (graph replay while the prefill offload is still
    // populating this request's rows) must not touch any cache row.
    const bool cache_row_writable = request_persistent_ready(p, req);
    if (owns_dim && cache_row_writable) {
      query_cache[q_cache_base + dim] = q_pre[(static_cast<int64_t>(n) * p.Hq + hq) * p.D + dim];
      attn_cache[q_cache_base + dim] = cache_valid ? float_to_bf16(acc) : float_to_bf16(0.0f);
    }
    if (threadIdx.x == 0 && cache_row_writable) {
      lse_cache[(static_cast<int64_t>(req) * p.M + dest_slot) * p.Hq + hq] =
          cache_valid ? cache_lse : -kInf;
    }
    __syncthreads();

    // Front-sink merge (output-only), for every head in every group mode: the
    // sink region [0, min(F, head_rect_start)) was recomputed under the CURRENT
    // post-RoPE query by the complete producers into the leading physical tiles.
    if (!per_head_rect && p.front_sink_tokens > 0) {
      int fs_mode = group_mode[go];
      int fs_group_end = group_rect_end[go];
      int fs_sink_tiles_total = front_sink_tiles_for_group(p, fs_mode, fs_group_end);
      int fs_sink_end = head_front_sink_end(p, head_begin, head_end);
      int fs_tile_end = fs_sink_end > 0 ? ceil_div_i32(fs_sink_end, p.tile_tokens) : 0;
      if (fs_tile_end > fs_sink_tiles_total) fs_tile_end = fs_sink_tiles_total;
      for (int tile = 0; tile < fs_tile_end; ++tile) {
        int64_t lse_off =
            (((static_cast<int64_t>(n) * p.Hkv + hkv) * p.max_tiles_context + tile) *
             p.group_size + lane_id);
        merge_partial_rect_tile_into_state(
            p, lse_off, false, partial_rect_o, partial_rect_lse,
            partial_rect_reduced_o, partial_rect_reduced_lse, dim, owns_dim, acc,
            state_lse, state_valid, &sh_lse, &sh_valid, &sh_w_state, &sh_w_other,
            &sh_take_other, &sh_other_valid);
      }
    }

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
        int begin = tail_begin_base + tile * p.tail_tile_tokens;
        int end = min(begin + p.tail_tile_tokens, tail_end_all);
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

// Two launch-bounds instantiations: block 128 (GQA<=4) and block 256 (GQA>4,
// e.g. Qwen3-30B). Without an explicit bound ptxas used 104 regs/thread,
// which rounds occupancy down to 4 blocks/SM; capping to 102 (block 128)
// lifts the persistent grid to 5 blocks/SM with negligible spill.
#ifndef MAC_PERSISTENT_LB_MIN_BLOCKS_128
#define MAC_PERSISTENT_LB_MIN_BLOCKS_128 5
#endif
#ifndef MAC_PERSISTENT_LB_MIN_BLOCKS_256
#define MAC_PERSISTENT_LB_MIN_BLOCKS_256 2
#endif
template <int BlockThreads>
__global__ void __launch_bounds__(
    BlockThreads,
    BlockThreads == 128 ? MAC_PERSISTENT_LB_MIN_BLOCKS_128
                        : MAC_PERSISTENT_LB_MIN_BLOCKS_256)
mac_persistent_decode_kernel(
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
  // Tail prepass: tail spans depend only on past_len, so they can stream
  // (bandwidth-bound) while the match scan stalls on probe latency. Writes
  // are idempotent stores; the scheduler emits no tail tasks when active.
  if (tail_prepass_active(p)) {
    phase_tail_attention_group_direct_z2<BlockThreads>(
        p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens,
        out_cache_loc_i32, out_cache_loc_i64, group_tail_begin, group_tail_end,
        group_tail_tiles, group_mode, tail_task_go, tail_task_tile, task_counts,
        partial_tail_o, partial_tail_lse, p.max_tiles_tail, dyn_smem,
        /*tail_group_direct_z2_on=*/false, /*prepass=*/true);
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
    phase_complete_attention_group_direct_z2<BlockThreads>(
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
                                       false, dyn_smem);
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
    phase_tail_attention_group_direct_z2<BlockThreads>(p, q_post, k_buffer, v_buffer, req_to_token,
                                         req_ids, past_lens, out_cache_loc_i32,
                                         out_cache_loc_i64, group_tail_begin,
                                         group_tail_end, group_tail_tiles, group_mode,
                                         tail_task_go, tail_task_tile, task_counts,
                                         partial_tail_o, partial_tail_lse,
                                         p.max_tiles_tail, dyn_smem,
                                         tail_group_direct_z2_on,
                                         /*prepass=*/false);
  }
  phase_attention_tiles_compact<true>(p, q_post, k_buffer, v_buffer, req_to_token, req_ids, past_lens,
                                      out_cache_loc_i32, out_cache_loc_i64,
                                      head_rect_start, head_new_end,
                                      group_tail_begin, group_tail_end, group_tail_tiles,
                                      group_mode,
                                      tail_task_go, tail_task_tile, task_counts,
                                      partial_tail_o, partial_tail_lse, p.max_tiles_tail,
                                      tail_group_direct_z2_on, dyn_smem);
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
      phase_reduce_full_fallback_head_vec<BlockThreads>(
          p, group_mode, group_rect_begin, group_rect_end, group_rect_tiles, task_counts,
          partial_rect_o, partial_rect_lse, partial_rect_reduced_o, partial_rect_reduced_lse,
          dyn_smem);
    } else if (task_counts[6] > 0) {
      phase_reduce_full_fallback_group_warp(
          p, group_mode, group_rect_begin, group_rect_end, group_rect_tiles, task_counts,
          partial_rect_o, partial_rect_lse, partial_rect_reduced_o, partial_rect_reduced_lse);
    }
    if (use_mixed_head_vec_reduce) {
      phase_reduce_mixed_rect_head_vec<BlockThreads>(
          p, group_mode, group_rect_begin, group_rect_end, head_rect_start, head_new_end,
          group_rect_tiles, task_counts, partial_rect_o, partial_rect_lse,
          partial_rect_reduced_o, partial_rect_reduced_lse, dyn_smem);
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
                     partial_tail_lse, dyn_smem);
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
    int64_t front_sink_tokens,
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
    int64_t bench_layer_id,
    Tensor query_post_cache,
    Tensor mac_cache_ready,
    Tensor mac_cache_epoch,
    Tensor mac_request_epoch) {
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
  // query_post_cache is a legacy argument from the (removed) front-sink
  // downdate design; it is accepted for signature compatibility and ignored.
  if (query_post_cache.defined() && query_post_cache.numel() > 0) {
    check_tensor(query_post_cache, "query_post_cache", at::kBFloat16, 4);
  }
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

  // Optional device-side readiness gate (empty tensors = every request
  // ready). Required for CUDA-graph replay, where the python-side
  // `_persistent_ready` gate cannot run per step.
  const uint8_t* cache_ready_ptr = nullptr;
  const int64_t* cache_epoch_ptr = nullptr;
  const int64_t* request_epoch_ptr = nullptr;
  if (mac_cache_ready.defined() && mac_cache_ready.numel() > 0) {
    CHECK_CUDA(mac_cache_ready);
    CHECK_CONTIG(mac_cache_ready);
    TORCH_CHECK(mac_cache_ready.scalar_type() == at::kBool ||
                    mac_cache_ready.scalar_type() == at::kByte,
                "mac_cache_ready must be bool or uint8");
    cache_ready_ptr = reinterpret_cast<const uint8_t*>(mac_cache_ready.data_ptr());
    if (mac_cache_epoch.defined() && mac_cache_epoch.numel() > 0 &&
        mac_request_epoch.defined() && mac_request_epoch.numel() > 0) {
      check_tensor(mac_cache_epoch, "mac_cache_epoch", at::kLong, 1);
      check_tensor(mac_request_epoch, "mac_request_epoch", at::kLong, 1);
      cache_epoch_ptr = mac_cache_epoch.data_ptr<int64_t>();
      request_epoch_ptr = mac_request_epoch.data_ptr<int64_t>();
    }
  }

  c10::cuda::CUDAGuard guard(q_post.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q_post.get_device()).stream();

  Params p{};
  p.cache_ready_ptr = cache_ready_ptr;
  p.cache_epoch_ptr = cache_epoch_ptr;
  p.request_epoch_ptr = request_epoch_ptr;
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
  // Tail granularity: the whole tail span is ~semantic_pos_ahead tokens per
  // group; two tiles keep some parallelism while cutting the task count 4x
  // (the per-task fixed costs dominated with tile_tokens-sized slices).
  p.tail_tile_tokens = env_int("MAC_PERSISTENT_TAIL_TILE_TOKENS", 0);
  if (p.tail_tile_tokens <= 0) {
    int half_span = p.semantic_pos_ahead > 0 ? (p.semantic_pos_ahead + 1) / 2 : 0;
    p.tail_tile_tokens = half_span > p.tile_tokens ? half_span : p.tile_tokens;
  }
  // The tail partial workspace has ceil(sem/tile_tokens) slots; a smaller
  // tail tile would overflow it.
  if (p.tail_tile_tokens < p.tile_tokens) p.tail_tile_tokens = p.tile_tokens;
  p.tail_prepass = env_flag("MAC_PERSISTENT_TAIL_PREPASS", 0);
  p.front_sink_tokens = (int)front_sink_tokens;
  if (p.front_sink_tokens < 0) p.front_sink_tokens = 0;
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
  // For GQA > 4 there is no z2 group producer (yet); the per-head direct
  // producer walks tokens serially with no staging and collapses on long
  // full-fallback bands (group-8 @128k: 100ms vs 26ms staged), so route full
  // fallback through the staged compact producer by default. Short mixed-band
  // per-head work (mixed_early_miss_direct) stays on the direct path.
  p.full_fallback_head_direct = env_flag(
      "MAC_PERSISTENT_FULL_FALLBACK_HEAD_DIRECT", group_size > 4 ? 0 : 1);
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
  if (p.front_sink_tokens > 0) {
    // Front-sink exclusion (v2): the producers and schedulers understand the
    // [sink | band] physical tile layout (complete_tile_bounds emits sink
    // tiles first; sink tiles are full-group work, never misspack or
    // per-miss-head), the fast reducers offset band tiles by the group's sink
    // tiles and reduce band tiles only, and the z2 producer keeps sink tiles
    // in natural-log domain for phase4's raw output-only sink merge. The one
    // remaining unaudited path under this layout is the standalone
    // full-fallback group merge, so keep it on the generic phase4 route.
    p.full_fallback_group_merge = 0;
    p.fuse_fallback_tail_in_merge = 0;
  }
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
  // Needs the block size to know whether a z2 producer instantiation exists.
  int block_threads = env_int("MAC_PERSISTENT_BLOCK_THREADS", group_size > 4 ? 256 : 128);
  if (block_threads < group_size * 32) block_threads = group_size * 32;
  if (block_threads != 128 && block_threads != 256) block_threads = group_size > 4 ? 256 : 128;
  int mixed_group_direct_default_min =
      (p.mixed_group_direct_z2 != 0 &&
       z2_parts_for_config(block_threads, p.group_size, p.D) > 0)
          ? 1
          : p.group_size;
  p.mixed_group_direct_min_miss_heads =
      env_int("MAC_PERSISTENT_MIXED_GROUP_DIRECT_MIN_MISS_HEADS", mixed_group_direct_default_min);
  p.task_overhead_tokens = env_int("MAC_PERSISTENT_TASK_OVERHEAD_TOKENS", 128);
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

  TORCH_CHECK(block_threads >= D, "persistent decode v1 merge requires at least one thread per head dim");
  // Every phase carves its bulk shared memory from one dynamic pool (see
  // smem_pool_bytes); the pool is required regardless of which fast paths are
  // active and scales with the block variant (z2 stage = blockDim/16 tokens).
  int coop_dynamic_smem = block_threads == 128 ? smem_pool_bytes(128) : smem_pool_bytes(256);
  void* kernel_fn = block_threads == 128
                        ? reinterpret_cast<void*>(mac_persistent_decode_kernel<128>)
                        : reinterpret_cast<void*>(mac_persistent_decode_kernel<256>);
  int device = q_post.get_device();
  // The attribute/occupancy queries are stream-capture legal (verified on
  // CUDA 13) but cost several driver calls per launch; cache them per
  // (device, block variant). dyn smem is the compile-time pool size, so the
  // occupancy result never changes after the first call.
  struct LaunchInfo {
    int sm_count;
    int active_per_sm;
  };
  static std::mutex launch_info_mu;
  static std::map<std::pair<int, int>, LaunchInfo> launch_info_cache;
  LaunchInfo info;
  {
    std::lock_guard<std::mutex> lock(launch_info_mu);
    auto key = std::make_pair(device, block_threads);
    auto it = launch_info_cache.find(key);
    if (it == launch_info_cache.end()) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
          coop_dynamic_smem));
      // The default shared-memory carveout (~100KB/SM) caps the 128-thread
      // variant at 4 blocks/SM once the z2 ring pool is 20.5KB; ask for the
      // full carveout so occupancy stays register-limited (5 at block 128,
      // 2 at block 256).
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      int coop = 0;
      C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, device));
      TORCH_CHECK(coop != 0, "device does not support cooperative launch");
      C10_CUDA_CHECK(cudaDeviceGetAttribute(&info.sm_count,
                                            cudaDevAttrMultiProcessorCount, device));
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &info.active_per_sm, kernel_fn, block_threads, coop_dynamic_smem));
      TORCH_CHECK(info.active_per_sm > 0, "persistent kernel has zero occupancy");
      launch_info_cache.emplace(key, info);
    } else {
      info = it->second;
    }
  }
  int sm_count = info.sm_count;
  int active_per_sm = info.active_per_sm;
  int max_grid = active_per_sm * sm_count;
  int default_cap = 1024 * (p.N > 0 ? p.N : 1);
  if (default_cap > max_grid) default_cap = max_grid;
  int cap = env_int("MAC_PERSISTENT_COOP_CTAS", default_cap);
  int grid = cap < max_grid ? cap : max_grid;
  if (grid < 1) grid = 1;
  if (env_flag("MAC_PERSISTENT_DEBUG_LAUNCH", 0)) {
    // ncu cannot profile cooperative launches; this is the occupancy story.
    static bool printed = false;
    if (!printed) {
      printed = true;
      cudaFuncAttributes attr{};
      C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
      printf(
          "[mac_persistent] grid=%d block=%d active_per_sm=%d sm_count=%d "
          "regs/thread=%d static_smem=%zuB dyn_smem=%dB local=%zuB\n",
          grid, block_threads, active_per_sm, sm_count, attr.numRegs,
          attr.sharedSizeBytes, coop_dynamic_smem, attr.localSizeBytes);
    }
  }

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
      kernel_fn,
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
