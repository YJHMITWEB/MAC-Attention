// mac_merge_downdate_cache.cu
// In-place fused update-then-downdate kernel
// Changes vs original:
//   * Exposes ONLY the 'mac_update_then_downdate_bf16_inplace' API
//   * Expects out1/out2: [N,H,D] (BF16), lse1/lse2: [N,H] (FP16/BF16/FP32)
//   * Writes results IN-PLACE into out2 (BF16) and lse2 (same dtype as lse2)
//   * No new output tensors are created
//
// One CTA per (N,H) group; double-buffered smem; cp.async on sm80+
// Tuned for H100-class GPUs; still requires sm_80+.
//
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>      // CUDAGuard + c10 CUDA types
#include <ATen/cuda/CUDAContext.h>    // at::cuda::getCurrentCUDAStream
#include <stdint.h>                   // uintptr_t
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

// ====================== Tunables (H100-friendly defaults) ======================
#ifndef TILE_D
#define TILE_D 1024   // BF16 elements per stage/phase over head_dim
#endif
#ifndef THREADS
#define THREADS 256   // 8 warps: warp 0 = loader, others compute
#endif
#ifndef SMEM_PAD
#define SMEM_PAD 8    // small pad to reduce potential bank conflicts
#endif
#ifndef LSE_RTOL
#define LSE_RTOL 1e-2f
#endif
#ifndef LSE_ATOL
#define LSE_ATOL 1e-3f
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#error "This kernel requires sm_80+ (cp.async); tuned for H100 (sm_90a)."
#endif

// ====================== BF16 <-> FP32 helpers ======================
__device__ __forceinline__ float bf16_to_f32(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(const float x) {
    return __float2bfloat16(x);
}

// ====================== Asynchronous 16B copy (8×bf16) ======================
static __device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
#else
    *((int4*)smem_ptr) = *((const int4*)gmem_ptr);
#endif
}
static __device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n");
#endif
}
static __device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

// ====================== Robust math helpers ======================
__device__ __forceinline__ float logaddexpf_robust(float a, float b) {
    float m = fmaxf(a, b);
    if (isinf(m)) return m; // handles +/-inf (we only expect -inf here)
    float d = fabsf(a - b);
    return m + log1pf(expf(-d));
}
__device__ __forceinline__ bool isclosef_relabs(float a, float b, float rtol, float atol) {
    if (isnan(a) || isnan(b)) return false;
    if (isinf(a) || isinf(b)) return a == b;  // +inf==+inf, -inf==-inf
    return fabsf(a - b) <= (atol + rtol * fabsf(b));
}

// ====================== LSE typed load/store helpers ======================
__device__ __forceinline__ float lse_load_as_f32(const float* p, int idx) {
    return p[idx];
}
__device__ __forceinline__ float lse_load_as_f32(const __half* p, int idx) {
    return __half2float(p[idx]);
}
__device__ __forceinline__ float lse_load_as_f32(const __nv_bfloat16* p, int idx) {
    return __bfloat162float(p[idx]);
}

__device__ __forceinline__ void lse_store_from_f32(float* p, int idx, float v) {
    p[idx] = v;
}
__device__ __forceinline__ void lse_store_from_f32(__half* p, int idx, float v) {
    p[idx] = __float2half_rn(v);
}
__device__ __forceinline__ void lse_store_from_f32(__nv_bfloat16* p, int idx, float v) {
    p[idx] = __float2bfloat16(v);
}

// ====================== Main kernel (in-place downdate) ======================
// Robust downdate: (out2, lse2) ⟵ downdate by (out1, lse1)
// Shapes: out1/out2: [M,D] (BF16), lse1/lse2: [M] (float|half|bf16)
// One CTA per group M = N*H
template<typename LSE1_T, typename LSE2_T>
__global__ void fused_kernel_bf16_downdate_inplace(
    const __nv_bfloat16* __restrict__ out1,   // [M, D] (BF16)
    __nv_bfloat16* __restrict__ out2,         // [M, D] (BF16)  (in-place write)
    const LSE1_T* __restrict__ lse1,          // [M]
    LSE2_T* __restrict__ lse2,                // [M]            (in-place write)
    int M, int D)
{
    int group = blockIdx.x;
    if (group >= M) return;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // Double-buffer layout: [out1_s0][out2_s0][out1_s1][out2_s1]
    int stride = TILE_D + SMEM_PAD;
    __nv_bfloat16* s_out1_stage0 = smem + 0;
    __nv_bfloat16* s_out2_stage0 = s_out1_stage0 + stride;
    __nv_bfloat16* s_out1_stage1 = s_out2_stage0 + stride;
    __nv_bfloat16* s_out2_stage1 = s_out1_stage1 + stride;

    int tid   = threadIdx.x;
    int warp  = tid >> 5;
    int lane  = tid & 31;

    const __nv_bfloat16* g_out1 = out1 + (size_t)group * D;
    __nv_bfloat16*       g_out2 = out2 + (size_t)group * D;

    // Load per-group LSE scalars as FP32
    float l1 = lse_load_as_f32(lse1, group);
    float l2 = lse_load_as_f32(lse2, group);

    // Robust downdate control scalars
    float lse_out_scalar;
    float ret_lse=0.f, t=0.f, scale=1.f;

    {
        // l1, l2 are float32 in NATURAL log (ln)
        constexpr float LAST_EPS_EXPM1 = 1e-7f;  // linear-domain closeness threshold

        float diff = l2 - l1;                    // expect <= 0 in normal downdate
        float em1  = expm1f(diff);               // = exp(diff) - 1, stable near 0

        // "last block" if exp(l2) and exp(l1) are almost equal (|exp(l2)/exp(l1) - 1| small)
        bool last_block = fabsf(em1) <= LAST_EPS_EXPM1;

        // "invalid" if l2 > l1 by more than the 'last_block' tolerance
        bool invalid    = (diff > 0.0f) && !last_block;
        if (last_block) {
            ret_lse = -INFINITY; t = 0.f; scale = 0.f; // revert to null
        } else if (invalid) {
            ret_lse = l1;        t = 0.f; scale = 1.f; // no-op
        } else {
            float diff = l2 - l1;                // <= 0
            float one_minus = 1.f - expf(diff);  // (0,1]
            ret_lse = l1 + logf(fmaxf(one_minus + 1e-7f, 1e-30f));
            t = expf(diff);                      // (0,1]
            scale = expf(l1 - ret_lse);          // >= 1
        }
        lse_out_scalar = ret_lse;
    }

    // Write LSE result in-place (dtype matches lse2)
    if (tid == 0) {
        lse_store_from_f32(lse2, group, lse_out_scalar);
    }

    int tiles = (D + TILE_D - 1) / TILE_D;
    int stage = 0;

    auto prefetch_tile = [&](int tile, int stage_id) {
        int tile_elems = min(TILE_D, D - tile * TILE_D);
        if (tile_elems <= 0) return;

        const __nv_bfloat16* g1 = g_out1 + tile * TILE_D;
        const __nv_bfloat16* g2 = g_out2 + tile * TILE_D;
        __nv_bfloat16* s1 = (stage_id==0)? s_out1_stage0 : s_out1_stage1;
        __nv_bfloat16* s2 = (stage_id==0)? s_out2_stage0 : s_out2_stage1;

        // 16B alignment check for cp.async
        uintptr_t src1_u = reinterpret_cast<uintptr_t>(g1);
        uintptr_t src2_u = reinterpret_cast<uintptr_t>(g2);
        uintptr_t dst1_u = reinterpret_cast<uintptr_t>(s1);
        uintptr_t dst2_u = reinterpret_cast<uintptr_t>(s2);
        bool async1 = ((src1_u | dst1_u) & 0xF) == 0;
        bool async2 = ((src2_u | dst2_u) & 0xF) == 0;

        int full16_bytes = (tile_elems / 8) * 16;
        int n16 = full16_bytes / 16;
        int rem_bf16 = tile_elems - (full16_bytes / 2);

        if (warp == 0) {
            // out1 path
            if (async1 && n16 > 0) {
                char* dst1 = reinterpret_cast<char*>(s1);
                const char* src1 = reinterpret_cast<const char*>(g1);
                for (int i = lane; i < n16; i += 32) {
                    cp_async_16B(dst1 + i*16, src1 + i*16);
                }
            } else {
                for (int e = lane; e < tile_elems; e += 32) s1[e] = g1[e];
            }
            // out2 path (read old values before we overwrite)
            if (async2 && n16 > 0) {
                char* dst2 = reinterpret_cast<char*>(s2);
                const char* src2 = reinterpret_cast<const char*>(g2);
                for (int i = lane; i < n16; i += 32) {
                    cp_async_16B(dst2 + i*16, src2 + i*16);
                }
            } else {
                for (int e = lane; e < tile_elems; e += 32) s2[e] = g2[e];
            }
            if ((async1 && n16 > 0) || (async2 && n16 > 0)) cp_async_commit();

            // copy remainders for async paths
            if (rem_bf16 > 0) {
                if (async1 && lane == 0) {
                    const __nv_bfloat16* g1_tail = g1 + (full16_bytes/2);
                    __nv_bfloat16* s1_tail = s1 + (full16_bytes/2);
                    for (int r = 0; r < rem_bf16; ++r) s1_tail[r] = g1_tail[r];
                }
                if (async2 && lane == 0) {
                    const __nv_bfloat16* g2_tail = g2 + (full16_bytes/2);
                    __nv_bfloat16* s2_tail = s2 + (full16_bytes/2);
                    for (int r = 0; r < rem_bf16; ++r) s2_tail[r] = g2_tail[r];
                }
            }
        }
    };

    if (tiles > 0) prefetch_tile(0, 0);
    cp_async_wait_all();
    __syncthreads();

    for (int tile = 0; tile < tiles; ++tile) {
        int next = tile + 1;
        int write_stage = stage ^ 1;
        if (next < tiles) prefetch_tile(next, write_stage);

        __nv_bfloat16* s1 = (stage==0)? s_out1_stage0 : s_out1_stage1;
        __nv_bfloat16* s2 = (stage==0)? s_out2_stage0 : s_out2_stage1;

        int tile_elems = min(TILE_D, D - tile * TILE_D);
        int g_off = tile * TILE_D;

        int vec2 = tile_elems >> 1;
        for (int i = threadIdx.x; i < vec2; i += blockDim.x) {
            int idx = i << 1;
            float f1a = bf16_to_f32(s1[idx + 0]);
            float f1b = bf16_to_f32(s1[idx + 1]);
            float f2a = bf16_to_f32(s2[idx + 0]);
            float f2b = bf16_to_f32(s2[idx + 1]);

            float o_a = (f1a - t * f2a) * scale;
            float o_b = (f1b - t * f2b) * scale;

            g_out2[g_off + idx + 0] = f32_to_bf16(o_a);
            g_out2[g_off + idx + 1] = f32_to_bf16(o_b);
        }

        if ((tile_elems & 1) && threadIdx.x == 0) {
            int idx = (vec2 << 1);
            float f1 = bf16_to_f32(s1[idx]);
            float f2 = bf16_to_f32(s2[idx]);
            float o = (f1 - t * f2) * scale;
            g_out2[g_off + idx] = f32_to_bf16(o);
        }

        cp_async_wait_all();
        __syncthreads();
        stage ^= 1;
    }
}

// ====================== Host wrapper (PyTorch stream aligned) ======================
static void check_same_device(const at::Tensor& a, const at::Tensor& b, const char* what) {
    TORCH_CHECK(a.device() == b.device(), what, ": tensors must be on the same device");
}

static inline void check_lse_dtype(const at::Tensor& t, const char* name) {
    auto dt = t.scalar_type();
    TORCH_CHECK(dt == at::kBFloat16 || dt == at::kFloat || dt == at::kHalf,
                name, " must be float32, bfloat16, or float16; got ", t.scalar_type());
}

template <typename LSE1_T, typename LSE2_T>
static void launch_kernel_inplace(
    const __nv_bfloat16* out1, __nv_bfloat16* out2,
    const LSE1_T* lse1, LSE2_T* lse2,
    int M, int D, cudaStream_t stream)
{
    dim3 grid((unsigned)M);
    dim3 block(THREADS);
    size_t smem_bytes = (size_t)((TILE_D + SMEM_PAD) * 4 * sizeof(__nv_bfloat16));

    fused_kernel_bf16_downdate_inplace<LSE1_T, LSE2_T>
        <<<grid, block, smem_bytes, stream>>>(
            out1, out2, lse1, lse2, M, D);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mac_update_then_downdate_bf16_inplace launch failed: ",
                cudaGetErrorString(err));
}

void mac_update_then_downdate_bf16_inplace(
    at::Tensor out1_bf16, at::Tensor lse1_any,
    at::Tensor out2_bf16, at::Tensor lse2_any)
{
    TORCH_CHECK(out1_bf16.is_cuda() && out2_bf16.is_cuda(), "out tensors must be CUDA");
    TORCH_CHECK(lse1_any.is_cuda() && lse2_any.is_cuda(),   "lse tensors must be CUDA");
    TORCH_CHECK(out1_bf16.scalar_type() == at::kBFloat16 && out2_bf16.scalar_type() == at::kBFloat16,
                "out tensors must be bfloat16");
    // Shape: [N,H,D] and [N,H]
    TORCH_CHECK(out1_bf16.dim() == 3, "out1 must be [N,H,D]");
    TORCH_CHECK(out2_bf16.sizes() == out1_bf16.sizes(), "out shapes must match [N,H,D]");
    TORCH_CHECK(lse1_any.dim() == 2, "lse1 must be [N,H]");
    TORCH_CHECK(lse2_any.sizes() == lse1_any.sizes(), "lse shapes must match [N,H]");

    // Require contiguous for fast striding (avoid creating new tensors)
    TORCH_CHECK(out1_bf16.is_contiguous() && out2_bf16.is_contiguous(),
                "out tensors must be contiguous");
    TORCH_CHECK(lse1_any.is_contiguous() && lse2_any.is_contiguous(),
                "lse tensors must be contiguous");

    // Same-device checks
    check_same_device(out1_bf16, out2_bf16, "out1/out2");
    check_same_device(out1_bf16, lse1_any,  "out1/lse1");
    check_same_device(lse1_any,  lse2_any,  "lse1/lse2");

    // Dtype checks
    check_lse_dtype(lse1_any, "lse1");
    check_lse_dtype(lse2_any, "lse2");

    // Guard device & use caller's current stream
    c10::cuda::CUDAGuard guard(out1_bf16.device());
    auto at_stream = at::cuda::getCurrentCUDAStream(out1_bf16.device().index());
    cudaStream_t stream = at_stream.stream();

    auto sizes = out1_bf16.sizes(); // [N,H,D]
    int64_t N = sizes[0], H = sizes[1], D = sizes[2];
    int64_t M = N * H;

    const __nv_bfloat16* out1 = reinterpret_cast<const __nv_bfloat16*>(out1_bf16.data_ptr<at::BFloat16>());
    __nv_bfloat16* out2       = reinterpret_cast<__nv_bfloat16*>(out2_bf16.data_ptr<at::BFloat16>());

    // Dispatch on (lse1 dtype, lse2 dtype) without creating temporaries
    auto dt1 = lse1_any.scalar_type();
    auto dt2 = lse2_any.scalar_type();

    if (dt1 == at::kFloat && dt2 == at::kFloat) {
        launch_kernel_inplace<float,float>(out1, out2,
            lse1_any.data_ptr<float>(), lse2_any.data_ptr<float>(), (int)M, (int)D, stream);
    } else if (dt1 == at::kFloat && dt2 == at::kHalf) {
        launch_kernel_inplace<float,__half>(out1, out2,
            lse1_any.data_ptr<float>(), reinterpret_cast<__half*>(lse2_any.data_ptr<at::Half>()),
            (int)M, (int)D, stream);
    } else if (dt1 == at::kFloat && dt2 == at::kBFloat16) {
        launch_kernel_inplace<float,__nv_bfloat16>(out1, out2,
            lse1_any.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(lse2_any.data_ptr<at::BFloat16>()),
            (int)M, (int)D, stream);
    } else if (dt1 == at::kHalf && dt2 == at::kFloat) {
        launch_kernel_inplace<__half,float>(out1, out2,
            reinterpret_cast<const __half*>(lse1_any.data_ptr<at::Half>()), lse2_any.data_ptr<float>(),
            (int)M, (int)D, stream);
    } else if (dt1 == at::kHalf && dt2 == at::kHalf) {
        launch_kernel_inplace<__half,__half>(out1, out2,
            reinterpret_cast<const __half*>(lse1_any.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(lse2_any.data_ptr<at::Half>()), (int)M, (int)D, stream);
    } else if (dt1 == at::kHalf && dt2 == at::kBFloat16) {
        launch_kernel_inplace<__half,__nv_bfloat16>(out1, out2,
            reinterpret_cast<const __half*>(lse1_any.data_ptr<at::Half>()),
            reinterpret_cast<__nv_bfloat16*>(lse2_any.data_ptr<at::BFloat16>()), (int)M, (int)D, stream);
    } else if (dt1 == at::kBFloat16 && dt2 == at::kFloat) {
        launch_kernel_inplace<__nv_bfloat16,float>(out1, out2,
            reinterpret_cast<const __nv_bfloat16*>(lse1_any.data_ptr<at::BFloat16>()), lse2_any.data_ptr<float>(),
            (int)M, (int)D, stream);
    } else if (dt1 == at::kBFloat16 && dt2 == at::kHalf) {
        launch_kernel_inplace<__nv_bfloat16,__half>(out1, out2,
            reinterpret_cast<const __nv_bfloat16*>(lse1_any.data_ptr<at::BFloat16>()),
            reinterpret_cast<__half*>(lse2_any.data_ptr<at::Half>()), (int)M, (int)D, stream);
    } else { // (BF16, BF16)
        launch_kernel_inplace<__nv_bfloat16,__nv_bfloat16>(out1, out2,
            reinterpret_cast<const __nv_bfloat16*>(lse1_any.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(lse2_any.data_ptr<at::BFloat16>()), (int)M, (int)D, stream);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mac_update_then_downdate_bf16_inplace", &mac_update_then_downdate_bf16_inplace,
          "In-place fused update then robust downdate; out1/out2 [N,H,D] BF16, lse1/lse2 [N,H] (f32/bf16/fp16). "
          "Writes results into out2 (BF16) and lse2 (same dtype as lse2).");
}
