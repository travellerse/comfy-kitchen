// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// RDNA2 W4A4 GEMM using packed signed dot8 instructions (v_dot8_i32_i4).
//
// Both operands are ConvRot W4A4 packed rows: K / 2 bytes, low nibble = even K,
// high nibble = odd K, signed two's-complement int4 with q in [-7, 7]. The dot8
// instruction consumes the packed nibbles directly, so there is no unpack kernel.
// K must be a multiple of 32 so a packed row stages in whole 16-byte chunks.
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../gemm_wmma.h"

namespace comfy::hip_backend {

template <typename OutT, int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(256) void gemm_w4a4_dp4a_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    OutT* __restrict__ C, int M, int N, int Kbytes, EpiRowwise epi) {

    constexpr int kThreads = 256;
    constexpr int kThreadCols = BN / TN;
    constexpr int kStride = BK + kLdsPad;
    static_assert(BM == (kThreads / kThreadCols) * TM);
    static_assert(BN == kThreadCols * TN);
    static_assert(BK % 16 == 0);

    __shared__ __align__(16) uint8_t As[BM * kStride];
    __shared__ __align__(16) uint8_t Bs[BN * kStride];

    const int tid = threadIdx.x;
    const int thread_row = tid / kThreadCols;
    const int thread_col = tid % kThreadCols;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    int acc[TM][TN] = {};
    TileStager<BM, BK, kThreads> sa;
    TileStager<BN, BK, kThreads> sb;

    if (Kbytes > 0) {
        sa.load(A, m0, M, 0, Kbytes);
        sb.load(B, n0, N, 0, Kbytes);
        sa.store(As);
        sb.store(Bs);
        __syncthreads();
    }

    for (int k0 = 0; k0 < Kbytes; k0 += BK) {
        const int knext = k0 + BK;
        const bool has_next = knext < Kbytes;
        if (has_next) {
            sa.load(A, m0, M, knext, Kbytes);
            sb.load(B, n0, N, knext, Kbytes);
        }

        #pragma unroll 1
        for (int kk = 0; kk < BK; kk += 4) {
            int av[TM];
            int bv[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row = thread_row + i * (BM / TM);
                av[i] = *reinterpret_cast<const int*>(As + row * kStride + kk);
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = thread_col + j * (BN / TN);
                bv[j] = *reinterpret_cast<const int*>(Bs + col * kStride + kk);
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
#if defined(COMFY_HAS_WMMA)
                    // gfx11/gfx12 compile this source for the combined wheel but
                    // never launch it: dispatch keeps the W4A4 sdot8 path on RDNA2,
                    // where the dot1-insts feature exists. A scalar nibble fallback
                    // keeps the kernel compiling where the intrinsic does not.
                    const uint8_t* av8 = reinterpret_cast<const uint8_t*>(&av[i]);
                    const uint8_t* bv8 = reinterpret_cast<const uint8_t*>(&bv[j]);
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int an = (av8[e >> 1] >> ((e & 1) * 4)) & 0xF;
                        const int bn = (bv8[e >> 1] >> ((e & 1) * 4)) & 0xF;
                        acc[i][j] += (an >= 8 ? an - 16 : an) * (bn >= 8 ? bn - 16 : bn);
                    }
#else
                    acc[i][j] = __builtin_amdgcn_sdot8(av[i], bv[j], acc[i][j], false);
#endif
                }
            }
        }

        if (has_next) {
            __syncthreads();
            sa.store(As);
            sb.store(Bs);
            __syncthreads();
        }
    }

    epi.init();
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = m0 + thread_row + i * (BM / TM);
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + thread_col + j * (BN / TN);
            if (col < N) {
                C[static_cast<int64_t>(row) * N + col] =
                    static_cast<OutT>(epi(row, col, static_cast<float>(acc[i][j])));
            }
        }
    }
}

template <typename OutT>
void launch_w4a4_dp4a(const uint8_t* A, const uint8_t* B, OutT* C,
                      int M, int N, int Kbytes, EpiRowwise epi, hipStream_t stream) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 128;
    constexpr int TM = 4;
    constexpr int TN = 4;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_w4a4_dp4a_kernel<OutT, BM, BN, BK, TM, TN>
        <<<grid, 256, 0, stream>>>(A, B, C, M, N, Kbytes, epi);
}

}  // namespace comfy::hip_backend

extern "C" void launch_w4a4_gemm_dp4a_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M == 0 || N == 0) return;
    // A packed row is K / 2 bytes and stages in 16-byte chunks, so K must be a
    // whole number of 32-element chunks.
    if (K % 32 != 0) {
        throw std::runtime_error("w4a4_gemm_dp4a: K=" + std::to_string(K) +
                                 " must be a multiple of 32");
    }

    EpiRowwise epi{static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
                   scale_b_stride, bias, bias_code};
    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);

    if (out_code == kBF16) {
        launch_w4a4_dp4a(A, B, static_cast<__bf16*>(c), M, N, K / 2, epi, stream);
    } else if (out_code == kF16) {
        launch_w4a4_dp4a(A, B, static_cast<__half*>(c), M, N, K / 2, epi, stream);
    } else if (out_code == kF32) {
        launch_w4a4_dp4a(A, B, static_cast<float*>(c), M, N, K / 2, epi, stream);
    } else {
        throw std::runtime_error("w4a4_gemm_dp4a: unsupported output dtype");
    }
}
