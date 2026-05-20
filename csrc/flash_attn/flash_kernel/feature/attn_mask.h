
#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <mctlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Params>
inline __host__ __device__ bool use_attn_mask_merge_ldg(Params &params) {
    /*merged impl require
        1. bias_col_shape % 4 == 0
        2. bias_col_stride == 1
    */
    if((params.attn_mask_col_shape % 4) == 0
        && params.attn_mask_col_stride == 1) {
        return true;
    } else {
        return false;
    }
}

template <bool mergeLdg=false, bool Is_even_MN=false,typename Engine, typename Layout, typename T>
inline __device__ void apply_attn_mask(Tensor<Engine, Layout> &tensor,
                                  const int col_idx_offset_,
                                  const int max_seqlen_k,
                                  const int row_idx_offset_,
                                  const int max_seqlen_q,
                                  const int warp_row_stride,
                                  const int warp_col_stride,
                                  const float softmax_scale,
                                  T *bias,
                                  const int bias_row_stride,
                                  const int bias_col_stride) {
    // tensor has shape (nrow=(1, MMA_M), ncol=(4, MMA_N)
    CUTE_STATIC_ASSERT_V((size<1, 0>(tensor)) == Int<4>{});
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    typedef __NATIVE_VECTOR__(2, int) VecType;
    const int lane_id = threadIdx.x % 64;
    const int row_idx_offset = row_idx_offset_;
    const int col_idx_offset = col_idx_offset_ + (lane_id / 16) * 4;
    if constexpr (mergeLdg) {
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * warp_col_stride;
                T bias_16[4];
                if constexpr (Is_even_MN) {
                    uint64_t *bias_64 = reinterpret_cast<uint64_t *>(bias_16);
                    bias_64[0] = *((uint64_t *)(bias + row_idx * bias_row_stride + col_idx_base));
                } else {
                    bool mask = row_idx < max_seqlen_q && col_idx_base < max_seqlen_k;
                    VecType *dst_ptr = reinterpret_cast<VecType *>(bias_16);
                    VecType *src_ptr = reinterpret_cast<VecType *>(bias + row_idx * bias_row_stride + col_idx_base);
                    *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                                mask, 1, MACA_ICMP_EQ);
                }
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    if (row_idx < max_seqlen_q && col_idx_base < max_seqlen_k) {
                        tensor(make_coord(0, mi), make_coord(j, nj)) += bias_16[j] / softmax_scale;
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx = row_idx_offset + mi * warp_row_stride;
            bool row_mask = Is_even_MN || row_idx < max_seqlen_q;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * warp_col_stride;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    /*naive impl will have ldg_u16*/
                    const int col_idx = col_idx_base + j;
                    if (row_mask && col_idx < max_seqlen_k) {
                        tensor(make_coord(0, mi), make_coord(j, nj)) += *(bias + row_idx * bias_row_stride + col_idx * bias_col_stride) / softmax_scale;
                    }
                }
            }
        }
    }
}

template <typename Tensor0, typename Tensor1>
inline __device__ void apply_attn_mask(Tensor0& acc_s, Tensor1& tSrMask, const float softmax_scale) {
    // acc_s shape is (4, MMA_M, MMA_N)
    CUTE_STATIC_ASSERT_V(size<0>(acc_s) == _4{});
    CUTE_STATIC_ASSERT_V(size<0>(acc_s) == size<0>(tSrMask));
    CUTE_STATIC_ASSERT_V(size<1>(acc_s) == size<1>(tSrMask));
    CUTE_STATIC_ASSERT_V(size<2>(acc_s) == size<2>(tSrMask));
    using T = typename Tensor1::value_type;
    typedef __NATIVE_VECTOR__(2, float) Float2;
    CONVERT_TENSOR_TYPE(T, float, tSrMask, rMask)
    float scale = 1 / softmax_scale;
    Float2 scale_vec = {scale, scale};
    #pragma unroll
    for (int m = 0; m < size<1>(acc_s); m++) {
        #pragma unroll
        for (int n = 0; n < size<2>(acc_s); n++) {
            #pragma unroll
            for (int i = 0; i < size<0>(acc_s); i += 2) {
                Float2 x_vec = {rMask(i, m, n), rMask(i + 1, m, n)};
                Float2 y_vec = {acc_s(i, m, n), acc_s(i + 1, m, n)};
                y_vec =  __builtin_mxc_pk_fma_f32(x_vec, scale_vec, y_vec);
                acc_s(i, m, n) = y_vec[0];
                acc_s(i + 1, m, n) = y_vec[1];
            }
        }
    }
}

template <bool mergeLdg=false, bool Is_even_M=false, bool Is_even_N=false, bool is_async = false, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void load_attn_mask(Tensor0& tSgMask, Tensor1& tSrMask, Tensor2& tScMask, const int max_N, const int max_M) {
    // load attn_mask bias from global -> register
    // tSgMask shape is (4, MMA_M, MMA_N)
    CUTE_STATIC_ASSERT_V(size<0>(tSgMask) == _4{});
    CUTE_STATIC_ASSERT_V(size<0>(tSgMask) == size<0>(tSrMask));
    CUTE_STATIC_ASSERT_V(size<1>(tSgMask) == size<1>(tSrMask));
    CUTE_STATIC_ASSERT_V(size<2>(tSgMask) == size<2>(tSrMask));
    typedef __NATIVE_VECTOR__(2, int) VecType;
    if constexpr (mergeLdg) {
        #pragma unroll
        for (int m = 0; m < size<1>(tSgMask); m++) {
            bool row_mask = Is_even_M || get<0>(tScMask(0, m, 0)) < max_M;
            #pragma unroll
            for (int n = 0; n < size<2>(tSgMask); n++) {
                bool col_mask = Is_even_N || get<1>(tScMask(0, 0, n)) < max_N;
                auto src_ptr = (VecType *)(tSgMask(_, m, n).data().get());    // gmem
                auto dst_ptr = (VecType *)(tSrMask(_, m, n).data());          // rf
                if constexpr (Is_even_M && Is_even_N) {
                    *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, is_async);
                } else {
                    *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, is_async,
                                                            row_mask && col_mask, 1, MACA_ICMP_EQ);
                }
            }
        }
    } else {
        typedef __NATIVE_VECTOR__(1, unsigned short) v1u16;
        #pragma unroll
        for (int m = 0; m < size<1>(tSgMask); m++) {
            bool row_mask = Is_even_M || get<0>(tScMask(0, m, 0)) < max_M;
            #pragma unroll
            for (int n = 0; n < size<2>(tSgMask); n++) {
                int col_base_idx = get<1>(tScMask(0, 0, n));
                #pragma unroll
                for (int i = 0; i < size<0>(tSgMask); i++) {
                    bool col_mask = Is_even_N || col_base_idx + i < max_N;
                    auto src_ptr = (v1u16 *)(&tSgMask(i, m, n));    // gmem
                    auto dst_ptr = (v1u16 *)(&tSrMask(i, m, n));    // rf
                    //     *dst_ptr = __builtin_mxc_ldg_b16(src_ptr, 0, -1, true, true, false, is_async);
                    // } else {
                    //     *dst_ptr = __builtin_mxc_ldg_b16_predicator(src_ptr, 0, true, true, false, is_async,
                    //                                             row_mask && col_mask, 1, MACA_ICMP_EQ);
                    // }
                    // ********************************************
                    *dst_ptr = __builtin_mxc_ldg_b16_predicator(src_ptr, 0, true, true, false, is_async,
                                                                row_mask && col_mask, 1, MACA_ICMP_EQ);
                }
            }
        }
    }
}

}  // namespace flash
