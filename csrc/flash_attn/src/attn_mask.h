
#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <mctlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

template <typename Engine, typename Layout, typename T>
inline __device__ void apply_attn_mask(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                       const int max_seqlen_k, const int row_idx_offset_, const int max_seqlen_q,
                                       const int warp_row_stride, const int warp_col_stride, const float softmax_scale,
                                       const T *bias, const int bias_row_stride, const int bias_col_stride) {
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 64;
    const int row_idx_offset = row_idx_offset_;
    const int col_idx_offset = col_idx_offset_ + (lane_id / 16) * 4;
#pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx = row_idx_offset + mi * warp_row_stride;
#pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const int col_idx_base = col_idx_offset + nj * warp_col_stride;
#pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j;
                if (col_idx < max_seqlen_k && row_idx < max_seqlen_q) {
                    tensor(make_coord(0, mi), make_coord(j, nj)) +=
                        *(bias + row_idx * bias_row_stride + col_idx * bias_col_stride) / softmax_scale;
                }
            }
        }
    }
}

}  // namespace flash