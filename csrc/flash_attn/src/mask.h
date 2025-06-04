/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

namespace flash {

using namespace cute;

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                                           const int col_idx_offset_ = 0, const int warp_col_stride = 16) {
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    static_assert(decltype(size<0, 0>(tensor))::value == 1);
    static_assert(decltype(size<1, 0>(tensor))::value == 4);
    const int col_idx_offset = col_idx_offset_ + ((__lane_id() >> 4) << 2);
#pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * warp_col_stride;
#pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
            if (col_idx >= max_seqlen_k) {
#pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <bool HasWSLeft = true, typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_local(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                                 const int max_seqlen_k, const int row_idx_offset,
                                                 const int max_seqlen_q, const int warp_row_stride,
                                                 const int window_size_left, const int window_size_right,
                                                 const int warp_col_stride = 16) {
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    static_assert(decltype(size<0, 0>(tensor))::value == 1);
    static_assert(decltype(size<1, 0>(tensor))::value == 4);
    const int col_idx_offset = col_idx_offset_ + ((__lane_id() >> 4) << 2);
#pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx = row_idx_offset + mi * warp_row_stride;
        const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
        const int col_idx_limit_right =
            std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
#pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const int col_idx_base = col_idx_offset + nj * warp_col_stride;
#pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j;
                if (col_idx >= col_idx_limit_right || (HasWSLeft && col_idx < col_idx_limit_left)) {
                    tensor(make_coord(0, mi), make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_causal(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                                  const int max_seqlen_k, const int row_idx_offset,
                                                  const int max_seqlen_q, const int warp_row_stride,
                                                  const int warp_col_stride = 16) {
    apply_mask_local<false>(tensor, col_idx_offset_, max_seqlen_k, row_idx_offset, max_seqlen_q, warp_row_stride, -1, 0,
                            warp_col_stride);
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void apply_mask_causal_w_idx(Tensor<Engine0, Layout0> &tensor,
                                                        Tensor<Engine1, Layout1> const &idx_rowcol,
                                                        const int col_idx_offset_, const int max_seqlen_k,
                                                        const int row_idx_offset) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 2, "Only support 2D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(idx_rowcol));
    CUTE_STATIC_ASSERT_V(size<1>(tensor) == size<1>(idx_rowcol));
#pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const int col_idx_limit = std::min(max_seqlen_k, 1 + row_idx_offset + get<0>(idx_rowcol(mi, 0)));
#pragma unroll
        for (int ni = 0; ni < size<1, 1>(tensor); ++ni) {
            if (col_idx_offset_ + get<1>(idx_rowcol(0, ni)) >= col_idx_limit) {
                tensor(mi, ni) = -INFINITY;
            }
        }
    }
}

template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {
    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q, const int window_size_left,
                                    const int window_size_right, const float alibi_slope = 0.f)
        : max_seqlen_k(max_seqlen_k),
          max_seqlen_q(max_seqlen_q),
          window_size_left(window_size_left),
          window_size_right(window_size_right),
          alibi_slope(!Has_alibi ? 0.0 : alibi_slope){};

    template <bool Causal_mask = false, bool Is_even_MN = true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_, const int col_idx_offset_,
                                               const int row_idx_offset, const int warp_row_stride) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || Is_local || !Is_even_MN;

        if constexpr (Need_masking) {
            Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));

            static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;
            const int col_idx_offset = col_idx_offset_ + ((__lane_id() >> 4) << 2);
            if constexpr (Col_idx_only) {
#pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 16;
#pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
#pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            if constexpr (Has_alibi) {
                                tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                            }
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                }
                            }
                        }
                    }
                }
            } else {
#pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
#pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 16;
                        const int col_idx_limit_left =
                            std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                        const int col_idx_limit_right =
                            std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
#pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 16;
#pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                const int col_idx = col_idx_base + j;
                                if constexpr (Has_alibi) {
                                    if constexpr (Is_causal) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) += alibi_slope * col_idx;
                                    } else {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) -=
                                            alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                                    }
                                }
                                if constexpr (Causal_mask) {
                                    if (col_idx >= col_idx_limit_right) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                                if constexpr (Is_local) {
                                    if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                                if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                    if (col_idx >= max_seqlen_k) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };
};

}  // namespace flash
