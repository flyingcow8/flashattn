/***************************************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>

#include <mctlass/mctlass.h>
#include <mctlass/array.h>
#include <mctlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <bool Clear_dQaccum = true, typename Kernel_traits, typename Params>
inline __device__ void compute_dot_do_o_hdim128_32_32(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int m_block = blockIdx.x;

    const int bidb = blockIdx.z;

    const int bidh = blockIdx.y;

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const BlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb) +
                                  m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq_accum =
        binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb) +
        (m_block * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) * params.h * params.d_rounded +
        bidh * params.d_rounded;
    const index_t row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM;

    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
    Tensor gdQaccum =
        make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                    Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.h * params.d_rounded, _1{}));
    Tensor dP_sum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);

    typename Kernel_traits::GmemTiledCopydQaccum_hdim128_32_32 gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    Tensor cdO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tdOcdO = gmem_thr_copy_dO.partition_S(cdO);

    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdOgdO)));

#pragma unroll
    for (int k = 0; k < size(tdOpdO); ++k) {
        tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d;
    }

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);
    flash::copy<false, false, true>(gmem_tiled_copy_dO, tdOgdO, tdOrdO, tdOcdO, tdOpdO,
                                    binfo.actual_seqlen_q - m_block * kBlockM);
    flash::copy<false, false, true>(gmem_tiled_copy_dO, tdOgO, tdOrO, tdOcdO, tdOpdO,
                                    binfo.actual_seqlen_q - m_block * kBlockM);

    dot_do_o<Kernel_traits::kGmemThreadsPerRow>(
        tdOrdO, tdOrO, dP_sum, Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
    if (Clear_dQaccum) {
        Tensor zero = make_fragment_like(tdQgdQaccum);
        clear(zero);
        cute::copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum);
    }
}

template <typename Kernel_traits, typename Params>
inline __device__ void convert_dQ_hdim128_32_32(const Params &params, const int nsplits) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];

    const int m_block = blockIdx.x;

    const int bidb = blockIdx.z;

    const int bidh = blockIdx.y;

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const bool is_tile32x32 = (Kernel_traits::kBlockN == 32 && Kernel_traits::kBlockM == 32);

    const BlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb) +
                                  m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum =
        binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb) +
        (m_block * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) * params.h * params.d_rounded +
        bidh * params.d_rounded;

    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum =
        make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                    Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.h * params.d_rounded, _1{}));

    Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutdQ{});

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd_hdim128_32_32 gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);

    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    Tensor tdQrdQaccum = make_fragment_like(tdQgdQaccum);
    clear(acc_dq);

#pragma unroll
    for (int s = 0; s < nsplits; ++s) {
        cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, tdQrdQaccum);
#pragma unroll
        for (int i = 0; i < size(acc_dq); ++i) {
            acc_dq(i) += tdQrdQaccum(i);
        }
        tdQgdQaccum.data() = tdQgdQaccum.data() + params.dq_accum_split_stride;
    }

#pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) {
        acc_dq(i) *= params.scale_softmax_rp_dropout;
    }

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_dq, rdQ)

    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);

    Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
#pragma unroll
    for (int k = 0; k < size(tdQpdQ); ++k) {
        tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d;
    }

    flash::copy<false, false, false, false>(gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
}

}  // namespace flash
