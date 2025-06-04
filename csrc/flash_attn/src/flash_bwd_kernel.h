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
#include "softmax.h"
#include "mask.h"
#include "dropout.h"

#include "alibi.h"
#include "bwd_hdim64_utils.h"
#include "flash_bwd_kernel_hdim64_64x64_8waves.h"
#include "flash_bwd_kernel_hdim64_64x64_4waves.h"
#include "flash_bwd_kernel_hdim128_32x64.h"
#include "flash_bwd_kernel_hdim128_32x128.h"

namespace flash {

using namespace cute;

enum class FlashBwdKernels {
    kBwdKernel_hdim128_32x64_256,
    kBwdKernel_hdim128_32x128_512,
};

template <int MMA_N, class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto make_tiled_copy_B_warpcontiguousN(Copy_Atom<Args...> const &copy_atom,
                                                        TiledMMA const &tiled_mma) {
    using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
    constexpr int TileShape_N = decltype(size<1>(TileShape_MNK{}))::value;
    constexpr int TileShape_K = decltype(size<2>(TileShape_MNK{}))::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;

    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;

    auto t = make_tile(Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>, Stride<_1, Int<MMAStride_N>, _8>>{},
                       make_layout(Int<TileShape_K>{}));

    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutB_TV(), t);
}

template <int MMA_N, class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto make_tiled_copy_C_warpcontiguousN(Copy_Atom<Args...> const &copy_atom,
                                                        TiledMMA const &tiled_mma) {
    using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
    constexpr int TileShape_M = decltype(size<0>(TileShape_MNK{}))::value;
    constexpr int TileShape_N = decltype(size<1>(TileShape_MNK{}))::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;

    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    auto t = make_tile(make_layout(Int<TileShape_M>{}),
                       Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>, Stride<_1, Int<MMAStride_N>, _8>>{});

    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Is_first, bool Is_last, bool Seq_parallel = false, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock(const Params &params, const int bidb, const int bidh,
                                                  const int n_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(typename Kernel_traits::TiledMmaSdP::TiledShape_MNK{}))::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
    if (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q -
                                                               binfo.actual_seqlen_k + params.window_size_left,
                                                           kBlockM));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 n_block * kBlockN * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 n_block * kBlockN * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb) +
                                  (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb) +
                                  (m_block_max - 1) * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum =
        binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb) +
        ((m_block_max - 1) * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) * params.h *
            params.d_rounded +
        bidh * params.d_rounded

        + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + (m_block_max - 1) * kBlockM;
    const index_t row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded + (m_block_max - 1) * kBlockM;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum =
        make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                    Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.h * params.d_rounded, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});

    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sdS = make_tensor(!Kernel_traits::Is_V_in_regs ? sV.data() + size(sV) : sK.data() + size(sK),
                             typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});

    Tensor sdQ = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutdQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO =
        std::conditional_t<Is_first, typename Kernel_traits::GmemTiledCopydO, typename Kernel_traits::GmemTiledCopyQKV>;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    using GmemLayoutAtomdQaccum = std::conditional_t<!Seq_parallel, typename Kernel_traits::GmemTiledCopydQaccum,
                                                     typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd>;
    GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);

    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);

    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    auto smem_tiled_copy_PdS = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);

    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);

    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
#pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }

    tdQgdQ.data() = tdQgdQ.data() + kBlockM * params.dq_row_stride;
    tdQgdQaccum.data() = tdQgdQaccum.data() + kBlockM * params.h * params.d_rounded;

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local) ? 0
                                                : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q -
                                                               binfo.actual_seqlen_k - params.window_size_right) /
                                                                  kBlockM);

    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb) +
                                      n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb) +
                                      n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
#pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) {
            tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d;
        }

        flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV,
                                                         binfo.actual_seqlen_k - n_block * kBlockN);
        flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV,
                                                         binfo.actual_seqlen_k - n_block * kBlockN);
        return;
    }

    if (Double_buffer && m_block % 2 == 1) {
        tQsQ.data() = tQsQ.data() + size(sQ);
        tSsQ.data() = tSsQ.data() + size(sQ);
        tdKsQt.data() = tdKsQt.data() + size(sQ);
    }

    if ((!Is_first && !Seq_parallel) || params.deterministic) {
        __syncthreads();
    }

    if (Kernel_traits::Is_V_in_regs) {
        flash::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV,
                                                 binfo.actual_seqlen_k - n_block * kBlockN);
        flash::cp_async_fence();
    }

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);
    if (!Is_first) {
        flash::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ,
                                                 binfo.actual_seqlen_q - m_block * kBlockM);
    } else {
        flash::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ,
                                                 binfo.actual_seqlen_q - m_block * kBlockM);
        flash::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ,
                                                 binfo.actual_seqlen_q - m_block * kBlockM);
    }
    flash::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                             binfo.actual_seqlen_q - m_block * kBlockM);

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);
    static_assert(decltype(size<0>(taccScS))::value == 4);

    Tensor taccScS_row = logical_divide(taccScS, Shape<_4>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    flash::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                             binfo.actual_seqlen_k - n_block * kBlockN);
    if (!Kernel_traits::Is_V_in_regs) {
        flash::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV,
                                                 binfo.actual_seqlen_k - n_block * kBlockN);
    }
    flash::cp_async_fence();

    if (Is_first) {
        cute::copy(tdOrdO, tdOsdO);
        dot_do_o<Kernel_traits::kGmemThreadsPerRow>(
            tdOrdO, tdOrO, gdPsum, Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
    }

    if (Kernel_traits::Is_V_in_regs) {
        cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    clear(acc_dv);
    clear(acc_dk);

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    for (; m_block >= m_block_min; --m_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        cute::cp_async_wait<0>();
        __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi)));
        }

        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp, smem_tiled_copy_QdO, smem_tiled_copy_KV,
                    smem_thr_copy_QdO, smem_thr_copy_KV);

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        if (Has_alibi) {
            alibi.apply_alibi(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * MMA_N_SdP * 16,
                              m_block * kBlockM + get<0>(taccScS_row(0)), AtomLayoutMS * 16);
        }

        if (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                flash::apply_mask(scores, binfo.actual_seqlen_k,
                                  n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * MMA_N_SdP * 16);
            }
        } else if (Is_causal) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k ||
                (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                flash::apply_mask_causal(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * MMA_N_SdP * 16,
                                         binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                         binfo.actual_seqlen_q,

                                         AtomLayoutMS * 16);
            }
        } else if (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k -
                                        params.window_size_right ||
                (m_block + 1) * kBlockM >=
                    n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left ||
                (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                flash::apply_mask_local(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16, params.window_size_left,
                                        params.window_size_right);
            }
        }

        flash::scale_apply_exp2<false>(scores, lse, params.scale_softmax_log2);
        if (Is_dropout) {
            int warp_id = tidx / 64;
            int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;

            int block_col_idx = n_block * (kBlockN / 64);
            dropout.template mc_apply_dropout<true>(acc_s, block_row_idx, block_col_idx, AtomLayoutMS, kBlockN,
                                                    n_block);
        }

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        if constexpr (Is_dropout) {
            flash::relu_(rP);
        }

        Tensor tPrP = make_tensor(rP.data(), acc_s.layout());
        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);

        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));

        clear(acc_dp);

        flash::gemm<false, Kernel_traits::Is_V_in_regs>(acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
                                                        smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO,
                                                        smem_thr_copy_KV);

        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) { return p * (!Is_dropout || p >= 0 ? dp - d : d); };
#pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
#pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                dS(mi, ni) = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
            }
        }

        Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
        tdQgdQaccum.data() = tdQgdQaccum.data() + (-int(kBlockM * params.h * params.d_rounded));
        if (Is_first || Seq_parallel) {
            clear(acc_dq);
        } else {
            Tensor acc_dq_reshaped = make_tensor(
                acc_dq.data(), make_layout(get<0>(acc_dq.layout()), get<2>(acc_dq.layout()), get<1>(acc_dq.layout())));
            cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, acc_dq_reshaped);
        }

        if (Double_buffer && m_block > m_block_min) {
            const int sQ_offset = m_block % 2 == 0 ? size(sQ) : -size(sQ);
            tQsQ.data() = tQsQ.data() + sQ_offset;
            tSsQ.data() = tSsQ.data() + sQ_offset;

            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            flash::cp_async_fence();
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());

        CONVERT_TENSOR_TYPE(ElementAccum, Element, dS_reshaped, tdSrdS)

        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        __syncthreads();

        flash::gemm(acc_dv, tdVrPt, tdVrdO, tdVsPt, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_PdSt, smem_tiled_copy_QdOt,
                    smem_thr_copy_PdSt, smem_thr_copy_QdOt);

        __syncthreads();

        if (m_block > m_block_min) {
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            if (Is_first) {
                tdOgO.data() = tdOgO.data() + (-int(kBlockM * params.o_row_stride));
                flash::copy<true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ);
                flash::copy<true, Is_even_K>(gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ);
            } else {
                flash::copy<true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ);
                flash::cp_async_fence();
            }
        }

        flash::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq, smem_tiled_copy_dS, smem_tiled_copy_Kt,
                    smem_thr_copy_dS, smem_thr_copy_Kt);

        if (m_block > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
#pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                lse(mi) = gLSE(get<0>(taccScS_row(mi)));
            }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
        }

        if (!Is_last) {
            Tensor acc_dq_reshaped = make_tensor(
                acc_dq.data(), make_layout(get<0>(acc_dq.layout()), get<2>(acc_dq.layout()), get<1>(acc_dq.layout())));
            if (!Seq_parallel) {
                cute::copy(gmem_tiled_copy_dQaccum, acc_dq_reshaped, tdQgdQaccum);
            } else {
                CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));
#pragma unroll
                for (int i = 0; i < size(acc_dq); ++i) {
                    atomicAdd(&tdQgdQaccum(i), acc_dq(i));
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < size(acc_dq); ++i) {
                acc_dq(i) *= params.scale_softmax_rp_dropout;
            }

            CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_dq, rdQ)
            Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
            cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
        }

        flash::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_PdSt, smem_tiled_copy_QdOt,
                    smem_thr_copy_PdSt, smem_thr_copy_QdOt);

        if (Double_buffer) {
            tdKsQt.data() = tdKsQt.data() + (m_block % 2 == 0 ? size(sQ) : -size(sQ));
        }
        if (!Double_buffer && m_block > m_block_min) {
            __syncthreads();

            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            flash::cp_async_fence();
        }

        if (Is_first && m_block > m_block_min) {
            cute::copy(tdOrdO, tdOsdO);
            dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, gdPsum,
                                                        Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow),
                                                        params.p_dropout);
        }

        if (Is_last) {
            __syncthreads();
            Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
            cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
            tdQgdQ.data() = tdQgdQ.data() + (-int(kBlockM * params.dq_row_stride));
            Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
            Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
#pragma unroll
            for (int m = 0; m < size<1>(tdQgdQ); ++m) {
                if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM) {
                    cute::copy(gmem_tiled_copy_dQ, tdQrdQ(_, m, _), tdQgdQ(_, m, _));
                }
            }
        }
    }

    if (Is_dropout) {
#pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) {
            acc_dv(i) *= params.rp_dropout;
        }
    }
#pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) {
        acc_dk(i) *= params.scale_softmax_rp_dropout;
    }

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_dk, rdK)
    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_dv, rdV)

    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{});

    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);

    if (!Is_last) {
        __syncthreads();
    }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb) +
                                  n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb) +
                                  n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
#pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) {
        tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d;
    }

    flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV,
                                                     binfo.actual_seqlen_k - n_block * kBlockN);
    flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV,
                                                     binfo.actual_seqlen_k - n_block * kBlockN);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Has_attn_mask, bool Is_even_M,
          bool Is_even_K, typename Params>
inline __device__ void compute_dq_dk_dv(const Params &params) {
    const int bidb = blockIdx.x;

    const int bidh = blockIdx.y;

    const int tidx = threadIdx.x;

    const int n_block_max = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    if (n_block_max == 1) {
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Has_attn_mask, Is_even_M, Is_even_K,
                                   true, true>(params, bidb, bidh, 0);
    } else {
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Has_attn_mask, Is_even_M, Is_even_K,
                                   true, false>(params, bidb, bidh, n_block_max - 1);
        for (int n_block = n_block_max - 2; n_block > 0; n_block--) {
            compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Has_attn_mask, Is_even_M,
                                       Is_even_K, false, false>(params, bidb, bidh, n_block);
        }
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Has_attn_mask, Is_even_M, Is_even_K,
                                   false, true>(params, bidb, bidh, 0);
    }
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Is_deterministic, typename Params>
inline __device__ void compute_dq_dk_dv_seqk_parallel(const Params &params) {
    int bidh = blockIdx.y;
    int bidb = blockIdx.z;
    if constexpr (Is_deterministic) {
        for (int n_block = blockIdx.x;
             n_block < (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN; n_block += gridDim.x) {
            if constexpr (Kernel_traits::kHeadDim == 64) {
                if constexpr (Kernel_traits::kNWarps == 4) {
                    compute_dq_dk_dv_1colblock_hdim64_64x64_4waves<Kernel_traits, Is_dropout, Is_causal, Is_local,
                                                                   Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K,
                                                                   false, false, true>(params, bidb, bidh, n_block);
                } else if constexpr (Kernel_traits::kNWarps == 8) {
                    compute_dq_dk_dv_1colblock_hdim64_64x64_8waves<Kernel_traits, Is_dropout, Is_causal, Is_local,
                                                                   Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K,
                                                                   false, false, true>(params, bidb, bidh, n_block);
                }
            } else if constexpr (Kernel_traits::kHeadDim == 128) {
                if constexpr (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 64) {
                    compute_dq_dk_dv_1colblock_hdim128_32x64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                             Has_attn_mask, Is_even_MN, Is_even_K, false, false, true>(
                        params, bidb, bidh, n_block);
                } else if constexpr (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 128) {
                    compute_dq_dk_dv_1colblock_hdim128_32x128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                              Has_attn_mask, Is_even_MN, Is_even_K, false, false, true>(
                        params, bidb, bidh, n_block);
                }
            } else {
                compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                           Is_even_MN, Is_even_K, false, false, true>(params, bidb, bidh, n_block);
            }
        }
    } else {
        int n_block = blockIdx.x;
        if constexpr (Kernel_traits::kHeadDim == 64) {
            if constexpr (Kernel_traits::kNWarps == 4) {
                compute_dq_dk_dv_1colblock_hdim64_64x64_4waves<Kernel_traits, Is_dropout, Is_causal, Is_local,
                                                               Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, false,
                                                               false, true>(params, bidb, bidh, n_block);
            } else if constexpr (Kernel_traits::kNWarps == 8) {
                compute_dq_dk_dv_1colblock_hdim64_64x64_8waves<Kernel_traits, Is_dropout, Is_causal, Is_local,
                                                               Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, false,
                                                               false, true>(params, bidb, bidh, n_block);
            }
        } else if constexpr (Kernel_traits::kHeadDim == 128) {
            if constexpr (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 64) {
                compute_dq_dk_dv_1colblock_hdim128_32x64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                         Has_attn_mask, Is_even_MN, Is_even_K, false, false, true>(
                    params, bidb, bidh, n_block);
            } else if constexpr (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 128) {
                compute_dq_dk_dv_1colblock_hdim128_32x128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                          Has_attn_mask, Is_even_MN, Is_even_K, false, false, true>(
                    params, bidb, bidh, n_block);
            }
        } else {
            compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                       Is_even_MN, Is_even_K, false, false, true>(params, bidb, bidh, n_block);
        }
    }
}

}  // namespace flash
