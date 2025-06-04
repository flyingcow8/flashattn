#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/cutlass.h>
#include <mctlass/array.h>
#include <mctlass/numeric_types.h>
#include <mctlass/numeric_conversion.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "philox.cuh"
#include "alibi.h"
#include "attn_mask.h"

namespace flash {

using namespace cute;

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Is_first, bool Is_last, bool Seq_parallel = false, typename Params>
__forceinline__ __device__ void compute_dq_dk_dv_1colblock_hdim128_32x64(const Params &params, const int bidb,
                                                                         const int bidh, const int n_block) {
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
    constexpr int AtomLayoutNS = Kernel_traits::kNWarps / AtomLayoutMS;
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

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::StsLayoutQdO{});

    Tensor sQ_r = make_tensor(sQ.data(), typename Kernel_traits::LdsLayoutQdO{});

    Tensor sQt = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutQdOtNoSwizzle{});

    Tensor sdO = make_tensor(sQt.data() + size(sQt), typename Kernel_traits::StsLayoutQdO{});
    Tensor sdO_r = make_tensor(sdO.data(), typename Kernel_traits::LdsLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutQdOtNoSwizzle{});

    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKVSwizzle{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKVSwizzle{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtSwizzle{});
    Tensor sKtNoSwizzle = make_tensor(sKt.data(), typename Kernel_traits::SmemLayoutKtNoSwizzle{});
    Tensor sdS = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});

    Tensor sdSt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt_r = make_tensor(sdSt.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdSt.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyKV_D128 gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    typename Kernel_traits::GmemTiledCopyBsm1x8 gmem_tiled_copy_QdO;
    auto gmem_thr_copy_QdO = gmem_tiled_copy_QdO.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    using GmemLayoutAtomdQaccum = std::conditional_t<!Seq_parallel, typename Kernel_traits::GmemTiledCopydQaccum,
                                                     typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd>;
    GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QdO.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QdO.partition_D(sQ);
    Tensor tQsQt = gmem_thr_copy_QdO.partition_D(sQt);
    Tensor tdOgdO = gmem_thr_copy_QdO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_QdO.partition_D(sdO);
    Tensor tdOsdOt = gmem_thr_copy_QdO.partition_D(sdOt);
    Tensor tdOgO = gmem_thr_copy_QdO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_KV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);
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
    Tensor tdKrQt = make_tensor<Element>(make_shape(_4{}, _4{}, _2{}));
    Tensor tdVrdO = make_tensor<Element>(make_shape(_4{}, _4{}, _2{}));
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyB64{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ_r);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO_r);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyB64{}, tiled_mma_sdp);

    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);

    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    auto smem_tiled_copy_PdS = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);

    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = make_thr_tensor_stmatrix_trans(sP);

    Element *smem_ds_ptr_w = reinterpret_cast<Element *>(sdS.data().get()) + threadIdx.x * 8;
    Tensor tdSsdS = make_tensor(make_smem_ptr(smem_ds_ptr_w), make_layout(Shape<_4, _1, _2>{}, Stride<_1, _0, _4>{}));
    Tensor tdStsdSt = make_thr_tensor_stmatrix_trans(sdSt);

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = make_thr_tensor_ldmatrix_trans(sPt);
    Tensor tdKsdSt = make_thr_tensor_ldmatrix_trans(sdSt);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Element *smem_qt_ptr = reinterpret_cast<Element *>(sQt.data().get()) +
                           threadIdx.x / 128 * Kernel_traits::kBlockKSmem * 32 +
                           __lane_id() / 16 * Kernel_traits::kBlockKSmem * 4 + __lane_id() % 16 * 4;
    Tensor tdKsQt =
        make_tensor(make_smem_ptr(smem_qt_ptr), make_layout(Shape<_4, _4, _2>{}, Stride<_1, _64, Int<64 * 16>>{}));

    Element *smem_dot_ptr = reinterpret_cast<Element *>(sdOt.data().get()) +
                            threadIdx.x / 128 * Kernel_traits::kBlockKSmem * 32 +
                            __lane_id() / 16 * Kernel_traits::kBlockKSmem * 4 + __lane_id() % 16 * 4;
    Tensor tdVsdOt =
        make_tensor(make_smem_ptr(smem_dot_ptr), make_layout(Shape<_4, _4, _2>{}, Stride<_1, _64, Int<64 * 16>>{}));

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Element *smem_ds_ptr_r =
        reinterpret_cast<Element *>(sdS.data().get()) + threadIdx.x / 64 % 2 * 512 + __lane_id() * 8;
    Tensor tdQsdS = make_tensor(make_smem_ptr(smem_ds_ptr_r),
                                make_layout(Shape<_4, _1, Shape<_2, _2>>{}, Stride<_1, _0, Stride<_4, _1024>>{}));

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Element *sKt_ptr_w = reinterpret_cast<Element *>(sKt.data().get()) + threadIdx.x * 16;
    Tensor tKtsKt = make_tensor(make_smem_ptr(sKt_ptr_w), make_layout(Shape<_4, _4, _2>{}, Stride<_1, _4, _4096>{}));
    Element *skt_ptr_r = reinterpret_cast<Element *>(sKt.data().get()) + threadIdx.x / 128 * 64 +
                         __lane_id() / 16 * 256 + __lane_id() % 16 * 4;
    Tensor tdQsKt = make_tensor(make_smem_ptr(skt_ptr_r),
                                make_layout(Shape<_4, Shape<_2, _2>, _4>{}, Stride<_1, Stride<_128, _4096>, _1024>{}));

    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QdO.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_KV.partition_D(cKV);

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

    if ((!Is_first && !Seq_parallel) || params.deterministic) {
        __syncthreads();
    }

    Tensor tKrK = make_fragment_like(tKsK);
    Tensor tVrV = make_fragment_like(tVsV);
    clear(tKrK);
    clear(tVrV);
    flash::copy2_4x4<Is_even_MN, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKrK, tVgV, tVrV, tKVcKV, tKVpKV,
                                            binfo.actual_seqlen_k - n_block * kBlockN);
    cute::copy(smem_tiled_copy_KV, tKrK, tKsK);
    cute::copy(smem_tiled_copy_KV, tVrV, tVsV);

    Tensor tQrQ = make_fragment_like(tQgQ);
    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);

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

    Tensor tKtrKt = make_tensor(tKrK.data(), make_layout(Shape<_4, _4, _2>{}));
    permute_4x4_b16(tKtrKt);
    __syncthreads();
    Tensor tSrK_copy_view = smem_thr_copy_KV.retile_D(tSrK);
    cute::copy(smem_tiled_copy_KV, tSsK, tSrK_copy_view);
    Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
    cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);

    __syncthreads();
    cute::copy(tKtrKt, tKtsKt);

    __syncthreads();
    cute::copy(tdQsKt, tdQrKt);

    __syncthreads();

    flash::clear_b128(tdOsdOt);
    flash::clear_b128(tQsQt);
    flash::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QdO, tdOgdO, tdOsdOt, tQcQ, tQpQ,
                                              binfo.actual_seqlen_q - m_block * kBlockM);
    flash::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QdO, tQgQ, tQsQt, tQcQ, tQpQ,
                                              binfo.actual_seqlen_q - m_block * kBlockM);

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
    flash::cp_async_wait<0>();

    for (; m_block >= m_block_min; --m_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<16>();

        flash::barrier();
        SWIZZLE_STORE_QDO(tQsQt, tQrQ, tQsQ)
        flash::sync_threads();

        Tensor dP_sum = make_fragment_like(lse);
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi)));
        }

        flash::gemm<false, true>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp, smem_tiled_copy_QdO, smem_tiled_copy_KV,
                                 smem_thr_copy_QdO, smem_thr_copy_KV);

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        if (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16, binfo.actual_seqlen_k,
                                   m_block * kBlockM + get<0>(taccScS_row(0)), binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                   AtomLayoutNS * 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        if (Has_alibi) {
            alibi.apply_alibi(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16,
                              m_block * kBlockM + get<0>(taccScS_row(0)), AtomLayoutMS * 16, AtomLayoutNS * 16);
        }

        if (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                flash::apply_mask(scores, binfo.actual_seqlen_k, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16,
                                  AtomLayoutNS * 16);
            }
        } else if (Is_causal) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k ||
                (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                flash::apply_mask_causal(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16,
                                         binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                         binfo.actual_seqlen_q,

                                         AtomLayoutMS * 16, AtomLayoutNS * 16);
            }
        } else if (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k -
                                        params.window_size_right ||
                (m_block + 1) * kBlockM >=
                    n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left ||
                (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                flash::apply_mask_local(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16, params.window_size_left,
                                        params.window_size_right, AtomLayoutNS * 16);
            }
        }

        flash::scale_apply_exp2<false>(scores, lse, params.scale_softmax_log2);
        if (Is_dropout) {
            int warp_id = tidx / 64;
            int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            int block_col_idx = n_block * (kBlockN / 64);
            dropout.template mc_apply_dropout<true, AtomLayoutMS, AtomLayoutNS>(acc_s, block_row_idx, block_col_idx);
        }

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        if constexpr (Is_dropout) {
            flash::relu_(rP);
        }

        Tensor tPrP = make_tensor(rP.data(), acc_s.layout());
        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);

        SWIZZLE_STORE_QDO(tdOsdOt, tdOrdO, tdOsdO)
        flash::sync_threads();
        flash::shuffle_4x4(tPaP);
        cute::copy(tPaP, tPsP);

        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_dp);

        flash::gemm<false, true>(acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp, smem_tiled_copy_QdO,
                                 smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);

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
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());

        CONVERT_TENSOR_TYPE(ElementAccum, Element, dS_reshaped, tdSrdS)
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        cute::copy(tdVsdOt, tdVrdO);
        flash::shuffle_4x4(tdSadS);

        flash::sync_threads();
        cute::copy(tdSadS, tdStsdSt);
        if (m_block > m_block_min) {
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QdO, tdOgdO, tdOsdOt, tQcQ, tQpQ);
        }

        cute::copy(tdKsQt, tdKrQt);
        flash::permute_4x4_b16(tdVrdO);
        cute::copy(tdVsPt, tdVrPt);
        flash::gemm(acc_dv, tdVrPt, tdVrdO, tiled_mma_dkv);

        flash::sync_threads();
        if (m_block > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
#pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                lse(mi) = gLSE(get<0>(taccScS_row(mi)));
            }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));

            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QdO, tQgQ, tQsQt, tQcQ, tQpQ);
        }

        cute::copy(tdQsdS, tdQrdS);
        flash::swap(*reinterpret_cast<uint64_t *>(tdQrdS(_, _, 1).data()),
                    *reinterpret_cast<uint64_t *>(tdQrdS(_, _, 2).data()));
        flash::gemm(acc_dq, tdQrdS, tdQrKt, tiled_mma_dq);

#pragma unroll
        for (int i = 0; i < size(acc_dq); ++i) {
            atomicAdd(&tdQgdQaccum(i), acc_dq(i));
        }

        cute::copy(tdKsdSt, tdKrdSt);
        flash::permute_4x4_b16(tdKrQt);
        flash::gemm(acc_dk, tdKrdSt, tdKrQt, tiled_mma_dkv);
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

    Tensor acc_dk_view =
        make_tensor(acc_dk.data(), make_layout(Shape<_2, Shape<_4, _4>>{}, Stride<_4, Shape<_8, _1>>{}));

    Tensor acc_dk_copy = make_tensor<ElementAccum>(make_shape(_16{}, _2{}));
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc_dk_copy(j, i) = acc_dk_view(i, j);
        }
    }

    Tensor acc_dv_view =
        make_tensor(acc_dv.data(), make_layout(Shape<_2, Shape<_4, _4>>{}, Stride<_4, Shape<_8, _1>>{}));

    Tensor acc_dv_copy = make_tensor<ElementAccum>(make_shape(_16{}, _2{}));
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc_dv_copy(j, i) = acc_dv_view(i, j);
        }
    }

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_dk_copy, rdK)
    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_dv_copy, rdV)

    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{});

    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    auto sdK_ptr =
        sdK.data() + tidx / 128 * 64 * 64 + tidx / 64 % 2 * 16 * 64 + __lane_id() % 16 * 64 + __lane_id() / 16 * 16;
    Tensor taccdKsdK = make_tensor(sdK_ptr, make_layout(Shape<_16, _2>{}, Stride<_1, _2048>{}));
    auto sdV_ptr =
        sdV.data() + tidx / 128 * 64 * 64 + tidx / 64 % 2 * 16 * 64 + __lane_id() % 16 * 64 + __lane_id() / 16 * 16;
    Tensor taccdVsdV = make_tensor(sdV_ptr, make_layout(Shape<_16, _2>{}, Stride<_1, _2048>{}));

    if (!Is_last) {
        __syncthreads();
    }

    cute::copy(rdK, taccdKsdK);
    cute::copy(rdV, taccdVsdV);

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

}  // namespace flash