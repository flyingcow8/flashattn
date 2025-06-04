/******************************************************************************
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
#include "rotary.h"
#include "attn_mask.h"

namespace flash {

using namespace cute;

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Return_lse, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh,
                                                       const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0
                                      : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q -
                                                     params.window_size_left) /
                                                        kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }

    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));

        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
#pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
            }
        }

        flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                         binfo.actual_seqlen_q - m_block * kBlockM);
        if (Return_lse) {
            Tensor gLSE =
                make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
#pragma unroll
            for (int m = 0; m < size<1>(tOgO); ++m) {
                const int row = get<0>(tOcO(0, m, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                    gLSE(row) = INFINITY;
                }
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;

    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) * params.seqlen_k_rounded +
        (n_block_max - 1) * kBlockN;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor tSgS = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

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

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    if (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        } else {
            flash::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV,
                                                     binfo.actual_seqlen_k - n_block * kBlockN);
        }
        cute::cp_async_fence();

        flash::gemm<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                 smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        if (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16);

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o<true, Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o<false, Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        flash::gemm<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                 smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        if (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf),
                                        kNWarps * 16);

        softmax.template softmax_rescale_o<false, Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    Tensor lse =
        softmax.template normalize_softmax_lse<Is_dropout, Return_lse>(acc_o, params.scale_softmax, params.rp_dropout);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    if (Return_lse) {
        Tensor gLSE =
            make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                        Shape<Int<kBlockM>>{}, Stride<_1>{});

        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
        if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                    gLSE(row) = lse(mi);
                }
            }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
    }

    flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                     binfo.actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Return_lse, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_opt(const Params &params, const int bidb, const int bidh,
                                                           const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    Tensor tQKVrQKV_t = make_tensor<uint32_t>(make_shape(Int<Kernel_traits::kRegSize>{}));
    auto tQrQ = tQKVrQKV_t.data();
    auto tVrV = tQKVrQKV_t.data();
    auto tKrK = tQKVrQKV_t.data() + Kernel_traits::kRegSize / 2;

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0
                                      : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q -
                                                     params.window_size_left) /
                                                        kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }

    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));

        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
#pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
            }
        }

        flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                         binfo.actual_seqlen_q - m_block * kBlockM);
        if (Return_lse) {
            Tensor gLSE =
                make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
#pragma unroll
            for (int m = 0; m < size<1>(tOgO); ++m) {
                const int row = get<0>(tOcO(0, m, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                    gLSE(row) = INFINITY;
                }
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;

    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 (n_block_max - 1) * kBlockN * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) * params.seqlen_k_rounded +
        (n_block_max - 1) * kBlockN;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor tSgS = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

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

    if (Kernel_traits::Share_Q_K_smem) {
        flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, tQpQ,
                                                         binfo.actual_seqlen_q - m_block * kBlockM);
    } else {
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
    }

    if (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tQrQ, tQsQ);
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV,
                                                     binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tKrK, tKsK);
        __syncthreads();

        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

            flash::copy_global_to_reg<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV);
        } else {
            flash::copy_global_to_reg<Is_even_MN, Is_even_K, true>(tVgV, tVrV, tKVcKV, tKVpKV,
                                                                   binfo.actual_seqlen_k - n_block * kBlockN);
        }
        cute::cp_async_fence();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16);

        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));

            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o<true, Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o<false, Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tKrK, tKsK);
        __syncthreads();

        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

        flash::copy_global_to_reg<true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));

            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);

            cute::cp_async_fence();
        }

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf),
                                        kNWarps * 16);

        softmax.template softmax_rescale_o<false, Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    Tensor lse =
        softmax.template normalize_softmax_lse<Is_dropout, Return_lse>(acc_o, params.scale_softmax, params.rp_dropout);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    if (Return_lse) {
        Tensor gLSE =
            make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                        Shape<Int<kBlockM>>{}, Stride<_1>{});

        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
        if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                    gLSE(row) = lse(mi);
                }
            }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
    }

    flash::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                     binfo.actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Return_lse, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_opt_hdim64(const Params &params, const int bidb, const int bidh,
                                                                  const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    uint32_t tQrQ[int(Kernel_traits::kRegSize)];
    uint32_t tVrV[int(Kernel_traits::kRegSize / 2)];
    uint32_t tKrK[int(Kernel_traits::kRegSize / 2)];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    static_assert(Kernel_traits::Is_Q_in_regs == true);
    static_assert(kNWarps == 4);

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    const int kBlockM_stride = m_block * kBlockM;
    if (kBlockM_stride >= binfo.actual_seqlen_q) return;

    const int n_block_min =
        !Is_local
            ? 0
            : std::max(0, (kBlockM_stride + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) /
                              kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div(kBlockM_stride + kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }

    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));

        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        flash::copy_zero_to_global<Is_even_MN, Is_even_K>(tOgO, tOcO, params.d, binfo.actual_seqlen_q - kBlockM_stride);
        if (Return_lse) {
            auto gLSE = reinterpret_cast<int32_t *>(params.softmax_lse_ptr) + row_offset_lse;
            auto inf = INFINITY;
#pragma unroll
            for (int m = 0; m < size<1>(tOgO); ++m) {
                const int row = get<0>(tOcO(0, m, 0));
                __builtin_mxc_stg_b32_predicator(
                    gLSE + row, 0, *reinterpret_cast<int32_t *>(&inf), true, false, false,
                    get<1>(tOcO(0, m, 0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 kBlockM_stride * params.q_row_stride + bidh * params.q_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

    const auto d = params.d;
    int n_block = n_block_max - 1;
    const int base_row_offset = n_block * kBlockN;

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, d, binfo.actual_seqlen_q - kBlockM_stride);

    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 base_row_offset * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 base_row_offset * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + kBlockM_stride) * params.seqlen_k_rounded +
        base_row_offset;

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor tSgS = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    flash::copy_reg_to_share(tQrQ, tQsQ);

    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, d, binfo.actual_seqlen_k - n_block * kBlockN);

    const uint32_t laneId = __lane_id();
    const uint32_t row = (laneId >> 4) << 2;
    const uint32_t col = laneId & 0xf;
    uint32_t cpy_offset[5];
#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        cpy_offset[i] = (row + i) * 32 + ((col ^ (((row + i) & 0x7) << 1)) << 1);
    }
    cpy_offset[4] = (((laneId & 0x7) << 1) + ((laneId >> 5) << 10));

    constexpr int ldg_Num = (kBlockN * kHeadDim / Kernel_traits::kNThreads) / 4;

    int tVgV_offset[ldg_Num];
    uint32_t *tVsV_ptr[ldg_Num];
    int tVcV[ldg_Num + (!Is_even_K)];
#pragma unroll
    for (int i = 0; i < ldg_Num; ++i) {
        /***********************************************************************
         * gv_row means laneId[0-31] read 0~7 rows,laneId[32~63] read 32~39 rows
         * gv_col means reading 64 elements with 16 threads for kBlockN=64
         * For kBlockN=64, ldg_num=4:
         * gv_rows means laneId[0-31] read 0~7 rows,
         * laneId[32~63] read 32~39 rows
         * For kBlockN=32, ldg_num=2:
         * gv_rows means laneId[0-15] read 0~1 rows, laneId[16-31] read 2~3 rows
         * laneId[32~47] read 4~5 rows, laneId[48~63] read 6~7 rows
         * For kBlockN=128, ldg_num=8:
         * gv_rows means gv_rows means laneId[0-31] read 0~7 rows, 64~71 rows
         * laneId[32~63] read 32~39 rows, 96~103 rows
         * For all kBlockN:
         * gv_col means [0~63] read 0~15th 4cols
         ***********************************************************************/
        const int gv_row = kBlockN == 32
                               ? ((laneId >> 4) << 1) + i
                               : (((laneId >> 4) & 0x1) << 2) + ((laneId >> 5) << 5) + (i & 0x3) + (i / 4 * 64);
        const int gv_col = (laneId & 0xf) << 2;

        const int old_gv_row = laneId >> 3;
        const int old_gv_col = (laneId & 0x7) << 3;
        tVgV_offset[i] = (gv_row - old_gv_row) * params.v_row_stride + (gv_col - old_gv_col);

        const int &sv_row = gv_row;
        const int sv_col = ((laneId & 0xf) ^ ((sv_row & 0x7) << 1)) << 2;
        const int &old_sv_row = old_gv_row;
        const int old_sv_col = ((laneId & 0x7) ^ old_sv_row) << 3;

        tVsV_ptr[i] = reinterpret_cast<uint32_t *>(tVsV(_, _0{}, _0{}).data().ptr_ + (sv_row - old_sv_row) * 64 +
                                                   (sv_col - old_sv_col));
        tVcV[i] = gv_row + (tidx / 64) * 8;
    }
    if (!Is_even_K) tVcV[ldg_Num] = (laneId & 0xf) << 2;

    {
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        flash::sync_threads();
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::copy_global_to_reg_V<Is_even_MN, Is_even_K, false, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d,
                                                                       binfo.actual_seqlen_k - n_block * kBlockN);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);
    flash::sync_threads();

    const int gK_offset = -int(kBlockN * params.k_row_stride);
    const int gV_offset = -int(kBlockN * params.v_row_stride);
    const auto mask_offset = kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf);
    const auto kWarps_offset = kNWarps * 16;
    const uint32_t perm_mask[2] = {0x05040100, 0x07060302};

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    flash::copy_reg_to_share(tKrK, tKsK);
    flash::sync_threads();

    flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                 smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

    if (Has_attn_mask) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) + bidb * params.attn_mask_batch_stride +
                            bidh * params.attn_mask_hdim_stride;
        flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                               m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q, kNWarps * 16,
                               16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                               params.attn_mask_col_stride);
    }

    mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);
    flash::copy_reg_to_share_V<false, ldg_Num>(tVrV, tVsV_ptr, perm_mask);

    if (n_block > n_block_min) {
        tKgK.data() = tKgK.data() + gK_offset;
        flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
    }
    flash::sync_threads();

    softmax.template softmax_rescale_o<true, Is_causal || Is_local, false, true>(acc_s, acc_o,
                                                                                 params.scale_softmax_log2);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

    const int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
    const int block_col_idx = n_block * (kBlockN / 64);
    if (Return_softmax) {
        Tensor rP_drop = make_fragment_like(rP);
        cute::copy(rP, rP_drop);
        dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        cute::copy(rP_drop, tSgS);
        tSgS.data() = tSgS.data() + (-kBlockN);
    }
    if (Is_dropout) {
        dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
    }

    flash::gemm_rs_hdim64<ldg_Num>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset);
    n_block--;

#pragma unroll
    for (int masking_step = 1; n_block >= n_block_min && masking_step < n_masking_steps; ++masking_step, --n_block) {
        clear(acc_s);
        if constexpr (kBlockM == 128) {
            tVgV.data() = tVgV.data() + gV_offset;
            flash::copy_global_to_reg_V<true, Is_even_K, false, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        }
        flash::copy_reg_to_share(tKrK, tKsK);
        if constexpr (kBlockM != 128) {
            tVgV.data() = tVgV.data() + gV_offset;
            flash::copy_global_to_reg_V<true, Is_even_K, false, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        }
        flash::sync_threads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<false, ldg_Num>(tVrV, tVsV_ptr, perm_mask);

        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + gK_offset;
            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
        }
        flash::sync_threads();

        softmax.template softmax_rescale_o<false, Is_causal || Is_local, false, true>(acc_s, acc_o,
                                                                                      params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        const int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs_hdim64<ldg_Num>(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, cpy_offset);

        if (n_block <= n_block_min) {
            n_block--;
            break;
        }
    }

#pragma unroll
    for (; n_block > n_block_min; --n_block) {
        clear(acc_s);
        if constexpr (kBlockM == 128) {
            tVgV.data() = tVgV.data() + gV_offset;
            flash::copy_global_to_reg_V<true, Is_even_K, false, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        }
        flash::copy_reg_to_share(tKrK, tKsK);
        if constexpr (kBlockM != 128) {
            tVgV.data() = tVgV.data() + gV_offset;
            flash::copy_global_to_reg_V<true, Is_even_K, false, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        }
        flash::sync_threads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<false, ldg_Num>(tVrV, tVsV_ptr, perm_mask);

        tKgK.data() = tKgK.data() + gK_offset;
        flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, d);

        softmax.template softmax_rescale_o<false, Is_local, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        const int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }
        flash::gemm_rs_hdim64<ldg_Num>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset);
    }

    if (n_block == n_block_min) {
        tVgV.data() = tVgV.data() + gV_offset;
        flash::copy_global_to_reg_V<true, Is_even_K, false, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        flash::sync_threads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<false, ldg_Num>(tVrV, tVsV_ptr, perm_mask);
        flash::sync_threads();

        softmax.template softmax_rescale_o<false, Is_local, false, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        flash::gemm_rs_hdim64<ldg_Num>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset);
    }

    Tensor lse =
        softmax.template normalize_softmax_lse<Is_dropout, Return_lse>(acc_o, params.scale_softmax, params.rp_dropout);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    flash::barrier();

    if constexpr (kBlockM == 64) {
        const int sm_col = ((((laneId & 0x7) ^ (laneId >> 5)) << 1) + ((laneId >> 4) & 0x1)) << 2;
        auto s_ptr = taccOsO.data().ptr_ - sm_col;
        auto ptr = reinterpret_cast<uint32_t *>(rO.data().ptr_);

        auto a = __builtin_mxc_byte_perm(ptr[2], ptr[0], perm_mask[0]);
        ptr[2] = __builtin_mxc_byte_perm(ptr[2], ptr[0], perm_mask[1]);
        ptr[0] = a;
        a = __builtin_mxc_byte_perm(ptr[3], ptr[1], perm_mask[0]);
        auto b = __builtin_mxc_byte_perm(ptr[3], ptr[1], perm_mask[1]);
        ptr[1] = __builtin_mxc_byte_perm(ptr[6], ptr[4], perm_mask[0]);
        ptr[3] = __builtin_mxc_byte_perm(ptr[6], ptr[4], perm_mask[1]);
        ptr[4] = a;
        ptr[6] = b;
        a = __builtin_mxc_byte_perm(ptr[7], ptr[5], perm_mask[0]);
        ptr[7] = __builtin_mxc_byte_perm(ptr[7], ptr[5], perm_mask[1]);
        ptr[5] = a;
#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            auto ptr = reinterpret_cast<uint64_t *>(rO.data().ptr_ + 4 * i);
            const int col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (i >> 1))) << 1) + (i & 0x1)) << 2;
            auto sm_ptr = reinterpret_cast<uint64_t *>(s_ptr + col);
            sm_ptr[0] = ptr[0];
        }
    } else if constexpr (kBlockM == 128) {
        const int sm_col = ((((laneId & 0x7) ^ (laneId >> 5)) << 1) + ((laneId >> 4) & 0x1)) << 2;
        auto s_ptr = taccOsO.data().ptr_ - sm_col;

        auto ptr = reinterpret_cast<uint32_t *>(rO.data().ptr_);
        auto a = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[0]);
        ptr[4] = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[1]);
        ptr[0] = a;
        a = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[0]);
        auto b = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[1]);
        ptr[1] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[0]);
        ptr[5] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[1]);
        ptr[8] = a;
        ptr[12] = b;
        a = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[0]);
        ptr[13] = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[1]);
        ptr[9] = a;

        ptr = reinterpret_cast<uint32_t *>(rO.data().ptr_ + 4);
        a = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[0]);
        ptr[4] = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[1]);
        ptr[0] = a;
        a = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[0]);
        b = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[1]);
        ptr[1] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[0]);
        ptr[5] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[1]);
        ptr[8] = a;
        ptr[12] = b;
        a = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[0]);
        ptr[13] = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[1]);
        ptr[9] = a;

#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            const int col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (i >> 1))) << 1) + (i & 0x1)) << 2;
            auto ptr0 = reinterpret_cast<uint64_t *>(rO.data().ptr_ + 8 * i);
            auto ptr1 = reinterpret_cast<uint64_t *>(rO.data().ptr_ + 8 * i + 4);

            auto sm_ptr0 = reinterpret_cast<uint64_t *>(s_ptr + col);
            auto sm_ptr1 = reinterpret_cast<uint64_t *>(s_ptr + col + 64 * 64);
            sm_ptr0[0] = ptr0[0];
            sm_ptr1[0] = ptr1[0];
        }
    }

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    flash::sync_threads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    if (Return_lse) {
        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor taccOcO = thr_mma.partition_C(caccO);
        static_assert(decltype(size<0>(taccOcO))::value == 4);

        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
        auto gLSE_ptr = reinterpret_cast<int32_t *>(params.softmax_lse_ptr) + row_offset_lse;
        auto lse_ptr = reinterpret_cast<int32_t *>(lse.data());
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            __builtin_mxc_stg_b32_predicator(
                gLSE_ptr + row, 0, lse_ptr[mi], true, false, false,
                get<1>(taccOcO_row(0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    flash::copy_reg_to_global<Is_even_MN, Is_even_K>(tOrO, tOgO, tOcO, d, binfo.actual_seqlen_q - kBlockM_stride);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Return_lse, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_opt_hdim128_blockM64(const Params &params, const int bidb,
                                                                            const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    uint32_t tQrQ[int(Kernel_traits::kRegSize)];
    uint32_t tVrV[int(Kernel_traits::kRegSize / 2)];
    uint32_t tKrK[int(Kernel_traits::kRegSize / 2)];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    static_assert(Kernel_traits::Is_Q_in_regs == true);
    static_assert(Kernel_traits::Share_Q_K_smem == true);
    static_assert(kBlockN == 32 || kBlockN == 64 || kBlockN == 128);
    static_assert(kNWarps == 4);

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    const int kBlockM_stride = m_block * kBlockM;
    if (kBlockM_stride >= binfo.actual_seqlen_q) return;

    const int n_block_min =
        !Is_local
            ? 0
            : std::max(0, (kBlockM_stride + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) /
                              kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div(kBlockM_stride + kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }

    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));

        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        flash::copy_zero_to_global<Is_even_MN, Is_even_K>(tOgO, tOcO, params.d, binfo.actual_seqlen_q - kBlockM_stride);
        if (Return_lse) {
            auto gLSE = reinterpret_cast<int32_t *>(params.softmax_lse_ptr) + row_offset_lse;
            auto inf = INFINITY;
#pragma unroll
            for (int m = 0; m < size<1>(tOgO); ++m) {
                const int row = get<0>(tOcO(0, m, 0));
                __builtin_mxc_stg_b32_predicator(
                    gLSE + row, 0, *reinterpret_cast<int32_t *>(&inf), true, false, false,
                    get<1>(tOcO(0, m, 0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 kBlockM_stride * params.q_row_stride + bidh * params.q_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

    const auto d = params.d;
    int n_block = n_block_max - 1;
    const int base_row_offset = n_block * kBlockN;

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, d, binfo.actual_seqlen_q - kBlockM_stride);

    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 base_row_offset * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 base_row_offset * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + kBlockM_stride) * params.seqlen_k_rounded +
        base_row_offset;

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor tSgS = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    flash::copy_reg_to_share(tQrQ, tQsQ);

    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, d, binfo.actual_seqlen_k - n_block * kBlockN);

    const uint32_t laneId = __lane_id();
    const uint32_t row = (laneId >> 4) << 2;
    const uint32_t col = laneId & 0xf;
    uint32_t cpy_offset[5];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        cpy_offset[i] = (row + i) * 32 + ((col ^ (((row + i) & 0x7) << 1)) << 1);
    }

    cpy_offset[4] = (((laneId & 0x7) << 1) + ((laneId >> 5) << 10));
    const uint32_t tOsVt_stride = get<1>(get<1>(tOsVt(_, _, _0{}).layout().layout_fn().stride())) / 2;
    const uint32_t tOrVt_stride = get<1>(get<1>(tOrVt(_, _, _0{}).layout().stride())) / 2;

    constexpr int ldg_Num = (kBlockN * kHeadDim / Kernel_traits::kNThreads) / 8;

    int tVgV_offset[ldg_Num];
    uint32_t *tVsV_ptr[ldg_Num];
    int tVcV[ldg_Num + (!Is_even_K)];
#pragma unroll
    for (int i = 0; i < ldg_Num; ++i) {
        /***********************************************************************
         * gv_row means laneId[0-31] read 0~7 rows,laneId[32~63] read 32~39 rows
         * gv_col means reading 128 elements with 16 threads for kBlockN=64
         * For kBlockN=64, ldg_num=4:
         * gv_rows means laneId[0-31] read 0~7 rows,
         * laneId[32~63] read 32~39 rows
         * For kBlockN=32, ldg_num=2:
         * gv_rows means laneId[0-15] read 0~1 rows, laneId[16-31] read 2~3 rows
         * laneId[32~47] read 4~5 rows, laneId[48~63] read 6~7 rows
         * For kBlockN=128, ldg_num=8:
         * gv_rows means gv_rows means laneId[0-31] read 0~7 rows, 64~71 rows
         * laneId[32~63] read 32~39 rows, 96~103 rows
         * For all kBlockN:
         * gv_col means [0~63] read 0~15th 8cols
         ***********************************************************************/
        const int gv_row = ldg_Num == 2
                               ? ((laneId >> 4) << 1) + i
                               : (((laneId >> 4) & 0x1) << 2) + ((laneId >> 5) << 5) + (i & 0x3) + (i / 4 * 64);
        const int gv_col = (laneId & 0xf) << 3;
        const int old_gv_row = laneId >> 3;
        const int old_gv_col = (laneId & 0x7) << 3;
        tVgV_offset[i] = (gv_row - old_gv_row) * params.v_row_stride + (gv_col - old_gv_col);

        const int sv_row = gv_row + ((old_gv_row & 0x1) * kBlockN);
        const int sv_col = ((laneId & 0x7) ^ (sv_row & 0x7)) << 3;
        const int &old_sv_row = old_gv_row;
        const int old_sv_col = ((laneId & 0x7) ^ old_sv_row) << 3;
        tVsV_ptr[i] = reinterpret_cast<uint32_t *>(tVsV(_, _0{}, _0{}).data().ptr_ + (sv_row - old_sv_row) * 64 +
                                                   (sv_col - old_sv_col));
        tVcV[i] = gv_row + (tidx / 64) * 8;
    }
    if (!Is_even_K) tVcV[ldg_Num] = (laneId & 0xf) << 3;

    {
        const int col = (((laneId & 0x7) ^ (laneId >> 5)) << 3) + (((laneId >> 4) & 0x1) << 2);
        flash::sync_threads();
        auto sm_ptr = reinterpret_cast<uint32_t *>(tSsQ.data().ptr_ - col);
        auto r_ptr = reinterpret_cast<uint32_t *>(tSrQ.data());
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = (((laneId & 0x7) ^ ((laneId >> 5) + 2 * i)) << 2) + (((laneId >> 4) & 0x1) << 1);
            auto ptr0 = reinterpret_cast<uint64_t *>(r_ptr + 2 * i);
            auto s_ptr0 = reinterpret_cast<uint64_t *>(sm_ptr + col);
            auto ptr1 = reinterpret_cast<uint64_t *>(r_ptr + 2 * i + 8);
            auto s_ptr1 = reinterpret_cast<uint64_t *>(sm_ptr + col + 64 * 32);
            ptr0[0] = s_ptr0[0];
            ptr1[0] = s_ptr1[0];
        }
    }

    clear(acc_o);

    flash::copy_global_to_reg_V<Is_even_MN, Is_even_K, true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d,
                                                                      binfo.actual_seqlen_k - n_block * kBlockN);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);
    flash::sync_threads();

    const int gK_offset = -int(kBlockN * params.k_row_stride);
    const int gV_offset = -int(kBlockN * params.v_row_stride);
    const auto mask_offset = kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf);
    const auto kWarps_offset = kNWarps * 16;
    const uint32_t perm_mask[2] = {0x05040100, 0x07060302};

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});

    flash::copy_reg_to_share(tKrK, tKsK);
    clear(acc_s);
    flash::sync_threads();

    flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                 smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

    if (Has_attn_mask) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) + bidb * params.attn_mask_batch_stride +
                            bidh * params.attn_mask_hdim_stride;
        flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                               m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q, kNWarps * 16,
                               16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                               params.attn_mask_col_stride);
    }

    mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

    flash::copy_reg_to_share_V<true, ldg_Num>(tVrV, tVsV_ptr, perm_mask);
    if (n_block > n_block_min) {
        tKgK.data() = tKgK.data() + gK_offset;
        flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
    }

    softmax.template softmax_rescale_o<true, Is_causal || Is_local, true, true>(acc_s, acc_o,
                                                                                params.scale_softmax_log2);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

    int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
    int block_col_idx = n_block * (kBlockN / 64);
    if (Return_softmax) {
        Tensor rP_drop = make_fragment_like(rP);
        cute::copy(rP, rP_drop);
        dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        cute::copy(rP_drop, tSgS);
        tSgS.data() = tSgS.data() + (-kBlockN);
    }
    if (Is_dropout) {
        dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
    }

    flash::gemm_rs_hdim128<ldg_Num>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    n_block--;

#pragma unroll
    for (int masking_step = 1; n_block >= n_block_min && masking_step < n_masking_steps; ++masking_step, --n_block) {
        tVgV.data() = tVgV.data() + gV_offset;
        flash::copy_global_to_reg_V<true, Is_even_K, true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        flash::copy_reg_to_share(tKrK, tKsK);
        clear(acc_s);
        flash::sync_threads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<true, ldg_Num>(tVrV, tVsV_ptr, perm_mask);
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + gK_offset;
            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
        }

        softmax.template softmax_rescale_o<false, Is_causal || Is_local, true, true>(acc_s, acc_o,
                                                                                     params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        flash::gemm_rs_hdim128<ldg_Num>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);

        if (n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

#pragma unroll
    for (; n_block > n_block_min; --n_block) {
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);

        tVgV.data() = tVgV.data() + gV_offset;
        flash::copy_global_to_reg_V<true, Is_even_K, true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        flash::sync_threads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<true, ldg_Num>(tVrV, tVsV_ptr, perm_mask);

        tKgK.data() = tKgK.data() + gK_offset;
        flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, d);

        softmax.template softmax_rescale_o<false, Is_local, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        flash::gemm_rs_hdim128<ldg_Num>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    }

    if (n_block == n_block_min) {
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);

        tVgV.data() = tVgV.data() + gV_offset;
        flash::copy_global_to_reg_V<true, Is_even_K, true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        flash::sync_threads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<true, ldg_Num>(tVrV, tVsV_ptr, perm_mask);
        softmax.template softmax_rescale_o<false, Is_local, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        flash::gemm_rs_hdim128<ldg_Num>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    }

    Tensor lse =
        softmax.template normalize_softmax_lse<Is_dropout, Return_lse>(acc_o, params.scale_softmax, params.rp_dropout);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);

    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    flash::barrier();

    int sm_col = ((((laneId & 0x7) ^ (laneId >> 5)) << 1) + ((laneId >> 4) & 0x1)) << 2;
    auto s_ptr = taccOsO.data().ptr_ - sm_col;
    auto ptr0 = reinterpret_cast<uint32_t *>(rO.data().ptr_);
    auto ptr1 = reinterpret_cast<uint32_t *>(rO.data().ptr_ + 16);
    auto ptr = reinterpret_cast<uint64_t *>(rO.data().ptr_);

    auto temp_a = __builtin_mxc_byte_perm(ptr0[2], ptr0[0], perm_mask[0]);
    ptr0[2] = __builtin_mxc_byte_perm(ptr0[2], ptr0[0], perm_mask[1]);
    ptr0[0] = temp_a;
    temp_a = __builtin_mxc_byte_perm(ptr0[3], ptr0[1], perm_mask[0]);
    auto temp_b = __builtin_mxc_byte_perm(ptr0[3], ptr0[1], perm_mask[1]);
    ptr0[1] = __builtin_mxc_byte_perm(ptr0[6], ptr0[4], perm_mask[0]);
    ptr0[3] = __builtin_mxc_byte_perm(ptr0[6], ptr0[4], perm_mask[1]);

    auto temp_c = __builtin_mxc_byte_perm(ptr1[2], ptr1[0], perm_mask[0]);
    ptr1[2] = __builtin_mxc_byte_perm(ptr1[2], ptr1[0], perm_mask[1]);
    ptr1[0] = temp_c;
    temp_c = __builtin_mxc_byte_perm(ptr1[3], ptr1[1], perm_mask[0]);
    auto temp_d = __builtin_mxc_byte_perm(ptr1[3], ptr1[1], perm_mask[1]);
    ptr1[1] = __builtin_mxc_byte_perm(ptr1[6], ptr1[4], perm_mask[0]);
    ptr1[3] = __builtin_mxc_byte_perm(ptr1[6], ptr1[4], perm_mask[1]);

    sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (0 >> 1))) << 1) + (0 & 0x1)) << 2;
    auto sm_ptr = reinterpret_cast<uint64_t *>(s_ptr + sm_col);
    sm_ptr[0] = ptr[0];
    sm_ptr[1024] = ptr[4];
    sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (1 >> 1))) << 1) + (1 & 0x1)) << 2;
    sm_ptr = reinterpret_cast<uint64_t *>(s_ptr + sm_col);
    sm_ptr[0] = ptr[1];
    sm_ptr[1024] = ptr[5];

    ptr0[4] = temp_a;
    ptr0[6] = temp_b;
    temp_a = __builtin_mxc_byte_perm(ptr0[7], ptr0[5], perm_mask[0]);
    ptr0[7] = __builtin_mxc_byte_perm(ptr0[7], ptr0[5], perm_mask[1]);
    ptr0[5] = temp_a;

    ptr1[4] = temp_c;
    ptr1[6] = temp_d;
    temp_c = __builtin_mxc_byte_perm(ptr1[7], ptr1[5], perm_mask[0]);
    ptr1[7] = __builtin_mxc_byte_perm(ptr1[7], ptr1[5], perm_mask[1]);
    ptr1[5] = temp_c;

    sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (2 >> 1))) << 1) + (2 & 0x1)) << 2;
    sm_ptr = reinterpret_cast<uint64_t *>(s_ptr + sm_col);
    sm_ptr[0] = ptr[2];
    sm_ptr[1024] = ptr[6];
    sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (3 >> 1))) << 1) + (3 & 0x1)) << 2;
    sm_ptr = reinterpret_cast<uint64_t *>(s_ptr + sm_col);
    sm_ptr[0] = ptr[3];
    sm_ptr[1024] = ptr[7];

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    flash::sync_threads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    if (Return_lse) {
        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor taccOcO = thr_mma.partition_C(caccO);
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
        auto gLSE_ptr = reinterpret_cast<int32_t *>(params.softmax_lse_ptr) + row_offset_lse;
        auto lse_ptr = reinterpret_cast<int32_t *>(lse.data());
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            __builtin_mxc_stg_b32_predicator(
                gLSE_ptr + row, 0, lse_ptr[mi], true, false, false,
                get<1>(taccOcO_row(0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    flash::copy_reg_to_global<Is_even_MN, Is_even_K>(tOrO, tOgO, tOcO, d, binfo.actual_seqlen_q - kBlockM_stride);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Return_lse, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_opt_hdim128_blockM128(const Params &params, const int bidb,
                                                                             const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    uint32_t tQrQ[int(Kernel_traits::kRegSize)];
    uint32_t tVrV[int(Kernel_traits::kRegSize / 2)];
    uint32_t tKrK[int(Kernel_traits::kRegSize / 2)];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    static_assert(Kernel_traits::Is_Q_in_regs == true);
    static_assert(Kernel_traits::Share_Q_K_smem == true);
    static_assert(kBlockN == 32 || kBlockN == 64 || kBlockN == 128);
    static_assert(kNWarps == 4);

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    const int kBlockM_stride = m_block * kBlockM;
    if (kBlockM_stride >= binfo.actual_seqlen_q) return;

    const int n_block_min =
        !Is_local
            ? 0
            : std::max(0, (kBlockM_stride + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) /
                              kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div(kBlockM_stride + kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }

    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));

        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        flash::copy_zero_to_global<Is_even_MN, Is_even_K>(tOgO, tOcO, params.d, binfo.actual_seqlen_q - kBlockM_stride);
        if (Return_lse) {
            auto gLSE = reinterpret_cast<int32_t *>(params.softmax_lse_ptr) + row_offset_lse;
            auto inf = INFINITY;
#pragma unroll
            for (int m = 0; m < size<1>(tOgO); ++m) {
                const int row = get<0>(tOcO(0, m, 0));
                __builtin_mxc_stg_b32_predicator(
                    gLSE + row, 0, *reinterpret_cast<int32_t *>(&inf), true, false, false,
                    get<1>(tOcO(0, m, 0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 kBlockM_stride * params.q_row_stride + bidh * params.q_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

    auto d = params.d;
    int n_block = n_block_max - 1;
    const int base_row_offset = n_block * kBlockN;

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, d, binfo.actual_seqlen_q - kBlockM_stride);

    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 base_row_offset * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 base_row_offset * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + kBlockM_stride) * params.seqlen_k_rounded +
        base_row_offset;

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor tSgS = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    flash::copy_reg_to_share(tQrQ, tQsQ);

    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, d, binfo.actual_seqlen_k - n_block * kBlockN);

    const uint32_t laneId = __lane_id();
    const uint32_t row = (laneId >> 4) << 2;
    const uint32_t col = laneId & 0xf;
    uint32_t cpy_offset[5];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        cpy_offset[i] = (row + i) * 32 + ((col ^ (((row + i) & 0x7) << 1)) << 1);
    }

    cpy_offset[4] = (((laneId & 0x7) << 1) + ((laneId >> 5) << 10));
    const uint32_t tOsVt_stride = get<1>(get<1>(tOsVt(_, _, _0{}).layout().layout_fn().stride())) / 2;
    const uint32_t tOrVt_stride = get<1>(get<1>(tOrVt(_, _, _0{}).layout().stride())) / 2;

    constexpr int ldg_Num = (kBlockN * kHeadDim / Kernel_traits::kNThreads) / 8;

    int tVgV_offset[ldg_Num];
    uint32_t *tVsV_ptr[ldg_Num];
    int tVcV[ldg_Num + (!Is_even_K)];
#pragma unroll
    for (int i = 0; i < ldg_Num; ++i) {
        /***********************************************************************
         * gv_row means laneId[0-31] read 0~7 rows,laneId[32~63] read 32~39 rows
         * gv_col means reading 128 elements with 16 threads for kBlockN=64
         * For kBlockN=64, ldg_num=4:
         * gv_rows means laneId[0-31] read 0~7 rows,
         * laneId[32~63] read 32~39 rows
         * For kBlockN=32, ldg_num=2:
         * gv_rows means laneId[0-15] read 0~1 rows, laneId[16-31] read 2~3 rows
         * laneId[32~47] read 4~5 rows, laneId[48~63] read 6~7 rows
         * For kBlockN=128, ldg_num=8:
         * gv_rows means gv_rows means laneId[0-31] read 0~7 rows, 64~71 rows
         * laneId[32~63] read 32~39 rows, 96~103 rows
         * For all kBlockN:
         * gv_col means [0~63] read 0~15th 8cols
         ***********************************************************************/
        const int gv_row = kBlockN == 32
                               ? ((laneId >> 4) << 1) + i
                               : (((laneId >> 4) & 0x1) << 2) + ((laneId >> 5) << 5) + (i & 0x3) + (i / 4 * 64);
        const int gv_col = (laneId & 0xf) << 3;
        const int old_gv_row = laneId >> 3;
        const int old_gv_col = (laneId & 0x7) << 3;
        tVgV_offset[i] = (gv_row - old_gv_row) * params.v_row_stride + (gv_col - old_gv_col);

        const int sv_row = gv_row + ((old_gv_row & 0x1) * kBlockN);
        const int sv_col = ((laneId & 0x7) ^ (sv_row & 0x7)) << 3;
        const int &old_sv_row = old_gv_row;
        const int old_sv_col = ((laneId & 0x7) ^ old_sv_row) << 3;
        tVsV_ptr[i] = reinterpret_cast<uint32_t *>(tVsV(_, _0{}, _0{}).data().ptr_ + (sv_row - old_sv_row) * 64 +
                                                   (sv_col - old_sv_col));
        tVcV[i] = gv_row + (tidx / 64) * 8;
    }
    if (!Is_even_K) tVcV[ldg_Num] = (laneId & 0xf) << 3;

    {
        const int col = (((laneId & 0x7) ^ (laneId >> 5)) << 3) + (((laneId >> 4) & 0x1) << 2);
        flash::sync_threads();
        auto sm_ptr = reinterpret_cast<uint32_t *>(tSsQ.data().ptr_ - col);
        auto r_ptr = reinterpret_cast<uint32_t *>(tSrQ.data());
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = (((laneId & 0x7) ^ ((laneId >> 5) + 2 * i)) << 2) + (((laneId >> 4) & 0x1) << 1);
            auto ptr0 = reinterpret_cast<uint64_t *>(r_ptr + 4 * i);
            auto s_ptr0 = reinterpret_cast<uint64_t *>(sm_ptr + col);
            ptr0[0] = s_ptr0[0];

            ptr0 = reinterpret_cast<uint64_t *>(r_ptr + 4 * i + 2);
            s_ptr0 = reinterpret_cast<uint64_t *>(sm_ptr + col + 64 * 32);
            ptr0[0] = s_ptr0[0];

            auto ptr1 = reinterpret_cast<uint64_t *>(r_ptr + 4 * i + 16);
            auto s_ptr1 = reinterpret_cast<uint64_t *>(sm_ptr + col + 128 * 32);
            ptr1[0] = s_ptr1[0];

            ptr1 = reinterpret_cast<uint64_t *>(r_ptr + 4 * i + 16 + 2);
            s_ptr1 = reinterpret_cast<uint64_t *>(sm_ptr + col + 192 * 32);
            ptr1[0] = s_ptr1[0];
        }
    }

    clear(acc_o);

    flash::copy_global_to_reg_V<Is_even_MN, Is_even_K, true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d,
                                                                      binfo.actual_seqlen_k - n_block * kBlockN);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);
    flash::sync_threads();

    const int gK_offset = -int(kBlockN * params.k_row_stride);
    const int gV_offset = -int(kBlockN * params.v_row_stride);
    const auto mask_offset = kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf);
    const auto kWarps_offset = kNWarps * 16;
    const uint32_t perm_mask[2] = {0x05040100, 0x07060302};

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + gV_offset;
            flash::copy_global_to_reg_V<true, Is_even_K, true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        }
        flash::copy_reg_to_share(tKrK, tKsK);
        clear(acc_s);
        flash::sync_threads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<true, ldg_Num>(tVrV, tVsV_ptr, perm_mask);
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + gK_offset;
            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
        }

        masking_step == 0 ? softmax.template softmax_rescale_o<true, Is_causal || Is_local, true, true>(
                                acc_s, acc_o, params.scale_softmax_log2)
                          : softmax.template softmax_rescale_o<false, Is_causal || Is_local, true, true>(
                                acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        flash::gemm_rs_hdim128<ldg_Num>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

#pragma unroll
    for (; n_block > n_block_min; --n_block) {
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);

        tVgV.data() = tVgV.data() + gV_offset;
        flash::copy_global_to_reg_V<true, Is_even_K, true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        flash::sync_threads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<true, ldg_Num>(tVrV, tVsV_ptr, perm_mask);

        tKgK.data() = tKgK.data() + gK_offset;
        flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, d);

        softmax.template softmax_rescale_o<false, Is_local, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        flash::gemm_rs_hdim128<ldg_Num>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    }

    if (n_block == n_block_min) {
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);

        tVgV.data() = tVgV.data() + gV_offset;
        flash::copy_global_to_reg_V<true, Is_even_K, true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        flash::sync_threads();

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                   m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, binfo.actual_seqlen_q,
                                   kNWarps * 16, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<true, ldg_Num>(tVrV, tVsV_ptr, perm_mask);
        softmax.template softmax_rescale_o<false, Is_local, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        flash::gemm_rs_hdim128<ldg_Num>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    }

    Tensor lse =
        softmax.template normalize_softmax_lse<Is_dropout, Return_lse>(acc_o, params.scale_softmax, params.rp_dropout);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);

    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    flash::barrier();

    const int sm_col = ((((laneId & 0x7) ^ (laneId >> 5)) << 1) + ((laneId >> 4) & 0x1)) << 2;
    auto s_ptr = taccOsO.data().ptr_ - sm_col;

#pragma unroll 2
    for (int i = 0; i < 2; ++i) {
        auto ptr = reinterpret_cast<uint32_t *>(rO.data().ptr_ + 4 * i);
        auto a = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[0]);
        ptr[4] = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[1]);
        ptr[0] = a;
        a = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[0]);
        auto b = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[1]);
        ptr[1] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[0]);
        ptr[5] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[1]);
        ptr[8] = a;
        ptr[12] = b;
        a = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[0]);
        ptr[13] = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[1]);
        ptr[9] = a;

        ptr = reinterpret_cast<uint32_t *>(rO.data().ptr_ + 32 + 4 * i);
        a = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[0]);
        ptr[4] = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[1]);
        ptr[0] = a;
        a = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[0]);
        b = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[1]);
        ptr[1] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[0]);
        ptr[5] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[1]);
        ptr[8] = a;
        ptr[12] = b;
        a = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[0]);
        ptr[13] = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[1]);
        ptr[9] = a;
#pragma unroll 4
        for (int j = 0; j < 4; ++j) {
            const int col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (j >> 1))) << 1) + (j & 0x1)) << 2;
            auto ptr0 = reinterpret_cast<uint64_t *>(rO.data().ptr_ + 8 * j + 4 * i);
            auto ptr1 = reinterpret_cast<uint64_t *>(rO.data().ptr_ + 8 * j + 32 + 4 * i);
            auto sm_ptr0 = reinterpret_cast<uint64_t *>(s_ptr + col + 64 * 64 * i);
            auto sm_ptr1 = reinterpret_cast<uint64_t *>(s_ptr + col + 128 * 64 + 64 * 64 * i);
            sm_ptr0[0] = ptr0[0];
            sm_ptr1[0] = ptr1[0];
        }
    }

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    flash::sync_threads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    if (Return_lse) {
        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor taccOcO = thr_mma.partition_C(caccO);
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
        auto gLSE_ptr = reinterpret_cast<int32_t *>(params.softmax_lse_ptr) + row_offset_lse;
        auto lse_ptr = reinterpret_cast<int32_t *>(lse.data());
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            __builtin_mxc_stg_b32_predicator(
                gLSE_ptr + row, 0, lse_ptr[mi], true, false, false,
                get<1>(taccOcO_row(0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    flash::copy_reg_to_global<Is_even_MN, Is_even_K>(tOrO, tOgO, tOcO, d, binfo.actual_seqlen_q - kBlockM_stride);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Return_lse, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_opt_hdim256(const Params &params, const int bidb, const int bidh,
                                                                   const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    uint32_t tQrQ[int(Kernel_traits::kRegSize)];
    uint32_t tVrV[int(Kernel_traits::kRegSize / 2)];
    uint32_t tKrK[int(Kernel_traits::kRegSize / 2)];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    static_assert(Kernel_traits::Is_Q_in_regs == true);
    static_assert(kNWarps == 4);

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t, bidb, bidh,
                           tidx, params.h);

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    const int kBlockM_stride = m_block * kBlockM;
    if (kBlockM_stride >= binfo.actual_seqlen_q) return;

    const int n_block_min =
        !Is_local
            ? 0
            : std::max(0, (kBlockM_stride + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) /
                              kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div(kBlockM_stride + kBlockM + binfo.actual_seqlen_k -
                                                               binfo.actual_seqlen_q + params.window_size_right,
                                                           kBlockN));
    }

    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                     kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));

        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        flash::copy_zero_to_global<Is_even_MN, Is_even_K>(tOgO, tOcO, params.d, binfo.actual_seqlen_q - kBlockM_stride);
        if (Return_lse) {
            auto gLSE = reinterpret_cast<int32_t *>(params.softmax_lse_ptr) + row_offset_lse;
            auto inf = INFINITY;
#pragma unroll
            for (int m = 0; m < size<1>(tOgO); ++m) {
                const int row = get<0>(tOcO(0, m, 0));
                __builtin_mxc_stg_b32_predicator(
                    gLSE + row, 0, *reinterpret_cast<int32_t *>(&inf), true, false, false,
                    get<1>(tOcO(0, m, 0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
            }
        }
        return;
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                                 kBlockM_stride * params.q_row_stride + bidh * params.q_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

    const auto d = params.d;
    int n_block = n_block_max - 1;
    const int base_row_offset = n_block * kBlockN;

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, d, binfo.actual_seqlen_q - kBlockM_stride);

    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                 base_row_offset * params.k_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                 base_row_offset * params.v_row_stride +
                                 (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + kBlockM_stride) * params.seqlen_k_rounded +
        base_row_offset;

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor tSgS = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    flash::copy_reg_to_share(tQrQ, tQsQ);

    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, d, binfo.actual_seqlen_k - n_block * kBlockN);

    const uint32_t laneId = __lane_id();
    const uint32_t row = (laneId >> 4) << 2;
    const uint32_t col = laneId & 0xf;
    uint32_t cpy_offset[5];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        cpy_offset[i] = (row + i) * 32 + ((col ^ (((row + i) & 0x7) << 1)) << 1);
    }

    cpy_offset[4] = (((laneId & 0x7) << 1) + ((laneId >> 5) << 10));
    const uint32_t tOsVt_stride = get<1>(get<1>(tOsVt(_, _, _0{}).layout().layout_fn().stride())) / 2;
    const uint32_t tOrVt_stride = get<1>(get<1>(tOrVt(_, _, _0{}).layout().stride())) / 2;

    constexpr int ldg_Num = (kBlockN * kHeadDim / Kernel_traits::kNThreads) / 8;

    constexpr int lds_Tuple = (kBlockN * kHeadDim / Kernel_traits::kNThreads) / 16;

    int tVgV_offset[ldg_Num];
    uint32_t *tVsV_ptr[ldg_Num];
    int tVcV[ldg_Num + (!Is_even_K)];
#pragma unroll
    for (int i = 0; i < ldg_Num; ++i) {
        /***********************************************************************
         * gv_row means laneId[0-31] read 0~7 rows,laneId[32~63] read 32~39 rows
         * gv_col means reading 128 elements with 16 threads for kBlockN=64
         * For kBlockN=64, ldg_num=8:
         * gv_rows means laneId[0-31] read 0~7 rows,
         * laneId[32~63] read 32~39 rows
         * gv_col means [0~63] read 0~15th 8cols,
         * and for lastest 4-ldg_num, gv_col need to increase 16 8cols
         ***********************************************************************/
        const int gv_row = (((laneId >> 4) & 0x1) << 2) + ((laneId >> 5) << 5) + (i & 0x3);
        const int gv_col = ((laneId & 0xf) << 3) + (i / 4) * 128;
        const int old_gv_row = laneId >> 3;
        const int old_gv_col = (laneId & 0x7) << 3;
        tVgV_offset[i] = (gv_row - old_gv_row) * params.v_row_stride + (gv_col - old_gv_col);

        const int sv_row = gv_row + ((old_gv_row & 0x1) * kBlockN) + (i / 4) * 128;
        const int sv_col = ((laneId & 0x7) ^ (sv_row & 0x7)) << 3;
        const int &old_sv_row = old_gv_row;
        const int old_sv_col = ((laneId & 0x7) ^ old_sv_row) << 3;
        tVsV_ptr[i] = reinterpret_cast<uint32_t *>(tVsV(_, _0{}, _0{}).data().ptr_ + (sv_row - old_sv_row) * 64 +
                                                   (sv_col - old_sv_col));
        tVcV[i] = gv_row + (tidx / 64) * 8;
    }
    if (!Is_even_K) tVcV[ldg_Num] = (laneId & 0xf) << 3;

    flash::sync_threads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope =
        !Has_alibi || params.alibi_slopes_ptr == nullptr
            ? 0.0f
            : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] /
                  params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q,
                                                     params.window_size_left, params.window_size_right, alibi_slope);
    flash::sync_threads();

    const auto mask_offset = kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf);
    const auto kWarps_offset = kNWarps * 16;
    const uint32_t perm_mask[2] = {0x05040100, 0x07060302};

    constexpr int n_masking_steps =
        (!Is_causal && !Is_local)
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        flash::sync_threads();

        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            flash::copy_global_to_reg_V<true, Is_even_K, true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        } else {
            flash::copy_global_to_reg_V<Is_even_MN, Is_even_K, true, ldg_Num>(
                tVgV, tVrV, tVcV, tVgV_offset, d, binfo.actual_seqlen_k - n_block * kBlockN);
        }

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k, mask_offset, binfo.actual_seqlen_q,
                                   kWarps_offset, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<true, ldg_Num>(tVrV, tVsV_ptr, perm_mask);

        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
        }

        masking_step == 0 ? softmax.template softmax_rescale_o<true, Is_causal || Is_local, true, true>(
                                acc_s, acc_o, params.scale_softmax_log2)
                          : softmax.template softmax_rescale_o<false, Is_causal || Is_local, true, true>(
                                acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        flash::gemm_rs<ldg_Num, lds_Tuple>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        flash::sync_threads();

        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        flash::copy_global_to_reg_V<true, Is_even_K, true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);

        flash::gemm_opt<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                                                     smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy_global_to_reg<true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
        }
        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr) +
                                bidb * params.attn_mask_batch_stride + bidh * params.attn_mask_hdim_stride;
            flash::apply_attn_mask(scores, n_block * kBlockN, binfo.actual_seqlen_k, mask_offset, binfo.actual_seqlen_q,
                                   kWarps_offset, 16, params.scale_softmax, bias_ptr, params.attn_mask_row_stride,
                                   params.attn_mask_col_stride);
        }

        mask.template apply_mask<false>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset);

        flash::copy_reg_to_share_V<true, ldg_Num>(tVrV, tVsV_ptr, perm_mask);

        softmax.template softmax_rescale_o<false, Is_local, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout<true>(rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        flash::gemm_rs<ldg_Num, lds_Tuple>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    }

    Tensor lse =
        softmax.template normalize_softmax_lse<Is_dropout, Return_lse>(acc_o, params.scale_softmax, params.rp_dropout);

    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    flash::barrier();

    {
        int old_sm_col = ((((laneId & 0x7) ^ (laneId >> 5)) << 1) + ((laneId >> 4) & 0x1)) << 2;
        auto s_ptr = taccOsO.data().ptr_ - old_sm_col;
#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            auto ptr = reinterpret_cast<uint32_t *>(rO.data().ptr_ + 16 * i);
            auto temp_a = __builtin_mxc_byte_perm(ptr[2], ptr[0], perm_mask[0]);
            ptr[2] = __builtin_mxc_byte_perm(ptr[2], ptr[0], perm_mask[1]);
            ptr[0] = temp_a;
            temp_a = __builtin_mxc_byte_perm(ptr[3], ptr[1], perm_mask[0]);
            auto temp_b = __builtin_mxc_byte_perm(ptr[3], ptr[1], perm_mask[1]);
            ptr[1] = __builtin_mxc_byte_perm(ptr[6], ptr[4], perm_mask[0]);
            ptr[3] = __builtin_mxc_byte_perm(ptr[6], ptr[4], perm_mask[1]);

            ptr[4] = temp_a;
            ptr[6] = temp_b;
            temp_a = __builtin_mxc_byte_perm(ptr[7], ptr[5], perm_mask[0]);
            ptr[7] = __builtin_mxc_byte_perm(ptr[7], ptr[5], perm_mask[1]);
            ptr[5] = temp_a;
        }

#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            int new_sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (i >> 1))) << 1) + (i & 0x1)) << 2;
#pragma unroll 4
            for (int j = 0; j < 4; ++j) {
                auto ptr = reinterpret_cast<uint64_t *>(rO.data().ptr_ + j * 16);
                auto sm_ptr = reinterpret_cast<uint64_t *>(s_ptr + new_sm_col) + 1024 * j;
                sm_ptr[0] = ptr[i];
            }
        }
    }

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
                                 kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    flash::sync_threads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    if (Return_lse) {
        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor taccOcO = thr_mma.partition_C(caccO);
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
        auto gLSE_ptr = reinterpret_cast<int32_t *>(params.softmax_lse_ptr) + row_offset_lse;
        auto lse_ptr = reinterpret_cast<int32_t *>(lse.data());
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            __builtin_mxc_stg_b32_predicator(
                gLSE_ptr + row, 0, lse_ptr[mi], true, false, false,
                get<1>(taccOcO_row(0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    flash::copy_reg_to_global<Is_even_MN, Is_even_K>(tOrO, tOgO, tOcO, d, binfo.actual_seqlen_q - kBlockM_stride);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
          bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Return_lse, typename Params>
__forceinline__ __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    const int bidb = blockIdx.z;
    const int bidh = blockIdx.y;

    if constexpr (Kernel_traits::kHeadDim == 192 || Kernel_traits::kHeadDim == 224) {
        flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                      Is_even_MN, Is_even_K, Return_softmax, Return_lse, Params>(params, bidb, bidh,
                                                                                                 m_block);
        return;
    }

    if constexpr (Kernel_traits::kHeadDim == 256) {
        flash::compute_attn_1rowblock_opt_hdim256<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                  Has_attn_mask, Is_even_MN, Is_even_K, Return_softmax, Return_lse,
                                                  Params>(params, bidb, bidh, m_block);
        return;
    }

    if constexpr (Kernel_traits::kHeadDim == 128) {
        if constexpr (Kernel_traits::kBlockM == 64) {
            flash::compute_attn_1rowblock_opt_hdim128_blockM64<Kernel_traits, Is_dropout, Is_causal, Is_local,
                                                               Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K,
                                                               Return_softmax, Return_lse, Params>(
                                                                params, bidb, bidh, m_block);
            return;
        }
        flash::compute_attn_1rowblock_opt_hdim128_blockM128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                            Has_attn_mask, Is_even_MN, Is_even_K, Return_softmax,
                                                            Return_lse, Params>(params, bidb, bidh, m_block);
        return;
    }

    if constexpr (Kernel_traits::kHeadDim == 64) {
        flash::compute_attn_1rowblock_opt_hdim64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                 Has_attn_mask, Is_even_MN, Is_even_K, Return_softmax, Return_lse,
                                                 Params>(params, bidb, bidh, m_block);
        return;
    }

    flash::compute_attn_1rowblock_opt<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                      Is_even_MN, Is_even_K, Return_softmax, Return_lse, Params>(params, bidb, bidh,
                                                                                                 m_block);
}

}  // namespace flash
