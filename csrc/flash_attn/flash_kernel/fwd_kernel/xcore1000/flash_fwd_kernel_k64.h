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

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ auto reorder_acc_o(Tensor<Engine0, Layout0> &rO_S, Tensor<Engine1, Layout1> &rO_D) {
    Tensor rO_view = make_tensor(rO_S.data(), make_layout(Shape <Shape<_4, _4>, _1, _2>{},
                                                          Stride<Shape<_4, _1>, _0, _16>{}));
    CUTE_STATIC_ASSERT_V(size<0>(rO_D) == _16{});
    CUTE_STATIC_ASSERT_V(size<1>(rO_D) == _1{});
    CUTE_STATIC_ASSERT_V(size<2>(rO_D) == _2{});
    #pragma unroll
    for (int i = 0; i < size<1>(rO_D); ++i) {
        #pragma unroll
        for (int j = 0; j < size<2>(rO_D); ++j) {
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                rO_D(k, i, j) = rO_view(k, i, j);
            }
        }
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_k64_opt(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    bool return_lse = params.softmax_lse_ptr != nullptr;
    bool return_max_logit = params.max_logit_ptr != nullptr;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockK = Kernel_traits::kBlockKSmem;
    constexpr int kBlockK_V = Kernel_traits::kBlockKSmemV;

    static_assert(kBlockM == 128 && kBlockN == 64 && kNWarps == 8 && kHeadDimV == 128,
        "Only support br=128, bc=64, nun_warps=8 and headdim_v=128");

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    const int kBlockM_stride = m_block * kBlockM;
    if (kBlockM_stride >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (kBlockM_stride + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div(kBlockM_stride + kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        //Tensor tOrO = make_tensor<Element>(shape(tOgO));
        //clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        flash::copy_zero_to_global<Is_even_MN, Is_even_K>(
            tOgO, tOcO, params.d_value, binfo.actual_seqlen_q - kBlockM_stride
        );
        if (return_lse) {
            auto gLSE = reinterpret_cast<int32_t *>(params.softmax_lse_ptr) + row_offset_lse;
            auto inf = INFINITY;
            #pragma unroll
            for (int m = 0; m < size<1>(tOgO); ++m) {
                const int row = get<0>(tOcO(0, m, 0));
                __builtin_mxc_stg_b32_predicator(gLSE + row, 0, *reinterpret_cast<int32_t *>(&inf), true, false, false,
                                        get<1>(tOcO(0, m, 0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
            }
        }

        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + kBlockM_stride * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + kBlockM_stride) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QK;
    auto gmem_thr_copy_QK = gmem_tiled_copy_QK.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QK.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QK.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QK.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QK.partition_D(sK);

    typename Kernel_traits::GmemTiledCopy_4x4 gmem_tiled_copy_V;
    auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tVgV = gmem_thr_copy_V.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVrV = make_fragment_like(tVgV);
    static_assert(decltype(size<1>(tVgV))::value == 1);
    Tensor tVrV_reshape = make_tensor(tVrV.data(), make_layout(make_shape(_8{}, _2{}, size<2>(tVrV))));
    Element *smem_v = reinterpret_cast<Element *>(sV.data().get()) + tidx / 64 * kBlockK_V * 16 + __lane_id() * 8;
    Tensor tVsV = make_tensor(make_smem_ptr(smem_v), make_layout(make_shape(_8{}, _2{}, size<2>(tVrV)),
                                                                 make_stride(_1{}, Int<kBlockK_V * 8>{}, Int<kBlockK_V * kBlockN>{})));

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = make_tensor<Element>(make_shape(_4{}, make_shape(_4{}, Int<kHeadDimV / kBlockK_V>{}), Int<kBlockN / 16>{}));

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    const auto d = params.d;
    const auto d_value = params.d_value;
    int n_block = n_block_max - 1;

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QK.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

     // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    Tensor tQrQ = make_fragment_like(tQgQ);
    flash::copy_b128<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, d,
                                       binfo.actual_seqlen_q - kBlockM_stride);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK_ori = smem_thr_copy_K.partition_S(sK);
    const int offset_swz334 = (__lane_id() % 2) ? ((__lane_id() / 16 % 2) ? -4 : 4) : 0;
    Tensor tSsK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(tSsK_ori.data().get()) + offset_swz334), tSsK_ori.layout());

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Element *smem_vt = reinterpret_cast<Element *>(sVt.data().get()) + __lane_id() * 8;
    Tensor tOsVt = make_tensor(make_smem_ptr(smem_vt), make_layout(Shape <_4, Shape<_2, _2, Int<kHeadDimV / kBlockK_V>>, Int<kBlockN / 16>>{},
                                                                   Stride<_1, Stride<_4, Int<kBlockK_V * 8>, Int<kBlockK_V * kBlockN>>, Int<kBlockK_V * 16>>{}));

    //
    // PREDICATES
    //

    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tKcK = gmem_thr_copy_QK.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV = gmem_thr_copy_V.partition_S(cV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Prologue
    cute::copy(gmem_tiled_copy_QK, tQrQ, tQsQ);
    Tensor tKrK = make_fragment_like(tKgK);
    flash::copy_b128<Is_even_MN, Is_even_K>(tKgK, tKrK, tKcK, d, binfo.actual_seqlen_k - n_block * kBlockN);

    flash::sync_threads();
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);

    flash::copy_multirow_b64<Is_even_MN, Is_even_K>(
            tVgV, tVrV, tVcV, d_value, binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    flash::sync_threads();

    const int gK_offset = -int(kBlockN * params.k_row_stride);
    const int gV_offset = -int(kBlockN * params.v_row_stride);
    const auto mask_offset = kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf);
    const auto kWarps_offset = kNWarps * 16;

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        swap_swz334(tKrK);
        cute::copy(gmem_tiled_copy_QK, tKrK, tKsK);
        clear(acc_s);
        // flash::sync_threads();
        // Advance gV
        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + gV_offset;
            flash::copy_multirow_b64</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);
        }
        flash::sync_threads();
        //flash::gemm_rs(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        if (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr)
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
            if(flash::use_attn_mask_merge_ldg(params)) {
                flash::apply_attn_mask</*mergeLdg=*/true, Is_even_MN>(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    mask_offset, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kWarps_offset,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            } else {
                flash::apply_attn_mask(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    mask_offset, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kWarps_offset,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            }
        }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, mask_offset, kWarps_offset, params.custom_alibi);

        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV_reshape, tVsV);
        // flash::sync_threads();
        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + gK_offset;
            flash::copy_b128</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKcK, d);
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, kNWarps, 1, kBlockN>(
                rP_drop, block_row_idx, block_col_idx, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/false, kNWarps, 1, kBlockN>(rP, block_row_idx, block_col_idx, n_block);
        }
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block > n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        swap_swz334(tKrK);
        cute::copy(gmem_tiled_copy_QK, tKrK, tKsK);
        clear(acc_s);
        // flash::sync_threads();
        // Advance gV
        tVgV.data() = tVgV.data() + gV_offset;
        flash::copy_multirow_b64</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, d_value);
        flash::sync_threads();
        //flash::gemm_rs(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV_reshape, tVsV);
        // Advance gK
        tKgK.data() = tKgK.data() + gK_offset;
        flash::copy_b128</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKcK, d);
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        if (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr)
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
            if(flash::use_attn_mask_merge_ldg(params)) {
                flash::apply_attn_mask</*mergeLdg=*/true, Is_even_MN>(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    mask_offset, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kWarps_offset,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            } else {
                flash::apply_attn_mask(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    mask_offset, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kWarps_offset,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            }
        }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, mask_offset, kWarps_offset, params.custom_alibi
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, kNWarps, 1, kBlockN>(
                rP_drop, block_row_idx, block_col_idx, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/false, kNWarps, 1, kBlockN>(rP, block_row_idx, block_col_idx, n_block);
        }
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    if (n_block == n_block_min) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        swap_swz334(tKrK);
        cute::copy(gmem_tiled_copy_QK, tKrK, tKsK);
        clear(acc_s);
        // flash::sync_threads();
        // Advance gV
        tVgV.data() = tVgV.data() + gV_offset;
        flash::copy_multirow_b64</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, d_value);
        flash::sync_threads();
        //flash::gemm_rs(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV_reshape, tVsV);
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        if (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr)
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
            if(flash::use_attn_mask_merge_ldg(params)) {
                flash::apply_attn_mask</*mergeLdg=*/true, Is_even_MN>(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    mask_offset, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kWarps_offset,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            } else {
                flash::apply_attn_mask(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    mask_offset, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kWarps_offset,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            }
        }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, mask_offset, kWarps_offset, params.custom_alibi
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, kNWarps, 1, kBlockN>(
                rP_drop, block_row_idx, block_col_idx, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/false, kNWarps, 1, kBlockN>(rP, block_row_idx, block_col_idx, n_block);
        }
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    auto o_reorder_shape = make_shape(_16{}, _1{}, _2{});
    Tensor acc_o_reorder = make_tensor<ElementAccum>(o_reorder_shape);
    reorder_acc_o(acc_o, acc_o_reorder);

    // Convert acc_o from fp32 to fp16/bf16
    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o_reorder, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutONoSwizzle{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    Element *smem_ptr_o = reinterpret_cast<Element *>(sO.data().get()) +
                                                      tidx / 64 * kBlockK_V * 16 +
                                                      __lane_id() % 16 * kBlockK_V +
                                                      __lane_id() / 16 * 16;
    Tensor taccOsO = make_tensor(make_smem_ptr(smem_ptr_o), make_layout(o_reorder_shape, Stride<_1, _0, Int<128 * kBlockK_V>>{}));

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { barrier(); }

    cute::copy(rO, taccOsO);

    const uint64_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
    const uint64_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    flash::sync_threads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);


    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    if (return_lse) {
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - kBlockM_stride) { gLSE(row) = lse(mi); }
            }
        }
    }

    if (return_max_logit) {
        const index_t row_offset_max_logit = bidb * params.seqlen_q * params.h + bidh * params.seqlen_q + m_block * kBlockM;
        Tensor gMaxLogit = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.max_logit_ptr) + row_offset_max_logit),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(softmax.row_max) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(softmax.row_max); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - kBlockM_stride) { gMaxLogit(row) = softmax.row_max(mi) * params.scale_softmax; }
            }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    flash::copy_reg_to_global<Is_even_MN, Is_even_K>(
        tOrO, tOgO, tOcO, d_value, binfo.actual_seqlen_q - kBlockM_stride
    );
}

// [br,bc,waves]: [128,128,4]
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_k64_N128(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using ElementSink = typename Kernel_traits::ElementSink;
    bool return_lse = params.softmax_lse_ptr != nullptr;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const uint32_t tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockK = Kernel_traits::kBlockKSmem;
    constexpr int kBlockK_V = Kernel_traits::kBlockKSmemV;

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    const uint32_t kBlockM_stride = m_block * kBlockM;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (kBlockM_stride >= binfo.actual_seqlen_q) return;
    const int m_block_stride = m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left;
    // const int n_block_min = !Is_local ? 0 : std::max(0, m_block_stride / kBlockN);
    const int n_block_min = !Is_local ? 0 : m_block_stride <0 ? 0 : std::max(0, m_block_stride / kBlockN);
    uint32_t n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        uint32_t curr_n =  std::max(0,(m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right);
        n_block_max = std::min(n_block_max,
                               uint32_t(cute::ceil_div(curr_n, kBlockN)));
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const uint64_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const uint64_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (uint32_t k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - kBlockM_stride
        );
        if (return_lse) {
            Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);
            #pragma unroll
            for (uint32_t m = 0; m < size<1>(tOgO); ++m) {
                const uint32_t row = get<0>(tOcO(0, m, 0));
                if (row < binfo.actual_seqlen_q - kBlockM_stride && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
            }
        }

        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    const uint32_t actual_bidh = params.h_h_k_ratio == 1 ? bidh : bidh / params.h_h_k_ratio;
    const uint64_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + kBlockM_stride * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const uint64_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + actual_bidh * params.k_head_stride;
    const uint64_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + actual_bidh * params.v_head_stride;
    const uint64_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + kBlockM_stride) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    constexpr bool first_layout_is_K = (kBlockM <= kBlockN);
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                                typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sQ = make_tensor(first_layout_is_K ? sK.data() + size(sK): sK.data(), typename Kernel_traits::SmemLayoutQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QK;
    auto gmem_thr_copy_QK = gmem_tiled_copy_QK.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QK.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QK.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QK.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QK.partition_D(sK);

    typename Kernel_traits::GmemTiledCopy_4x4 gmem_tiled_copy_V;
    auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tVgV = gmem_thr_copy_V.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVrV = make_fragment_like(tVgV);
    Element *smem_v = reinterpret_cast<Element *>(sV.data().get()) + tidx * 16;
    // static_assert(decltype(size<1>(tVgV))::value == 1);
    Tensor tVsV = make_tensor(make_smem_ptr(smem_v), make_layout(make_shape(size<0>(tVgV), size<1>(tVgV), size<2>(tVgV)),
                                                                                  /**ldg size (4x4) * Nthread */
                                                                 make_stride(_1{}, Int<16 * kNWarps * 64>{}, Int<kBlockK_V * kBlockN>{})));

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK_ori = smem_thr_copy_K.partition_S(sK);
    const int offset_swz334 = (__lane_id() % 2) ? ((__lane_id() / 16 % 2) ? -4 : 4) : 0;
    Tensor tSsK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(tSsK_ori.data().get()) + offset_swz334), tSsK_ori.layout());

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Element *smem_vt = reinterpret_cast<Element *>(sVt.data().get()) + __lane_id() / 16 * kBlockK * 4 + __lane_id() % 16 * 4;
    Tensor tOsVt = make_tensor(make_smem_ptr(smem_vt), make_layout(Shape <_4, Shape<Int<kBlockK / 16>, Int<kHeadDimV / kBlockK_V>>, Int<kBlockN / 16>>{},
                                                                   Stride<_1, Stride<Int<kBlockK_V>, Int<kBlockK_V * kBlockN>>, Int<kBlockK_V * 16>>{}));

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QK.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKcK = gmem_thr_copy_QK.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV = gmem_thr_copy_V.partition_S(cV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (uint32_t k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    Tensor tQrQ = make_fragment_like(tQgQ);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QK, tQgQ, tQrQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - kBlockM_stride);
    cute::copy(tQrQ, tQsQ);
    flash::sync_threads();
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);

    if (!first_layout_is_K){flash::sync_threads();}

    int n_block = n_block_max - 1;
    Tensor tKrK = make_fragment_like(tKgK);
    flash::copy_b128<Is_even_MN, Is_even_K>(tKgK, tKrK, tKcK, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);
    const bool has_sink = params.s_aux_ptr != nullptr;

    flash::Softmax<size<1>(acc_o)> softmax(has_sink);
    if (has_sink) {
        const int num_heads_q = params.h * params.ngroups;
        Tensor gS_aux = make_tensor(make_gmem_ptr(reinterpret_cast<ElementSink*>(params.s_aux_ptr)), make_shape(num_heads_q));
        softmax.template load_s_aux<kBlockM, kBlockN>(gS_aux, tiled_mma, params.ngroups, bidh, m_block);
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr uint32_t n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

    // Advance gV_part1
    flash::copy_multirow_b64_part1<Is_even_MN, Is_even_K>(tVgV, tVrV, tVcV, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

    #pragma unroll
    for (uint32_t masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        swap_swz334(tKrK);
        cute::copy(gmem_tiled_copy_QK, tKrK, tKsK);
        clear(acc_s);

        // load gV_part2
        if (masking_step > 0) {
            flash::copy_multirow_b64_part2</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);
        } else {
            flash::copy_multirow_b64_part2<Is_even_MN, Is_even_K>(
                tVgV, tVrV, tVcV, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        }

        flash::sync_threads<4>();

        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy_b128</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKcK, params.d);
        }

        flash::gemm_prefetch_lds</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        if (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr)
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
            if(flash::use_attn_mask_merge_ldg(params)) {
                flash::apply_attn_mask</*mergeLdg=*/true, Is_even_MN>(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    kBlockM_stride + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            } else {
                flash::apply_attn_mask(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    kBlockM_stride + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            }
        }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi);

        // perm && sts_V
        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<1>(tVrV)*size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV, tVsV);

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        __builtin_mxc_schedbound_begin();
        Tensor tCrB_copy_view = smem_thr_copy_V.retile_D(tOrVt);
        CUTE_STATIC_ASSERT_V(size<1>(tOsVt) == size<1>(tCrB_copy_view));            // N
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        // Convert acc_s from fp32 to fp16/bf16
        __builtin_mxc_schedbound_end();
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)

        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, 1>(
                rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }

        if (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/false, 1>(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        if (n_block > n_block_min) {
            // Advance gV_part1
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            flash::copy_multirow_b64_part1</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);
        }

        #pragma unroll
        for (uint32_t i = 0; i < size<2>(tOrP); ++i) {
            if (i < size<2>(tOrP) - 1) {
                cute::copy(smem_tiled_copy_V, tOsVt(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
        }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block > n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        swap_swz334(tKrK);
        cute::copy(gmem_tiled_copy_QK, tKrK, tKsK);
        clear(acc_s);

        // load gV_part2
        flash::copy_multirow_b64_part2</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);

        flash::sync_threads<4>();

        flash::gemm_prefetch_lds</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        // perm && sts_V
        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<1>(tVrV)*size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV, tVsV);

        // Advance gK
        tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
        __builtin_mxc_schedbound_begin();
        flash::copy_b128</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKcK, params.d);
        __builtin_mxc_schedbound_end();

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        if (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr)
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
            if(flash::use_attn_mask_merge_ldg(params)) {
                flash::apply_attn_mask</*mergeLdg=*/true, Is_even_MN>(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    kBlockM_stride + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            } else {
                flash::apply_attn_mask(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    kBlockM_stride + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            }
        }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi
        );

        // compute row max
        Tensor scores_max_prev = make_fragment_like(softmax.row_max);
        softmax.template get_row_max</*Is_first=*/false>(acc_s, scores_max_prev, params.scale_softmax_log2);

        // compute softmax
        softmax.template softmax_rescale_o_without_row_max</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, true>(acc_s, acc_o, scores_max_prev, params.scale_softmax_log2);

        flash::sync_threads<4>();

        __builtin_mxc_schedbound_begin();
        Tensor tCrB_copy_view = smem_thr_copy_V.retile_D(tOrVt);
        CUTE_STATIC_ASSERT_V(size<1>(tOsVt) == size<1>(tCrB_copy_view));            // N
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        // Convert acc_s from fp32 to fp16/bf16
        __builtin_mxc_schedbound_end();
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // WIP: below code frag is to replace CONVERT_TENSOR_TYPE
        // constexpr int numel = decltype(size(acc_s))::value;
        // mctlass::NumericArrayConverter<Element, ElementAccum, numel> convert_op;
        // auto frag = convert_op(*reinterpret_cast<const mctlass::Array<ElementAccum, numel> *>(acc_s.data()));
        // Tensor tOrP = make_tensor(make_rmem_ptr<Element>(&frag), acc_s.layout());

        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, 1>(
                rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }

        if (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/false, 1>(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        // Advance gV_part1
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        flash::copy_multirow_b64_part1</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);

        #pragma unroll
        for (uint32_t i = 0; i < size<2>(tOrP); ++i) {
            if (i < size<2>(tOrP) - 1) {
                cute::copy(smem_tiled_copy_V, tOsVt(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
        }
    }

        // These are the iterations where we don't need masking on S
    if (n_block == n_block_min) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        swap_swz334(tKrK);
        cute::copy(gmem_tiled_copy_QK, tKrK, tKsK);
        clear(acc_s);

        // load gV_part2
        flash::copy_multirow_b64_part2</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);

        flash::sync_threads<4>();

        flash::gemm_prefetch_lds</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        // perm && sts_V
        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<1>(tVrV)*size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV, tVsV);

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        if (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr)
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
            if(flash::use_attn_mask_merge_ldg(params)) {
                flash::apply_attn_mask</*mergeLdg=*/true, Is_even_MN>(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    kBlockM_stride + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            } else {
                flash::apply_attn_mask(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    kBlockM_stride + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            }
        }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi
        );

        // compute row max
        Tensor scores_max_prev = make_fragment_like(softmax.row_max);
        softmax.template get_row_max</*Is_first=*/false>(acc_s, scores_max_prev, params.scale_softmax_log2);

        // compute softmax
        softmax.template softmax_rescale_o_without_row_max</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, true>(acc_s, acc_o, scores_max_prev, params.scale_softmax_log2);

        flash::sync_threads<4>();

        __builtin_mxc_schedbound_begin();
        Tensor tCrB_copy_view = smem_thr_copy_V.retile_D(tOrVt);
        CUTE_STATIC_ASSERT_V(size<1>(tOsVt) == size<1>(tCrB_copy_view));            // N
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        // Convert acc_s from fp32 to fp16/bf16
        __builtin_mxc_schedbound_end();
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // WIP: below code frag is to replace CONVERT_TENSOR_TYPE
        // constexpr int numel = decltype(size(acc_s))::value;
        // mctlass::NumericArrayConverter<Element, ElementAccum, numel> convert_op;
        // auto frag = convert_op(*reinterpret_cast<const mctlass::Array<ElementAccum, numel> *>(acc_s.data()));
        // Tensor tOrP = make_tensor(make_rmem_ptr<Element>(&frag), acc_s.layout());

        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, 1>(
                rP_drop, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }

        if (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/false, 1>(rP, block_row_idx, block_col_idx, kNWarps, kBlockN, n_block);
        }

        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        #pragma unroll
        for (uint32_t i = 0; i < size<2>(tOrP); ++i) {
            if (i < size<2>(tOrP) - 1) {
                cute::copy(smem_tiled_copy_V, tOsVt(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
        }
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const uint64_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
    const uint64_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    if (return_lse) {
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (uint32_t mi = 0; mi < size(lse); ++mi) {
                const uint32_t row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - kBlockM_stride) { gLSE(row) = lse(mi); }
            }
        }
    }
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (uint32_t k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - kBlockM_stride
    );
}

// [br,bc,waves]: [64,64,4], [64,32,4], [128,64,4], [128,64,8]
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params, bool Merge_attn_mask_ldg = false>
__forceinline__ __device__ void compute_attn_1rowblock_k64_base(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using ElementSink = typename Kernel_traits::ElementSink;
    bool return_lse = params.softmax_lse_ptr != nullptr;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const uint32_t tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockK = Kernel_traits::kBlockKSmem;
    constexpr int kBlockK_V = Kernel_traits::kBlockKSmemV;
    constexpr bool Is_even_M = Is_even_MN;
    constexpr bool Is_even_N = Is_even_MN;

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    const uint32_t kBlockM_stride = m_block * kBlockM;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (kBlockM_stride >= binfo.actual_seqlen_q) return;
    const int m_block_stride = m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left;
    // const int n_block_min = !Is_local ? 0 : std::max(0, m_block_stride / kBlockN);
    const int n_block_min = !Is_local ? 0 : m_block_stride <0 ? 0 : std::max(0, m_block_stride / kBlockN);
    uint32_t n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        uint32_t curr_n =  std::max(0,(m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right);
        n_block_max = std::min(n_block_max,
                               uint32_t(cute::ceil_div(curr_n, kBlockN)));
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const uint64_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const uint64_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (uint32_t k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - kBlockM_stride
        );
        if (return_lse) {
            Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);
            #pragma unroll
            for (uint32_t m = 0; m < size<1>(tOgO); ++m) {
                const uint32_t row = get<0>(tOcO(0, m, 0));
                if (row < binfo.actual_seqlen_q - kBlockM_stride && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
            }
        }

        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    const uint32_t actual_bidh = params.h_h_k_ratio == 1 ? bidh : bidh / params.h_h_k_ratio;
    const uint64_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + kBlockM_stride * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const uint64_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + actual_bidh * params.k_head_stride;
    const uint64_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + actual_bidh * params.v_head_stride;
    const uint64_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + kBlockM_stride) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    constexpr bool first_layout_is_K = (kBlockM <= kBlockN);
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                                typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sQ = make_tensor(first_layout_is_K ? sK.data() + size(sK): sK.data(), typename Kernel_traits::SmemLayoutQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QK;
    auto gmem_thr_copy_QK = gmem_tiled_copy_QK.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QK.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QK.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QK.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QK.partition_D(sK);

    typename Kernel_traits::GmemTiledCopy_4x4 gmem_tiled_copy_V;
    auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tVgV = gmem_thr_copy_V.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVrV = make_fragment_like(tVgV);
    Element *smem_v = reinterpret_cast<Element *>(sV.data().get()) + tidx * 16;
    Tensor tVsV = make_tensor(make_smem_ptr(smem_v), make_layout(make_shape(size<0>(tVgV), size<1>(tVgV), size<2>(tVgV)),
                                                                                  /**ldg size (4x4) * Nthread */
                                                                 make_stride(_1{}, Int<16 * kNWarps * 64>{}, Int<kBlockK_V * kBlockN>{})));

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    // =============preprocessing code for attn mask begin========
    // these tensor only use when attn_mask = True
    const index_t row_offset_mask = (n_block_max - 1) * kBlockN * params.attn_mask_col_stride + m_block * kBlockM * params.attn_mask_row_stride
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
    Tensor gMask = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.attn_mask_ptr) + row_offset_mask),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.attn_mask_row_stride, params.attn_mask_col_stride));
    Tensor sMask = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutMask{});
    typename Kernel_traits::GmemTiledCopyMask gmem_tiled_copy_mask;
    auto gmem_thr_copy_mask = gmem_tiled_copy_mask.get_thread_slice(tidx);
    Tensor tMaskgMask = gmem_thr_copy_mask.partition_S(gMask);
    Tensor tMasksMask = gmem_thr_copy_mask.partition_D(sMask);
    Tensor tMaskrMask = make_fragment_like(tMaskgMask);
    Tensor cMask = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor tMaskcMask = gmem_thr_copy_mask.partition_D(cMask);                           // (MMA,MMA_N,MMA_N)
    auto smem_tiled_copy_mask = make_tiled_copy_C(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_mask = smem_tiled_copy_mask.get_thread_slice(tidx);
    Tensor tSsMask = smem_thr_copy_mask.partition_S(sMask);
    Tensor tSrMask = make_fragment_like(tSsMask);
    int seqlen_k_padded = params.seqlen_k;
    if constexpr (Has_attn_mask) {
        if (params.attn_mask_col_shape != seqlen_k_padded && params.attn_mask_col_shape > 1) {
            seqlen_k_padded = params.attn_mask_col_shape;
        }
    }
    // =============preprocessing code for attn mask end========

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK_ori = smem_thr_copy_K.partition_S(sK);
    const int offset_swz334 = (__lane_id() % 2) ? ((__lane_id() / 16 % 2) ? -4 : 4) : 0;
    Tensor tSsK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(tSsK_ori.data().get()) + offset_swz334), tSsK_ori.layout());

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Element *smem_vt = reinterpret_cast<Element *>(sVt.data().get()) + __lane_id() / 16 * kBlockK * 4 + __lane_id() % 16 * 4;
    Tensor tOsVt = make_tensor(make_smem_ptr(smem_vt), make_layout(Shape <_4, Shape<Int<kBlockK / 16>, Int<kHeadDimV / kBlockK_V>>, Int<kBlockN / 16>>{},
                                                                   Stride<_1, Stride<Int<kBlockK_V>, Int<kBlockK_V * kBlockN>>, Int<kBlockK_V * 16>>{}));

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QK.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKcK = gmem_thr_copy_QK.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV = gmem_thr_copy_V.partition_S(cV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (uint32_t k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    Tensor tQrQ = make_fragment_like(tQgQ);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QK, tQgQ, tQrQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - kBlockM_stride);
    cute::copy(tQrQ, tQsQ);
    flash::sync_threads();
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);

    if (!first_layout_is_K){flash::sync_threads();}
    int n_block = n_block_max - 1;
    Tensor tKrK = make_fragment_like(tKgK);
    flash::copy_b128<Is_even_MN, Is_even_K>(tKgK, tKrK, tKcK, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);
    const bool has_sink = params.s_aux_ptr != nullptr;

    flash::Softmax<size<1>(acc_o)> softmax(has_sink);
    if (has_sink) {
        const int num_heads_q = params.h * params.ngroups;
        Tensor gS_aux = make_tensor(make_gmem_ptr(reinterpret_cast<ElementSink*>(params.s_aux_ptr)), make_shape(num_heads_q));
        softmax.template load_s_aux<kBlockM, kBlockN>(gS_aux, tiled_mma, params.ngroups, bidh, m_block);
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr uint32_t n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (uint32_t masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        swap_swz334(tKrK);
        cute::copy(gmem_tiled_copy_QK, tKrK, tKsK);
        clear(acc_s);
        if constexpr (Has_attn_mask) {
            // this schedbound to prevent the compiler scheduling the ldg attn_mask after ldgV
            if (masking_step > 0) {
                tMaskgMask.data() = tMaskgMask.data() + -int(kBlockN * params.attn_mask_col_stride);
            }
            flash::load_attn_mask<Merge_attn_mask_ldg, Is_even_M, Is_even_N>(tMaskgMask, tMaskrMask, tMaskcMask, seqlen_k_padded - n_block * kBlockN, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        // Advance gV
        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            flash::copy_multirow_b64</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);
        } else {
            flash::copy_multirow_b64<Is_even_MN, Is_even_K>(
                tVgV, tVrV, tVcV, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        flash::sync_threads();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        if constexpr (Has_attn_mask) {
            cute::copy(gmem_tiled_copy_mask, tMaskrMask, tMasksMask);
            flash::sync_threads();
            cute::copy(smem_tiled_copy_mask, tSsMask, tSrMask);
            flash::apply_attn_mask(acc_s, tSrMask, params.scale_softmax);
            flash::sync_threads();
        }

        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy_b128</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKcK, params.d);
        }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }
        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi);

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<1>(tVrV)*size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV, tVsV);
        flash::sync_threads();

        __builtin_mxc_schedbound_begin();
        Tensor tCrB_copy_view = smem_thr_copy_V.retile_D(tOrVt);
        CUTE_STATIC_ASSERT_V(size<1>(tOsVt) == size<1>(tCrB_copy_view));            // N
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        // Convert acc_s from fp32 to fp16/bf16
        __builtin_mxc_schedbound_end();
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if constexpr (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, kNWarps, 1, kBlockN>(
                rP_drop, block_row_idx, block_col_idx, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if constexpr (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/false, kNWarps, 1, kBlockN>(rP, block_row_idx, block_col_idx, n_block);
        }
        // flash::sync_threads();
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        #pragma unroll
        for (uint32_t i = 0; i < size<2>(tOrP); ++i) {
            if (i < size<2>(tOrP) - 1) {
                cute::copy(smem_tiled_copy_V, tOsVt(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
        }

        // flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block > n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        swap_swz334(tKrK);
        cute::copy(gmem_tiled_copy_QK, tKrK, tKsK);
        clear(acc_s);
        if constexpr (Has_attn_mask) {
            // this schedbound to prevent the compiler scheduling the ldg attn_mask after ldgV
            tMaskgMask.data() = tMaskgMask.data() + -int(kBlockN * params.attn_mask_col_stride);
            flash::load_attn_mask<Merge_attn_mask_ldg, Is_even_M, /*Is_even_N=*/true>(tMaskgMask, tMaskrMask, tMaskcMask, seqlen_k_padded - n_block * kBlockN, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        // Advance gV
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        flash::copy_multirow_b64</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);
        flash::sync_threads();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        // Advance gK
        tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
        flash::copy_b128</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKcK, params.d);

        if constexpr (Has_attn_mask) {
            cute::copy(gmem_tiled_copy_mask, tMaskrMask, tMasksMask);
            flash::sync_threads<4>();
            cute::copy(smem_tiled_copy_mask, tSsMask, tSrMask);
            flash::apply_attn_mask(acc_s, tSrMask, params.scale_softmax);
            flash::sync_threads<4>();
        }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }
        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi
        );


        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, false, true>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<1>(tVrV)*size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV, tVsV);
        flash::sync_threads<4>();
        __builtin_mxc_schedbound_begin();
        Tensor tCrB_copy_view = smem_thr_copy_V.retile_D(tOrVt);
        CUTE_STATIC_ASSERT_V(size<1>(tOsVt) == size<1>(tCrB_copy_view));            // N
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        // Convert acc_s from fp32 to fp16/bf16
        __builtin_mxc_schedbound_end();
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // WIP: below code frag is to replace CONVERT_TENSOR_TYPE
        // constexpr int numel = decltype(size(acc_s))::value;
        // mctlass::NumericArrayConverter<Element, ElementAccum, numel> convert_op;
        // auto frag = convert_op(*reinterpret_cast<const mctlass::Array<ElementAccum, numel> *>(acc_s.data()));
        // Tensor tOrP = make_tensor(make_rmem_ptr<Element>(&frag), acc_s.layout());

        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if constexpr (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, kNWarps, 1, kBlockN>(
                rP_drop, block_row_idx, block_col_idx, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if constexpr (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/false, kNWarps, 1, kBlockN>(rP, block_row_idx, block_col_idx, n_block);
        }
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        // flash::sync_threads();
        #pragma unroll
        for (uint32_t i = 0; i < size<2>(tOrP); ++i) {
            if (i < size<2>(tOrP) - 1) {
                cute::copy(smem_tiled_copy_V, tOsVt(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
        }

        // flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    }

        // These are the iterations where we don't need masking on S
    if (n_block == n_block_min) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        swap_swz334(tKrK);
        cute::copy(gmem_tiled_copy_QK, tKrK, tKsK);
        clear(acc_s);
        if constexpr (Has_attn_mask) {
            // this schedbound to prevent the compiler scheduling the ldg attn_mask after ldgV
            tMaskgMask.data() = tMaskgMask.data() + -int(kBlockN * params.attn_mask_col_stride);
            flash::load_attn_mask<Merge_attn_mask_ldg, Is_even_M, /*Is_even_N=*/true>(tMaskgMask, tMaskrMask, tMaskcMask, seqlen_k_padded - n_block * kBlockN, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        // Advance gV
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        flash::copy_multirow_b64</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);
        flash::sync_threads();
        //flash::gemm_rs(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);


        if constexpr (Has_attn_mask) {
            cute::copy(gmem_tiled_copy_mask, tMaskrMask, tMasksMask);
            flash::sync_threads();
            cute::copy(smem_tiled_copy_mask, tSsMask, tSrMask);
            flash::apply_attn_mask(acc_s, tSrMask, params.scale_softmax);
            flash::sync_threads();
        }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }
        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi
        );

        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<1>(tVrV)*size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV, tVsV);

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        __builtin_mxc_schedbound_begin();
        Tensor tCrB_copy_view = smem_thr_copy_V.retile_D(tOrVt);
        CUTE_STATIC_ASSERT_V(size<1>(tOsVt) == size<1>(tCrB_copy_view));            // N
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        // Convert acc_s from fp32 to fp16/bf16
        __builtin_mxc_schedbound_end();
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // WIP: below code frag is to replace CONVERT_TENSOR_TYPE
        // constexpr int numel = decltype(size(acc_s))::value;
        // mctlass::NumericArrayConverter<Element, ElementAccum, numel> convert_op;
        // auto frag = convert_op(*reinterpret_cast<const mctlass::Array<ElementAccum, numel> *>(acc_s.data()));
        // Tensor tOrP = make_tensor(make_rmem_ptr<Element>(&frag), acc_s.layout());

        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if constexpr (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, kNWarps, 1, kBlockN>(
                rP_drop, block_row_idx, block_col_idx, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if constexpr (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/false, kNWarps, 1, kBlockN>(rP, block_row_idx, block_col_idx, n_block);
        }
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        // flash::sync_threads();
        #pragma unroll
        for (uint32_t i = 0; i < size<2>(tOrP); ++i) {
            if (i < size<2>(tOrP) - 1) {
                cute::copy(smem_tiled_copy_V, tOsVt(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
        }

        // flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if constexpr (Kernel_traits::Share_Q_K_smem) { flash::sync_threads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const uint64_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
    const uint64_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    flash::sync_threads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    if (return_lse) {
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (uint32_t mi = 0; mi < size(lse); ++mi) {
                const uint32_t row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - kBlockM_stride) { gLSE(row) = lse(mi); }
            }
        }
    }
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (uint32_t k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - kBlockM_stride
    );
}


} // namespace flash
