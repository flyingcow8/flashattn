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

namespace xcore1500 {

using namespace cute;

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_k64(const Params &params, const int bidb, const int bidh, const int m_block) {

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
    constexpr int Num_Stages = Kernel_traits::Num_Stages;

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if constexpr ((Is_causal || Is_local || !Is_even_MN)) {
        if (n_block_max <= n_block_min) {
            const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
                + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
            const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
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
                for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
            }
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
            );
            if (return_lse) {
                Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);
                #pragma unroll
                for (int m = 0; m < size<1>(tOgO); ++m) {
                    const int row = get<0>(tOcO(0, m, 0));
                    if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
                }
            }

            return;
        }
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

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
    Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutK_k_stage{});
    Tensor sKNoSwizzle = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKNoSwizzle_k_stage{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV242{});
    Tensor sVNoSwizzle = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyBsm gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    const int swz333_offset = cute::get_swizzle_offset<8,3,3,3>(tidx);
    const int swz242_offset = cute::get_swizzle_offset<8,2,4,2>(tidx);
    Tensor tQgQ_Noswizzle = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQgQ = make_tensor(tQgQ_Noswizzle.data() + swz333_offset, layout(tQgQ_Noswizzle));
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQNoSwizzle);
    Tensor tKgK_Noswizzle = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKgK = make_tensor(tKgK_Noswizzle.data() + swz333_offset, layout(tKgK_Noswizzle));
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sKNoSwizzle);
    Tensor tVgV_Noswizzle = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVgV = make_tensor(tVgV_Noswizzle.data() + swz242_offset, layout(tVgV_Noswizzle));
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sVNoSwizzle);

    typename Kernel_traits::TiledMma tiled_mma_o;
    typename Kernel_traits::TiledMma_S tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx);
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK(_, _, 0));                  // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma_s.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB128{}, tiled_mma_s);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB128{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::LDSB64Trans4x16Atom{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ_Noswizzle = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKcK_Noswizzle = gmem_thr_copy_QKV.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV_Noswizzle = gmem_thr_copy_QKV.partition_S(cV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tQcQ = make_tensor(tQcQ_Noswizzle.data() + make_coord(0, swz333_offset), layout(tQcQ_Noswizzle));
    Tensor tKcK = make_tensor(tKcK_Noswizzle.data() + make_coord(0, swz333_offset), layout(tKcK_Noswizzle));
    Tensor tVcV = make_tensor(tVcV_Noswizzle.data() + make_coord(0, swz242_offset), layout(tVcV_Noswizzle));
    int Ksmem_read_index = 0;
    int Ksmem_write_index = 0;

    // Prologue

    flash::copy_b128_bsm_async<Is_even_MN, Is_even_K>(tQgQ, tQsQ, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    if constexpr (Kernel_traits::Share_Q_K_smem) {
        flash::barrier_gvm<0>();
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);
        flash::sync_threads();
    }

    int n_block = n_block_max - 1;
    flash::copy_b128_bsm_async<Is_even_MN, Is_even_K>(tKgK, tKsK(_, _, _, Ksmem_write_index), tKcK, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    if constexpr (Num_Stages == 2) {
        Ksmem_write_index ^= 1;
    }

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

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
    constexpr int ldg_K_cnt = size<1>(tKgK) * size<2>(tKgK);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::barrier_gvm<0>();
        // Advance gV
        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            flash::copy_b128_bsm_async</*Is_even_MN=*/true, Is_even_K>(tVgV, tVsV, tVcV, params.d_value);
        } else {
            flash::copy_b128_bsm_async<Is_even_MN, Is_even_K>(
                tVgV, tVsV, tVcV, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, Ksmem_read_index), tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);
        if constexpr (Num_Stages == 2) {
            Ksmem_read_index ^= 1;
        } else {
            flash::barrier();
        }

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr)
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
            if(flash::use_attn_mask_merge_ldg(params)) {
                flash::apply_attn_mask</*mergeLdg=*/true, Is_even_MN>(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
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
                    m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
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
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi);

        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy_b128_bsm_async</*Is_even_MN=*/true, Is_even_K>(tKgK, tKsK(_, _, _, Ksmem_write_index), tKcK, params.d);
            if constexpr (Num_Stages == 2) {
                Ksmem_write_index ^= 1;
            }
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, false, true>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, false, true>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
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
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::barrier_gvm<0>();
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma_o, smem_tiled_copy_V, smem_thr_copy_V);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block > n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        // Advance gV
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        flash::barrier_gvm<0>();
        flash::copy_b128_bsm_async</*Is_even_MN=*/true, Is_even_K>(tVgV, tVsV, tVcV, params.d_value);
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, Ksmem_read_index), tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);
        if constexpr (Num_Stages == 2) {
            Ksmem_read_index ^= 1;
        } else {
            flash::barrier();
        }

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr)
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
            if(flash::use_attn_mask_merge_ldg(params)) {
                flash::apply_attn_mask</*mergeLdg=*/true, Is_even_MN>(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
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
                    m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            }
        }

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi
        );

        Tensor scores_max_prev = make_fragment_like(softmax.row_max);
        softmax.template get_row_max</*Is_first=*/false>(acc_s, scores_max_prev, params.scale_softmax_log2);
        __builtin_mxc_schedbound_begin();
        tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
        flash::copy_b128_bsm_async</*Is_even_MN=*/true, Is_even_K>(tKgK, tKsK(_, _, _, Ksmem_write_index), tKcK, params.d);
        if constexpr (Num_Stages == 2) {
            Ksmem_write_index ^= 1;
        }
        __builtin_mxc_schedbound_end();
        softmax.template softmax_rescale_o_without_row_max</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, true>(acc_s, acc_o, scores_max_prev, params.scale_softmax_log2);

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
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::barrier_gvm<ldg_K_cnt>();
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma_o, smem_tiled_copy_V, smem_thr_copy_V);
    }

    if (n_block == n_block_min) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::barrier_gvm<0>();
        // Advance gV
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        flash::copy_b128_bsm_async</*Is_even_MN=*/true, Is_even_K>(tVgV, tVsV, tVcV, params.d_value);
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, Ksmem_read_index), tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr)
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
            if(flash::use_attn_mask_merge_ldg(params)) {
                flash::apply_attn_mask</*mergeLdg=*/true, Is_even_MN>(
                    scores,
                    n_block * kBlockN, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
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
                    m_block * kBlockM + (tidx / 64) * 16 + tidx % 16, /*row_idx_offset_*/
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
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, false, true>(acc_s, acc_o, params.scale_softmax_log2);

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
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        barrier_gvm<0>();
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma_o, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_o);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if constexpr (Kernel_traits::Share_Q_K_smem) { sync_threads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    sync_threads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
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
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
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
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gMaxLogit(row) = softmax.row_max(mi) * params.scale_softmax; }
            }
        }
    }
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    flash::copy_reg_to_global<Is_even_MN, Is_even_K>(
        tOrO, tOgO, tOcO, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

} // namespace xcore1500

} // namespace flash
