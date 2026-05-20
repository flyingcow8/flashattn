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

#include "flash_fwd_kernel_hdim64.h"
#include "flash_fwd_kernel_hdim128_blockM64.h"
#include "flash_fwd_kernel_hdim128_blockM128.h"
#include "flash_fwd_kernel_hdim192.h"
#include "flash_fwd_kernel_hdim256.h"

#include "flash_fwd_kernel_k32_blockN32.h"
#include "flash_fwd_kernel_k32.h"
#include "flash_fwd_kernel_k64.h"
#include "flash_fwd_kernel_k64_tail_32.h"
#include "flash_fwd_kernel_k256.h"

namespace flash {
namespace xcore1000 {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];
    Tensor tQKVrQKV_t = make_tensor<uint32_t>(make_shape(Int<Kernel_traits::kRegSize>{}));
    auto tQrQ = tQKVrQKV_t.data();
    auto tVrV = tQKVrQKV_t.data();
    auto tKrK = tQKVrQKV_t.data() + Kernel_traits::kRegSize / 2;

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
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
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }

        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

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
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);
    if (Kernel_traits::Share_Q_K_smem) {
        flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, tQpQ,
                                                 binfo.actual_seqlen_q - m_block * kBlockM);
    } else {
            flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                                 binfo.actual_seqlen_q - m_block * kBlockM);
    }

    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tQrQ, tQsQ);
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);
    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(
                tKgK, tKrK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
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
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tKrK, tKsK);
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            //flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
            flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV);
        } else {
            // Clear the smem tiles to account for predicated off loads
            // flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            //     gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            // );
            flash::copy_global_to_reg<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                tVgV, tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
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

        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            //flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || Has_attn_mask>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || Has_attn_mask>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        //Tensor rP = flash::convert_type<Element>(acc_s);
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

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        //Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tKrK, tKsK);
        __syncthreads();
        // Advance gV
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        //flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        flash::cp_async_wait<0>();
        flash::copy_reg_to_share(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            //flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        if (Has_attn_mask) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
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

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask>(acc_s, acc_o, params.scale_softmax_log2);

        //Tensor rP = flash::convert_type<Element>(acc_s);
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

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        //Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    //Tensor rO = flash::convert_type<Element>(acc_o);
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

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);

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

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params, int Rowblock_Parallel_Num = 1, bool Merge_attn_mask_ldg = false>
__forceinline__ __device__ void compute_attn(const Params &params, const int &m_block_max, const int &block_type) {

    const dim3 bidInf = get_bidInfo(block_type);
    const int m_block = bidInf.x;
    // The block index for the batch.
    const int bidb = bidInf.y;
    // The block index for the head.
    const int bidh = bidInf.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    if constexpr (Kernel_traits::kHeadDim == 32 || Kernel_traits::kHeadDim == 160) {
        if constexpr (Kernel_traits::kBlockN == 32) {
            compute_attn_1rowblock_k32_blockN32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, bidh, m_block);
            if constexpr (Rowblock_Parallel_Num == 2) {
                compute_attn_1rowblock_k32_blockN32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, bidh, m_block_max - m_block - 1);
            }
        } else {
            compute_attn_1rowblock_k32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, bidh, m_block);
            if constexpr (Rowblock_Parallel_Num == 2) {
                compute_attn_1rowblock_k32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, bidh, m_block_max - m_block - 1);
            }
        }
        return;
    } else if constexpr (Kernel_traits::kHeadDim == 96) {
        compute_attn_1rowblock_k64_tail_32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, bidh, m_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            compute_attn_1rowblock_k64_tail_32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, bidh, m_block_max - m_block - 1);
        }
    } else if constexpr (Kernel_traits::kHeadDim == 256) {
        const int curr_block = m_block_max - m_block - 1;
        compute_attn_1rowblock_opt_hdim256<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, bidh, curr_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            compute_attn_1rowblock_opt_hdim256<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, params.h - bidh - 1, m_block);
        }
        return;
    } else if constexpr (Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 128) {
        const int curr_block = m_block_max - m_block - 1;
        compute_attn_1rowblock_k64_opt<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, bidh, curr_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            compute_attn_1rowblock_k64_opt<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, params.h - bidh - 1, m_block);
        }
        return;
    } else if constexpr (Kernel_traits::kHeadDim == 192) {
        const int curr_block = m_block_max - m_block - 1;
        compute_attn_1rowblock_opt_hdim192<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, bidh, curr_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            compute_attn_1rowblock_opt_hdim192<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, params.h - bidh - 1, m_block);
        }
        return;
    } else if constexpr (Kernel_traits::kHeadDim == 128) {
        const int curr_block = m_block_max - m_block - 1;
        if constexpr(Kernel_traits::kBlockM == 64) {
            compute_attn_1rowblock_opt_hdim128_blockM64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, bidh, curr_block);
            if constexpr (Rowblock_Parallel_Num == 2) {
                compute_attn_1rowblock_opt_hdim128_blockM64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, params.h - bidh - 1, m_block);
            }
            return;
        }
        compute_attn_1rowblock_opt_hdim128_blockM128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, bidh, curr_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            compute_attn_1rowblock_opt_hdim128_blockM128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, params.h - bidh - 1, m_block);
        }
        return;
    } else if constexpr (Kernel_traits::kHeadDim == 64) {
        const int curr_block = m_block_max - m_block - 1;
        if constexpr (Kernel_traits::kBlockM == 128 && Has_attn_mask && Kernel_traits::kNWarps == 4) { //opt_hdim64 kernel has dedicated opt for hdim64 with attn_mask case (conformer)
            compute_attn_1rowblock_opt_hdim64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, bidh, curr_block);
            if constexpr (Rowblock_Parallel_Num == 2) {
                compute_attn_1rowblock_opt_hdim64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, params.h - bidh - 1, m_block);
            }
            return;
        }
        if constexpr (Kernel_traits::kBlockN == 128 && !Has_attn_mask) {
            compute_attn_1rowblock_k64_N128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, bidh, curr_block);
            if constexpr (Rowblock_Parallel_Num == 2) {
                compute_attn_1rowblock_k64_N128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, params.h - bidh - 1, m_block);
            }
        } else { // kBlockN == 64 && kBlockN == 32 && kBlockN == 16
            compute_attn_1rowblock_k64_base<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, bidh, curr_block);
            if constexpr (Rowblock_Parallel_Num == 2) {
                compute_attn_1rowblock_k64_base<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, params.h - bidh - 1, m_block);
            }
        }
        return;
    } else if constexpr (Kernel_traits::kHeadDim == 512) {
        const int curr_block = m_block_max - m_block - 1;
        compute_attn_1rowblock_k256<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, bidh, curr_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            compute_attn_1rowblock_k256<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, params.h - bidh - 1, m_block);
        }
    } else {
        compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, bidh, m_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, bidh, m_block_max - m_block - 1);
        }
    }

}

} // xcore1000

} // namespace flash
