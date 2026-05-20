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

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_k256(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    bool return_lse = params.softmax_lse_ptr != nullptr;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int wave_idx = tidx / 64;
    const int wave_group_idx = wave_idx / 4;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kBlockK = Kernel_traits::kBlockKSmem;
    constexpr int kBlockK_V = Kernel_traits::kBlockKSmemV;
    constexpr int kBlockKGmemV = 256;
    static_assert(kHeadDimV % 256 == 0);

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
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
            Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});
            #pragma unroll
            for (int m = 0; m < size<1>(tOgO); ++m) {
                const int row = get<0>(tOcO(0, m, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
            }
        }

        return;
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
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutK424{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sRowMax = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum *>(smem_) + size(sK) / 2), //sK is bf16, yet row_max is fp32
                            typename Kernel_traits::SmemLayoutRowMax{});
    Tensor sRowSum = make_tensor(sRowMax.data() + size(sRowMax), typename Kernel_traits::SmemLayoutRowSum{}); //row_sum need exist together with row_max
    Tensor sP = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutP{});

    typename Kernel_traits::GmemTiledCopyB128 gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    typename Kernel_traits::GmemTiledCopyB64 gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    typename Kernel_traits::GmemTiledCopy_4x4 gmem_tiled_copy_V;
    auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tVgV = gmem_thr_copy_V.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVrV = make_fragment_like(tVgV);
    static_assert(decltype(size<1>(tVgV))::value == 1);
    Element *smem_v = reinterpret_cast<Element *>(sV.data().get()) + tidx * 16;
    Tensor tVsV = make_tensor(make_smem_ptr(smem_v), make_layout(make_shape(size<0>(tVgV), size<1>(tVgV), size<2>(tVgV)),
                                                                 make_stride(_1{}, Int<16 * kNThreads>{}, Int<kBlockKGmemV * kBlockN>{})));
    // gemm S is 4x1 wave layout, wave(n) compute the same S with wave(n + 4)
    int tidx_mma_s = tidx & 0xFF;
    typename Kernel_traits::TiledMmaS_k256 tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    typename Kernel_traits::TiledMmaO_k256 tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma_s.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_s);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx_mma_s);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx_mma_s);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Element *smem_vt = reinterpret_cast<Element *>(sVt.data().get()) + wave_idx / 4 * kBlockK_V + __lane_id() / 16 * kBlockK_V * 4 +
                                                  __lane_id() % 16 * 4;
    Tensor tOsVt = make_tensor(make_smem_ptr(smem_vt), make_layout(Shape <_4, Shape<_2, Int<kHeadDim / kBlockK_V>>, Int<kBlockN / 16>>{}, // (MMA,MMA_N,MMA_K)
                                                                   Stride<_1, Stride<Int<kBlockK_V * 2>, Int<kBlockK_V * kBlockN>>, Int<kBlockK_V * 16>>{}));

    auto smem_tiled_copy_stsP = make_tiled_copy_C(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_s);
    auto smem_thr_copy_stsP = smem_tiled_copy_stsP.get_thread_slice(tidx_mma_s);
    Tensor tPsP = smem_thr_copy_stsP.partition_D(sP);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    auto smem_tiled_copy_ldsP = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_o);
    auto smem_thr_copy_ldsP = smem_tiled_copy_ldsP.get_thread_slice(tidx);
    Tensor tOsP = smem_thr_copy_ldsP.partition_S(sP);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKcK = gmem_thr_copy_KV.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV = gmem_thr_copy_V.partition_S(cV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    Tensor tQrQ = make_fragment_like(tQgQ);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tQgQ, tQrQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    cute::copy(tQrQ, tQsQ);
    flash::sync_threads();
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);

    flash::sync_threads();
    int n_block = n_block_max - 1;
    Tensor tKrK = make_fragment_like(tKgK);
    flash::copy_b64<Is_even_MN, Is_even_K>(tKgK, tKrK, tKcK, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

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
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        cute::copy(gmem_tiled_copy_KV, tKrK, tKsK);
        clear(acc_s);
        // Advance gV
        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            flash::copy_multirow_b64</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);
        } else {
            flash::copy_multirow_b64<Is_even_MN, Is_even_K>(
                tVgV, tVrV, tVcV, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        flash::sync_threads();
        if (wave_group_idx == 0) {
            flash::gemm_prefetch_lds</*A_in_regs=*/Kernel_traits::Is_Q_in_regs, false, 3>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
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
                acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) % 4 * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi);
        }

        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy_b64</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKcK, params.d);
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, false, true>(acc_s, acc_o, sRowMax, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, false, true>(acc_s, acc_o, sRowMax, params.scale_softmax_log2);
        flash::sync_threads();
        Tensor tOrP = make_tensor(make_rmem_ptr(reinterpret_cast<Element *>(acc_s.data())), acc_s.layout());
        if (wave_group_idx == 0) {
            // Convert acc_s from fp32 to fp16/bf16
            CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
            cute::copy(rP, tOrP);
            cute::copy(smem_tiled_copy_stsP, tOrP, tPsP);
        }
        flash::sync_threads();
        if (wave_group_idx == 1) {
            cute::copy(smem_tiled_copy_ldsP, tOsP, tOrP);
        }
        flash::sync_threads();

        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<1>(tVrV)*size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV, tVsV);
        flash::sync_threads();
        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(tOrP);
            cute::copy(tOrP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, kNWarps, 1, kBlockN>(
                rP_drop, block_row_idx, block_col_idx, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/false, kNWarps, 1, kBlockN>(tOrP, block_row_idx, block_col_idx, n_block);
        }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma_o, smem_tiled_copy_V, smem_thr_copy_V);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        cute::copy(gmem_tiled_copy_KV, tKrK, tKsK);
        clear(acc_s);
        // Advance gV
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        flash::copy_multirow_b64</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tVcV, params.d_value);
        flash::sync_threads();
        if (wave_group_idx == 0) {
            flash::gemm_prefetch_lds</*A_in_regs=*/Kernel_traits::Is_Q_in_regs, false, 3>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
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
                acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) % 4 * 16 + (tidx & 0xf), kNWarps * 16, params.custom_alibi
            );
        }


        typedef __NATIVE_VECTOR__(2, int) VecType;
        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tKgK); ++m) {
            bool row_mask = true;
            #pragma unroll
            for (int k = 0; k < size<2>(tKgK)/2; ++k) {
                auto src_ptr = (VecType *)(tKgK(_, m, k).data().get());    // gmem
                auto dst_ptr = (VecType *)(tKrK(_, m, k).data());          // rf
                bool col_mask = Is_even_K || get<1>(tKcK(0, 0, k)) < params.d;
                if constexpr (Is_even_K) {
                    *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
                } else {
                    *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                                row_mask && col_mask, 1, MACA_ICMP_EQ);
                }
            }
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, false, true>(acc_s, acc_o, sRowMax, params.scale_softmax_log2);

        #pragma unroll
        for (int m = 0; m < size<1>(tKgK); ++m) {
            bool row_mask = true;
            #pragma unroll
            for (int k = size<2>(tKgK)/2; k < size<2>(tKgK); ++k) {
                auto src_ptr = (VecType *)(tKgK(_, m, k).data().get());    // gmem
                auto dst_ptr = (VecType *)(tKrK(_, m, k).data());          // rf
                bool col_mask = Is_even_K || get<1>(tKcK(0, 0, k)) < params.d;

                if constexpr (Is_even_K) {
                    *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
                } else {
                    *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                                row_mask && col_mask, 1, MACA_ICMP_EQ);
                }
            }
        }
        Tensor tOrP = make_tensor(make_rmem_ptr(reinterpret_cast<Element *>(acc_s.data())), acc_s.layout());
        if (wave_group_idx == 0) {
            // Convert acc_s from fp32 to fp16/bf16
            CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
            cute::copy(rP, tOrP);
            cute::copy(smem_tiled_copy_stsP, tOrP, tPsP);
        }
        flash::sync_threads();
        if (wave_group_idx == 1) {
            cute::copy(smem_tiled_copy_ldsP, tOsP, tOrP);
        }
        Tensor tVrV_view = make_tensor(tVrV.data(), make_layout(make_shape(size<0, 0>(tVrV), size<0, 1>(tVrV), size<1>(tVrV)*size<2>(tVrV))));
        permute_4x4_b16(tVrV_view);
        cute::copy(tVrV, tVsV);
        flash::sync_threads();
        // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 64 % kNWarps;
        int block_col_idx = n_block * (kBlockN / 64);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(tOrP);
            cute::copy(tOrP, rP_drop);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, kNWarps, 1, kBlockN>(
                rP_drop, block_row_idx, block_col_idx, n_block
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, kNWarps, 1, kBlockN>(tOrP, block_row_idx, block_col_idx, n_block);
        }
        // Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma_o, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, sRowSum, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_o);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { sync_threads(); }

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

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

}
