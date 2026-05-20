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
namespace xcore1000 {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax,  typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_opt_hdim128_blockM128(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    bool return_lse = params.softmax_lse_ptr != nullptr;

    // Shared memory.
    extern __shared__ char smem_[];
    uint32_t tQrQ[int(Kernel_traits::kRegSize)];
    uint32_t tVrV[int(Kernel_traits::kRegSize / 2)];
    uint32_t tKrK[int(Kernel_traits::kRegSize / 2)];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    static_assert(Kernel_traits::Is_Q_in_regs == true);
    static_assert(Kernel_traits::Share_Q_K_smem == true);
    static_assert(kBlockN == 32 || kBlockN == 64 || kBlockN == 128);
    static_assert(kNWarps == 4);

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
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        flash::copy_zero_to_global<Is_even_MN, Is_even_K>(
            tOgO, tOcO, params.d, binfo.actual_seqlen_q - kBlockM_stride
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

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + kBlockM_stride * params.q_row_stride + bidh * params.q_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)

    auto d = params.d;
    int n_block = n_block_max - 1;
    const int base_row_offset = n_block * kBlockN;

    //
    // Copy Atom retiling
    //
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    //
    // PREDICATES
    //
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(
                tQgQ, tQrQ, tQcQ, d, binfo.actual_seqlen_q - kBlockM_stride);

    // We move K and V to the last block.
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + base_row_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + base_row_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + kBlockM_stride) * params.seqlen_k_rounded + base_row_offset;

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    flash::copy_reg_to_share(tQrQ, tQsQ);

    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(
                tKgK, tKrK, tKVcKV, d, binfo.actual_seqlen_k - n_block * kBlockN);

    const uint32_t laneId = __lane_id();
    const int cpy_offset = ((laneId & 0xf) << 2) - (laneId & 0xf);
    const uint32_t tOsVt_stride = get<1>(get<1>(tOsVt(_, _, _0{}).layout().stride()));
    const uint32_t tOrVt_stride = get<1>(get<1>(tOrVt(_, _, _0{}).layout().stride()));

    constexpr int ldg_Num = (kBlockN * kHeadDim / Kernel_traits::kNThreads) / 8;
    constexpr int lds_Tuple = kHeadDim / 64;
    constexpr bool Is_perm_4x4 = (kBlockN == 64);

    int tVgV_offset[ldg_Num];
    uint32_t *tVsV_ptr[ldg_Num];
    int tVcV[ldg_Num + (!Is_even_K * ldg_Num)]; //pred for V
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
        const int gv_row = kBlockN == 32 ? ((laneId >> 4) << 1) + i:
                                        (((laneId >> 4) & 0x1) << 2) +
                                            ((laneId >> 5) << 5) + (i & 0x3) + ((i >> 2) << 6);
        const int gv_col = (laneId & 0xf) << 3;
        const int old_gv_row = laneId >> 3;
        const int old_gv_col = (laneId & 0x7) << 3;
        tVgV_offset[i] = (gv_row - old_gv_row) * params.v_row_stride + (gv_col - old_gv_col);

        const int sv_row = gv_row + ((old_gv_row & 0x1) * kBlockN);
        const int& old_sv_row = old_gv_row;
        tVsV_ptr[i] = reinterpret_cast<uint32_t *>(tVsV(_, _0{}, _0{}).data().ptr_ + ((sv_row - old_sv_row) << 6));
        tVcV[i] = gv_row + ((tidx >> 6) << 3);
        if (!Is_even_K) tVcV[ldg_Num + i] = gv_col;
    }

    {
        const int col = (((laneId & 0x7) ^ (laneId >> 5)) << 3) + (((laneId >> 4) & 0x1) << 2);
        flash::sync_threads();
        auto sm_ptr = reinterpret_cast<uint32_t *>(tSsQ.data().ptr_ - col);
        auto r_ptr = reinterpret_cast<uint32_t *>(tSrQ.data());
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = (((laneId & 0x7) ^ ((laneId >> 5) + (i << 1))) << 2) + (((laneId >> 4) & 0x1) << 1);
            auto ptr0 = reinterpret_cast<uint64_t *>(r_ptr + (i << 2));
            auto s_ptr0 = reinterpret_cast<uint64_t *>(sm_ptr + col);
            ptr0[0] = s_ptr0[0];

            ptr0 = reinterpret_cast<uint64_t *>(r_ptr + (i << 2) + 2);
            s_ptr0 = reinterpret_cast<uint64_t *>(sm_ptr + col + 64 * 32);
            ptr0[0] = s_ptr0[0];

            auto ptr1 = reinterpret_cast<uint64_t *>(r_ptr + (i << 2) + 16);
            auto s_ptr1 = reinterpret_cast<uint64_t *>(sm_ptr + col + 128 * 32);
            ptr1[0] = s_ptr1[0];

            ptr1 = reinterpret_cast<uint64_t *>(r_ptr + (i << 2) + 16 + 2);
            s_ptr1 = reinterpret_cast<uint64_t *>(sm_ptr + col + 192 * 32);
            ptr1[0] = s_ptr1[0];
        }
    }

    clear(acc_o);

    flash::copy_global_to_reg_V<Is_even_MN, Is_even_K,/*Is_ldg_B128=*/true, ldg_Num>(
                tVgV, tVrV, tVcV, tVgV_offset, d, binfo.actual_seqlen_k - n_block * kBlockN);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    flash::sync_threads();

    const int gK_offset = -int(kBlockN * params.k_row_stride);
    const int gV_offset = -int(kBlockN * params.v_row_stride);
    const auto mask_offset = kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf);
    const auto kWarps_offset = kNWarps * 16;
    const uint32_t perm_mask[2] = {0x05040100, 0x07060302};

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
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        // Advance gV
        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + gV_offset;
            flash::copy_global_to_reg_V</*Is_even_MN=*/true, Is_even_K, /*Is_ldg_B128=*/true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        }
        flash::copy_reg_to_share(tKrK, tKsK);
        clear(acc_s);
        flash::sync_threads();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

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

        mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset, params.custom_alibi);

        flash::copy_reg_to_share_V</*Is_sts_B128=*/true, ldg_Num>(tVrV, tVsV_ptr, perm_mask);
        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + gK_offset;
            flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2);
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

        flash::gemm_rs<Is_perm_4x4, lds_Tuple>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    #pragma unroll
    for (; n_block > n_block_min; --n_block) {
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        // Advance gV
        tVgV.data() = tVgV.data() + gV_offset;
        flash::copy_global_to_reg_V</*Is_even_MN=*/true,Is_even_K,/*Is_ldg_B128=*/true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        flash::sync_threads();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

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

        mask.template apply_mask</*Causal_mask=*/false>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset, params.custom_alibi);

        flash::copy_reg_to_share_V</*Is_sts_B128=*/true, ldg_Num>(tVrV, tVsV_ptr,perm_mask);
        // Advance gK
        tKgK.data() = tKgK.data() + gK_offset;
        flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKVcKV, d);

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, /*Syncthreads=*/true, true>(acc_s, acc_o, params.scale_softmax_log2);

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

        flash::gemm_rs<Is_perm_4x4, lds_Tuple>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    }

    if (n_block == n_block_min) {
        clear(acc_s);
        flash::copy_reg_to_share(tKrK, tKsK);
        // Advance gV
        tVgV.data() = tVgV.data() + gV_offset;
        flash::copy_global_to_reg_V</*Is_even_MN=*/true,Is_even_K,/*Is_ldg_B128=*/true, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        flash::sync_threads();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

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

        mask.template apply_mask</*Causal_mask=*/false>(acc_s, n_block * kBlockN, mask_offset, kWarps_offset, params.custom_alibi);

        flash::copy_reg_to_share_V</*Is_sts_B128=*/true, ldg_Num>(tVrV, tVsV_ptr,perm_mask);
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local || Has_attn_mask, true, true>(acc_s, acc_o, params.scale_softmax_log2);

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

        flash::gemm_rs<Is_perm_4x4, lds_Tuple>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
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
    //Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    //We don't need arrive here.
    //flash::sync_threads();
    flash::barrier();

    //cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    const int sm_col = ((((laneId & 0x7) ^ (laneId >> 5)) << 1) + ((laneId >> 4) & 0x1)) << 2;
    auto s_ptr = taccOsO.data().ptr_ - sm_col;

    #pragma unroll 2
    for (int i = 0; i < 2; ++i) {
        auto ptr = reinterpret_cast<uint32_t *>(rO.data().ptr_ + (i << 2));
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

        ptr = reinterpret_cast<uint32_t *>(rO.data().ptr_ + 32 + (i << 2));
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
            auto ptr0 = reinterpret_cast<uint64_t *>(rO.data().ptr_ + (j << 3) + (i << 2));
            auto ptr1 = reinterpret_cast<uint64_t *>(rO.data().ptr_ + (j << 3) + 32 + (i << 2));
            auto sm_ptr0 = reinterpret_cast<uint64_t *>(s_ptr + col + (i << 12));
            auto sm_ptr1 = reinterpret_cast<uint64_t *>(s_ptr + col + 128 * 64 + (i << 12));
            sm_ptr0[0] = ptr0[0];
            sm_ptr1[0] = ptr1[0];
        }
    }


    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
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
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }
    }


    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    flash::copy_reg_to_global<Is_even_MN, Is_even_K>(
        tOrO, tOgO, tOcO, d, binfo.actual_seqlen_q - kBlockM_stride);
}

} // xcore1000

} // namespace flash
