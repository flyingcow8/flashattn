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

#include "flash_bwd_kernel_hdim64_64x64_4waves_xcore1500.h"
#include "flash_bwd_kernel_hdim96_32x64_4waves_xcore1500.h"
#include "flash_bwd_kernel_hdim128_32x64_4waves_xcore1500.h"

#include "flash_bwd_kernel_k32_blockN128_xcore1500.h"
#include "flash_bwd_kernel_k32_xcore1500.h"
#include "flash_bwd_kernel_k64_xcore1500.h"


namespace flash {

namespace xcore1500 {

using namespace cute;


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_deterministic, bool Is_balance,  typename Params>
inline __device__ void compute_dq_dk_dv_seqk_parallel(const Params &params, int gridtype) {

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    // The block index for the batch.
    int bidb = blockIdx.z;
    // The block index for the head.
    int bidh = blockIdx.y;
    // The block index for nblock.
    int n_block = blockIdx.x;

    if constexpr (Is_deterministic) {
        for (n_block = blockIdx.x; n_block < (params.seqlen_k + kBlockN - 1) / kBlockN; n_block += gridDim.x) {
            if constexpr (kHeadDim == 64) {
                compute_dq_dk_dv_1colblock_hdim64_64x64_4waves<Kernel_traits, Is_dropout, Is_causal, Is_local,
                                                               Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K,
                                                               Is_softcap, false, false, /*Seq_parallel=*/true>(
                    params, bidb, bidh, n_block);
            } else if constexpr (kHeadDim == 128) {
                compute_dq_dk_dv_1colblock_hdim128_32x64_xcore1500<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                         Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, false, false,
                                                         true>(params, bidb, bidh, n_block);
            } else if constexpr (kHeadDim == 32) {
                compute_dq_dk_dv_1colblock_k32_blockN128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                         Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, false, false,
                                                         /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
            } else if constexpr (kHeadDim == 96) {
                compute_dq_dk_dv_1colblock_hdim96_32x64_4waves<Kernel_traits, Is_dropout, Is_causal, Is_local,
                                                               Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K,
                                                               Is_softcap, false, false, true>(params, bidb, bidh,
                                                                                               n_block);
            } else if constexpr (kHeadDim == 160) {
                bidb = blockIdx.z;
                bidh = blockIdx.y;
                compute_dq_dk_dv_1colblock_k32_32x32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                            Is_even_MN, Is_even_K, Is_softcap, false, false, /*Seq_parallel=*/true>(
                                            params, bidb, bidh, n_block);
            } else if constexpr (kHeadDim == 192 || kHeadDim == 256) {
                compute_dq_dk_dv_1colblock_k64_xcore1500<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                                Is_even_MN, Is_even_K, Is_softcap, false, false, /*Seq_parallel=*/true>(
                                                params, bidb, bidh, n_block);
            }
        }
    } else {
        UNPACK_GRID(n_block, bidh, bidb, gridtype);
        const int n_block_max = (params.seqlen_k + kBlockN - 1) / kBlockN;
        if constexpr (kHeadDim == 64) {
            compute_dq_dk_dv_1colblock_hdim64_64x64_4waves<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                           Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, false,
                                                           false, /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
        } else if constexpr (kHeadDim == 128) {
            compute_dq_dk_dv_1colblock_hdim128_32x64_xcore1500<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                    Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, false, false,
                                                    true>(params, bidb, bidh, n_block);
            if constexpr (Is_balance) {
                compute_dq_dk_dv_1colblock_hdim128_32x64_xcore1500<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                        Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, false, false,
                                                        /*Seq_parallel=*/true>(params, bidb, bidh, n_block_max - n_block - 1);
            }
        } else if constexpr (kHeadDim == 32) {
            compute_dq_dk_dv_1colblock_k32_blockN128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                     Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, false, false,
                                                     /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
            if constexpr (Is_balance) {
                compute_dq_dk_dv_1colblock_k32_blockN128<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                         Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, false, false,
                                                         /*Seq_parallel=*/true>(params, bidb, bidh,
                                                                                n_block_max - n_block - 1);
            }
        } else if constexpr (kHeadDim == 96) {
            compute_dq_dk_dv_1colblock_hdim96_32x64_4waves<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                           Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, false,
                                                           false, true>(params, bidb, bidh, n_block);
            if constexpr (Is_balance) {
                compute_dq_dk_dv_1colblock_hdim96_32x64_4waves<Kernel_traits, Is_dropout, Is_causal, Is_local,
                                                               Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K,
                                                               Is_softcap, false, false, true>(
                    params, bidb, bidh, n_block_max - n_block - 1);
            }
        } else if constexpr (kHeadDim == 160) {
            bidb = blockIdx.z;
            bidh = blockIdx.y;
            compute_dq_dk_dv_1colblock_k32_32x32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                                 Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, false, false,
                                                 /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
        } else if constexpr (kHeadDim == 192 || kHeadDim == 256) {
            compute_dq_dk_dv_1colblock_k64_xcore1500<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                            Is_even_MN, Is_even_K, Is_softcap, false, false, /*Seq_parallel=*/true>(
                                            params, bidb, bidh, n_block);
            if constexpr (Is_balance) {
                compute_dq_dk_dv_1colblock_k64_xcore1500<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                            Is_even_MN, Is_even_K, Is_softcap, false, false, true>(
                                            params, bidb, bidh, n_block_max - n_block - 1);
            }
        }
    }
}

} // namespace xcore1500

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace flash
