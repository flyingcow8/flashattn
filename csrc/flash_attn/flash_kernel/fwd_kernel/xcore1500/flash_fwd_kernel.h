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

#include "flash_fwd_kernel_k32_xcore1500.h"
#include "flash_fwd_kernel_k64_xcore1500.h"


namespace flash {

namespace xcore1500 {

using namespace cute;

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

    if constexpr (Kernel_traits::kHeadDim == 32 || Kernel_traits::kHeadDim == 96 || Kernel_traits::kHeadDim == 160){
        compute_attn_1rowblock_k32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, bidh, m_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            compute_attn_1rowblock_k32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params, Merge_attn_mask_ldg>(params, bidb, bidh, m_block_max - m_block - 1);
        }
        return;
    } else if constexpr (Kernel_traits::kHeadDim == 64 || Kernel_traits::kHeadDim == 128 || Kernel_traits::kHeadDim == 192 || Kernel_traits::kHeadDim == 256) {
        const int curr_block = m_block_max - m_block - 1;
        compute_attn_1rowblock_k64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, bidh, curr_block);
        if constexpr (Rowblock_Parallel_Num == 2) {
            compute_attn_1rowblock_k64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Params>(params, bidb, params.h - bidh - 1, m_block);
        }
        return;
    }
}

} // namespace xcore1500

} // namespace flash
