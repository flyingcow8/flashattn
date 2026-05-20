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

#include "flash_fwd_split_kernel_k32_xcore1500.h"
#include "flash_fwd_split_kernel_k64_xcore1500.h"

namespace flash {

namespace xcore1500 {

using namespace cute;

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Is_page_attn, typename Params>
__forceinline__ __device__ void compute_attn_splitkv(const Params &params, const int m_block_max, const int& block_type) {

    int n_split_idx = 0;
    const dim3 bidInf = get_bidInfo<Split>(block_type, params.h, n_split_idx);
    const int m_block = bidInf.x;
    // The block index for the batch.
    const int bidb = bidInf.y;
    // The block index for the head.
    const int bidh = bidInf.z;
    const int num_n_splits = Split ? params.num_splits : 1;

    if constexpr (Kernel_traits::kHeadDim == 128 || Kernel_traits::kHeadDim == 64 || Kernel_traits::kHeadDim == 192 || Kernel_traits::kHeadDim == 256){
        if constexpr (Kernel_traits::kHeadDim == 128 && Kernel_traits::kBlockN == 32) {
            flash::xcore1500::compute_attn_1rowblock_splitkv_k64_decode<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN,
                                                            Is_even_K, Is_softcap, Split, Append_KV, Is_page_attn>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
        } else {
            flash::xcore1500::compute_attn_1rowblock_splitkv_k64<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN,
                                                        Is_even_K, Is_softcap, Split, Append_KV, Is_page_attn>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
        }
    } else if constexpr (Kernel_traits::kHeadDim == 32 || Kernel_traits::kHeadDim == 96 || Kernel_traits::kHeadDim == 160){
        compute_attn_1rowblock_splitkv_k32<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Is_page_attn>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
    }
}

} // namespace xcore1500

} // namespace flash
