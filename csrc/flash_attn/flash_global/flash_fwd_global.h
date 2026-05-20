#pragma once

#include "flash_parameter.h"
#include "arch.h"
#include "xcore1000/flash_fwd_kernel.h"
#include "xcore1500/flash_fwd_kernel.h"

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, int Rowblock_Parallel_Num = 1, bool Merge_attn_mask_ldg = false, Arch arch = Arch::xcore1000>
__global__ void flash_fwd_kernel(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type) {
    static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
    static_assert(Rowblock_Parallel_Num == 1 || Rowblock_Parallel_Num == 2);
    if constexpr (arch == Arch::xcore1000) {
        #if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1000)
        flash::xcore1000::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, mcFlashAttn::Flash_fwd_params, Rowblock_Parallel_Num, Merge_attn_mask_ldg>(params, num_m_block, block_type);
        #endif
    } else if constexpr (arch == Arch::xcore1500) {
        #if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1500)
        flash::xcore1500::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, mcFlashAttn::Flash_fwd_params, Rowblock_Parallel_Num, Merge_attn_mask_ldg>(params, num_m_block, block_type);
        #endif
    }
}
