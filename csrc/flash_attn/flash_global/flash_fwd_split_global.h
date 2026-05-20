#pragma once

#include "flash_parameter.h"
#include "arch.h"
#include "xcore1000/flash_fwd_split_kernel.h"
#include "xcore1500/flash_fwd_split_kernel.h"

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Is_page_attn, Arch arch = Arch::xcore1000>
__global__ void flash_fwd_splitkv_kernel(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type) {
    if constexpr (arch == Arch::xcore1000) {
        #if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1000)
        flash::xcore1000::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Is_page_attn>(params, num_m_block, block_type);
        #endif
    } else if constexpr (arch == Arch::xcore1500) {
        #if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1500)
        flash::xcore1500::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Is_page_attn>(params, num_m_block, block_type);
        #endif
    }
}
