#pragma once

#include "flash_parameter.h"
#include "arch.h"
#include "xcore1000/flash_bwd_kernel.h"
#include "xcore1500/flash_bwd_kernel.h"

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_deterministic=false, bool Is_balance=false, Arch arch = Arch::xcore1000>
__global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel(mcFlashAttn::Flash_bwd_params params, int gridtype) {
    static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
    if constexpr (arch == Arch::xcore1000) {
        #if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1000)
            flash::xcore1000::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Is_deterministic, Is_balance>(params, gridtype);
        #endif
    } else if constexpr (arch == Arch::xcore1500) {
        #if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1500)
        flash::xcore1500::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K, Is_softcap, Is_deterministic, Is_balance>(params, gridtype);
        #endif
    }
}
