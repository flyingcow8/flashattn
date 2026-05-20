#pragma once

#include <cuda.h>

#include "flash_fwd_split_launch_template.h"
#include "flash_parameter.h"
#include "static_switch.h"
#include "flash_launch_parameter.h"
#include "kernel_traits.h"

namespace Xcore1000 {

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_splitkv_fwd_template(mcFlashAttn::Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream){
    using Kernel_traits = Flash_fwd_kernel_traits<kHeadDim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type,kHeadDimV>;
    constexpr Arch arch = Arch::xcore1000;
    CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
        if(!launch_params.performance_mode){
            flash_fwd_splitkv_compute_launch_parameter(params,kHeadDim,kBlockM,kBlockN,launch_params);
        }
        run_flash_splitkv_fwd<Kernel_traits, Is_causal, arch>(params, launch_params, stream);
    });
}

}

namespace Xcore1500 {

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_splitkv_fwd_template(mcFlashAttn::Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream){
    using Kernel_traits = Flash_fwd_kernel_traits<kHeadDim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type,kHeadDimV>;
    constexpr Arch arch = Arch::xcore1500;
    CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
        if(!launch_params.performance_mode){
            flash_fwd_splitkv_compute_launch_parameter(params,kHeadDim,kBlockM,kBlockN,launch_params);
        }
        run_flash_splitkv_fwd<Kernel_traits, Is_causal, arch>(params, launch_params, stream);
    });
}

}
