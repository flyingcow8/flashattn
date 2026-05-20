#include <cuda.h>

#include "flash_bwd_launch_template.h"
#include "flash_parameter.h"
#include "static_switch.h"
#include "flash_launch_parameter.h"

using namespace mcFlashAttn;

namespace Xcore1000 {

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    int AtomLayoutMSdP,
    int AtomLayoutNdKV,
    int AtomLayoutMdQ,
    bool Is_V_in_regs,
    bool Is_K_in_regs,
    bool No_double_buffer,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_bwd_template(mcFlashAttn::Flash_bwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream){
    using Kernel_traits = Flash_bwd_kernel_traits<kHeadDim, kBlockM, kBlockN, kNWarps, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, Is_V_in_regs, Is_K_in_regs, No_double_buffer, elem_type, kHeadDimV>;
    constexpr Arch arch = Arch::xcore1000;
    CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            if(!launch_params.performance_mode){
                flash_bwd_compute_launch_parameter<arch>(params,kHeadDim,kBlockM,kBlockN,launch_params);
            }
            run_flash_bwd<Kernel_traits,Is_dropout,Is_causal, arch>(params, launch_params, stream);
        });
    });
}

}


namespace Xcore1500 {

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    int AtomLayoutMSdP,
    int AtomLayoutNdKV,
    int AtomLayoutMdQ,
    bool Is_V_in_regs,
    bool Is_K_in_regs,
    bool No_double_buffer,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_bwd_template(mcFlashAttn::Flash_bwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream){
    using Kernel_traits = Flash_bwd_kernel_traits<kHeadDim, kBlockM, kBlockN, kNWarps, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, Is_V_in_regs, Is_K_in_regs, No_double_buffer, elem_type, kHeadDimV>;
    CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
    constexpr Arch arch = Arch::xcore1500;
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            if(!launch_params.performance_mode){
                flash_bwd_compute_launch_parameter<arch>(params,kHeadDim,kBlockM,kBlockN,launch_params);
            }
            run_flash_bwd<Kernel_traits,Is_dropout,Is_causal, arch>(params, launch_params, stream);
        });
    });
}

}
