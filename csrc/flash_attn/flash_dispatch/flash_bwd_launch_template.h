/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <set>

#include "flash_parameter.h"
#include "static_switch.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_performance_mode.h"
#include "print_parameter.h"
#include "utils.h"
#include "host_utils.h"
#include "flash_attn_global.h"

using namespace mcFlashAttn;

template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
    flash::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_clear_dkvaccum_kernel(const Flash_bwd_params params) {
    flash::clear_dKVaccum<Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(const Flash_bwd_params params, const int nsplits) {
    flash::convert_dQ<Kernel_traits>(params, nsplits);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_kernel(const Flash_bwd_params params) {
    flash::convert_dKV<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, Arch arch>
void run_flash_bwd_seqk_parallel(Flash_bwd_params &params, const Flash_launch_params &launch_params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.h, params.b);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;

    bool is_balance = launch_params.is_balance;
    if(num_n_block % 2 != 0) is_balance = false;
    int gridDimx = is_balance ? (num_n_block / 2) : num_n_block;
    if (params.deterministic) {
        auto dprops = flash::mcGetCurrentDeviceProperties();
        gridDimx = (dprops.multiProcessorCount + params.b * params.h - 1) / (params.b * params.h);
    }

    dim3 grid_n = flash_bwd_compute_grid_dim(gridDimx, params.h, params.b, launch_params.block_type);

    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    CUDA_KERNEL_LAUNCH_CHECK();

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr bool use_k64_kernel = Kernel_traits::kHeadDim == 256 ||
                                    Kernel_traits::kHeadDim == 192 ||
                                    (Kernel_traits::kHeadDim == 64 && Kernel_traits::kBlockN == 128) ||
                                    Kernel_traits::kHeadDim == 32;
    constexpr int smem_size_dq_dk_dv = use_k64_kernel ? Kernel_traits::VarHeadDim::kSmemSize1colblock : Kernel_traits::kSmemSize1colblock;


    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        LOCAL_SWITCH((!Is_causal), (params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
            EVENMN_SWITCH((IsEvenKConst && !Is_local), is_even_MN, IsEvenMNConst, [&] {
                ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                    ATTN_MASK_SWITCH((!Is_causal && !Is_local && !Has_alibi), params.has_attn_mask, Has_attn_mask, [&] {
                        DETERMINISTIC_SWITCH(params.deterministic, Is_deterministic, [&] {
                            BALANCE_SWITCH((!Is_deterministic && Is_causal), is_balance, IsBalance, [&] {
                                SOFTCAP_SWITCH((!Is_dropout), params.softcap > 0.0, Is_softcap, [&] {
                                    // number of templates reduce conditions :
                                    //      If Is_causal, set Is_local to false
                                    //      If not IsEvenKConst, we also set IsEvenMNConst to false
                                    //      If Is_local, set IsEvenMNConst to false
                                    //      If not Is_causal, set IsBalance to false
                                    //      If Is_deterministic, set IsBalance to false
                                    //      If Is_causal, set Has_attn_mask to false
                                    //      If Is_local, set Has_attn_mask to false
                                    //      If Has_alibi, set Has_attn_mask to false
                                    //      If Is_dropout, set Is_softcap to false
                                    if (std::getenv("MHA_PRINT_PARA")) {
                                        shape_print(params, launch_params, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, "bwd");
                                    }
                                    if (std::getenv("MHA_DEBUG_PARA")){
                                        std::vector<int> extra_debug{static_cast<int>(grid_m.x), static_cast<int>(grid_m.y), static_cast<int>(grid_m.z),
                                                                    static_cast<int>(grid_n.x), static_cast<int>(grid_n.y), static_cast<int>(grid_n.z),
                                                                    Kernel_traits::kNThreads, smem_size_dq_dk_dv};
                                        debug_print(params, launch_params, "bwd", &extra_debug);
                                    }
                                    auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits,
                                            Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, arch>;

                                    if (smem_size_dq_dk_dv >= 48 * 1024)  {
                                        CUDA_CHECK(cudaFuncSetAttribute(
                                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                                    }
                                    kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params, launch_params.block_type);
                                    CUDA_KERNEL_LAUNCH_CHECK();
                                });
                            });
                        });
                    });
                });
            });
        });
    });

    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 32 * 1024)  {
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    CUDA_KERNEL_LAUNCH_CHECK();

}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, Arch arch>
void run_flash_bwd(Flash_bwd_params &params, const Flash_launch_params &launch_params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    run_flash_bwd_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, arch>(params, launch_params, stream);
#endif
}
