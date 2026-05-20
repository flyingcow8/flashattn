#pragma once

#include <cuda.h>

#include "flash_parameter.h"
#include "static_switch.h"
#include "flash_fwd_split_combine_kernel.h"
#include "flash_performance_mode.h"
#include "print_parameter.h"
#include "flash_attn_global.h"

using namespace mcFlashAttn;

#define CHECK_MSG(x, ...) do { if((x) == false) {throw std::invalid_argument(__VA_ARGS__);} }while(0)
#define CUDA_CHECK(expr) {auto x = (expr); CHECK_MSG(x == cudaSuccess, #expr + std::string(" check failed!"));}
#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, bool Has_sink>
__global__ void flash_fwd_splitkv_combine_kernel(const Flash_fwd_params params) {
    static_assert(Log_max_splits >= 1);
    flash::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K, Has_sink>(params);
}

template<typename Kernel_traits, bool Is_causal, Arch arch>
void run_flash_splitkv_fwd(Flash_fwd_params &params, const Flash_launch_params &launch_params, cudaStream_t stream) {
    const int block_type = launch_params.block_type;

    //static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
    //static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    //dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    dim3 grid = flash_fwd_splitkv_compute_grid_dim(num_m_block, params.num_splits, params.h, params.b, block_type);

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        LOCAL_SWITCH((!Is_causal), (params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
            SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
                APPENDKV_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                    EVENMN_SWITCH((!Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128), is_even_MN, IsEvenMNConst, [&] {
                        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                            SOFTCAP_SWITCH(true, params.softcap > 0.0, Is_softcap, [&]{
                                PAGE_ATTN_SWITCH(params.block_table != nullptr, Is_page_attn, [&] {
                                    // number of templates reduce conditions :
                                    //      If Is_causal, set Is_local to false
                                    //      If Append_KV, set IsEvenMNConst to false
                                    //      If not IsEvenKConst, set IsEvenMNConst to false
                                    //      If Is_local, set IsEvenMNConst to false
                                    //      If head dim > 128, set IsEvenMNConst to false
                                    if (std::getenv("MHA_PRINT_PARA")) {
                                        shape_print(static_cast<Flash_bwd_params&>(params), launch_params, false, Is_causal, Is_local, Has_alibi, false, "kvcache");
                                    }
                                    if (std::getenv("MHA_DEBUG_PARA")){
                                        debug_print(static_cast<Flash_bwd_params&>(params), launch_params, "kvcache");
                                    }

                                    size_t smem_size = Kernel_traits::Share_Q_K_smem ? std::max((Split ? 2 : 1) * Kernel_traits::kSmemQSize, Kernel_traits::kSmemKVSize) : Kernel_traits::kSmemQSize + Kernel_traits::kSmemKVSize;

                                    if constexpr (Split) {
                                        // we don't want to use 64k shared memory size under the following one exceptional cases
                                        if constexpr(Kernel_traits::kHeadDim == 256){
                                            if (params.page_block_size != 1 && (params.page_block_size / 4) % 2 == 0) {
                                                smem_size = Kernel_traits::Share_Q_K_smem ? std::max(Kernel_traits::kSmemQSize, Kernel_traits::kSmemKVSize) : Kernel_traits::kSmemQSize + Kernel_traits::kSmemKVSize;
                                            }
                                        }
                                    }

                                    auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, arch>;
                                    if (smem_size >= 32 * 1024) {
                                        CUDA_CHECK(cudaFuncSetAttribute(
                                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                                    }
                                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params, num_m_block, block_type);
                                    // kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params, num_m_block, block_type);
                                    CUDA_KERNEL_LAUNCH_CHECK();
                                });
                            });
                        });
                    });
                });
            });
        });
    });
    if (params.num_splits > 1) {
        // With 256 threads we can load 1024 elements at a time, so if headdim is divisible by 256, kBlockM_min = 4.
        // If headdim is divisible by 128, then we set min kBlockM_min = 8, etc.
        constexpr static int kBlockM_min = Kernel_traits::kHeadDimV % 256 == 0 ? 4 : (Kernel_traits::kHeadDimV % 128 == 0 ? 8 : (Kernel_traits::kHeadDimV % 64 == 0 ? 16 : 32));
        // We have a heuristic selection strategy to choose blockM for different cases
        COMBINE_BLOCKM_SWITCH(params.b, params.h, params.seqlen_q, kBlockM_min, kBlockM, [&] {
            dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
            constexpr int kNThreads = 256; /*Kernel_traits::kNThreads;*/
            EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
                SINK_SWITCH(params.s_aux_ptr != nullptr, Has_sink, [&] {
                    if (params.num_splits <= 2) {
                        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst, Has_sink><<<grid_combine, kNThreads, 0, stream>>>(params);
                    } else if (params.num_splits <= 4) {
                        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst, Has_sink><<<grid_combine, kNThreads, 0, stream>>>(params);
                    } else if (params.num_splits <= 8) {
                        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst, Has_sink><<<grid_combine, kNThreads, 0, stream>>>(params);
                    } else if (params.num_splits <= 16) {
                        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst, Has_sink><<<grid_combine, kNThreads, 0, stream>>>(params);
                    } else if (params.num_splits <= 32) {
                        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst, Has_sink><<<grid_combine, kNThreads, 0, stream>>>(params);
                    } else if (params.num_splits <= 64) {
                        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst, Has_sink><<<grid_combine, kNThreads, 0, stream>>>(params);
                    } else if (params.num_splits <= 128) {
                        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst, Has_sink><<<grid_combine, kNThreads, 0, stream>>>(params);
                    }
                    CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    }
}
