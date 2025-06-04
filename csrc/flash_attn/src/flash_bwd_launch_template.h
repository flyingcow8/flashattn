/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>

#include "flash_parameter.h"
#include "static_switch.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_kernel.h"

#include "flash_bwd_preprocess_kernel_hdim128_32_32.h"
#include "flash_bwd_kernel_hdim128_32_32.h"

using flash::FlashBwdKernels;
using namespace mcFlashAttn;

template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
    const bool is_32x32 = (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 32);
    if constexpr (is_32x32) {
        flash::compute_dot_do_o_hdim128_32_32<Clear_dQaccum, Kernel_traits>(params);
    } else {
        flash::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
    }
}

template<typename Kernel_traits>
__global__ void flash_bwd_clear_dkvaccum_kernel(const Flash_bwd_params params) {
    flash::clear_dKVaccum<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Has_attn_mask, bool Is_even_M, bool Is_even_K>
__global__ void flash_bwd_dq_dk_dv_loop_kernel(const Flash_bwd_params params) {
    flash::compute_dq_dk_dv<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Has_attn_mask, Is_even_M, Is_even_K>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN,
        bool Is_even_K, bool Is_deterministic = false>
__global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel(const Flash_bwd_params params) {
    static_assert(!(Is_causal && Is_local));

    const bool is_32x32 = (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 32);
    if constexpr (is_32x32) {
        flash::compute_dq_dk_dv_seqk_parallel_hdim128_32_32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                                            Is_even_MN, Is_even_K, Is_deterministic>(params);
    } else {
        flash::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN,
                                              Is_even_K,Is_deterministic>(params);
    }
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(const Flash_bwd_params params, const int nsplits) {
    const bool is_32x32 = (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 32);
    if constexpr (is_32x32) {
        flash::convert_dQ_hdim128_32_32<Kernel_traits>(params, nsplits);
    } else {
        flash::convert_dQ<Kernel_traits>(params, nsplits);
    }
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_kernel(const Flash_bwd_params params) {
    flash::convert_dKV<Kernel_traits>(params);
}

__inline__ FlashBwdKernels flash_bwd_kernel_select_hdim128(const Flash_bwd_params &params) {
    if ((params.seqlen_k == 4096 && params.b == 2 && params.h_k == 56 && params.is_causal && !params.deterministic) ||
        ((params.seqlen_k == 8192 || params.seqlen_k == 4096) && params.b == 1 && params.is_causal && params.deterministic)) {
        return FlashBwdKernels::kBwdKernel_hdim128_32x128_512;
    } else {
        return FlashBwdKernels::kBwdKernel_hdim128_32x64_256;
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd_seqk_parallel(Flash_bwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.h, params.b);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = num_n_block;
    if (params.deterministic) {
        auto dprops = flash::mcGetCurrentDeviceProperties();
        gridDimx = (dprops.multiProcessorCount + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_n(gridDimx, params.h, params.b);

    const int num_threads = (Kernel_traits::kBlockM == 32 && Kernel_traits::kNThreads == 512) ? 256 : Kernel_traits::kNThreads;
    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, num_threads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, num_threads, 0, stream>>>(params);
    }
    CUDA_KERNEL_LAUNCH_CHECK();

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                            params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        LOCAL_SWITCH_AND_CONST_PRECOND((!Is_causal), (params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
            BOOL_SWITCH_AND_CONST_PRECOND((IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128), is_even_MN, IsEvenMNConst, [&] {
                ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                    BOOL_SWITCH_AND_CONST_PRECOND((!Is_causal && !Is_local && !Has_alibi), params.has_attn_mask, Has_attn_mask, [&] {
                        BOOL_SWITCH(params.deterministic, Is_deterministic, [&] {
                            auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask,
                                                                                        IsEvenMNConst, IsEvenKConst, Is_deterministic>;
                            if (smem_size_dq_dk_dv >= 48 * 1024)  {
                                CUDA_CHECK(cudaFuncSetAttribute(
                                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                            }
                            kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                            CUDA_KERNEL_LAUNCH_CHECK();
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

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    run_flash_bwd_seqk_parallel<Kernel_traits, Is_dropout, Is_causal>(params, stream);
#endif
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim32(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      CUDA_CHECK(status_);
    }

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 4, 4, 1, 4, true, true, true, T>, Is_dropout, Is_causal>(params, stream);
        #if 0
        if (max_smem_per_block >= 144 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream);
        } else {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream);
        }
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      CUDA_CHECK(status_);
    }

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      CUDA_CHECK(status_);
    }

#ifdef __USE_128_32x32
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, T>, Is_dropout, Is_causal>(params, stream);
    });
#else

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        switch (flash_bwd_kernel_select_hdim128(params)) {
            case FlashBwdKernels::kBwdKernel_hdim128_32x128_512:
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 128, 8, 2, 4, 2, true, true, true, T>, Is_dropout, Is_causal>(params, stream);
                break;
            case FlashBwdKernels::kBwdKernel_hdim128_32x64_256:
            default:
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 64, 4, 2, 2, 2, true, false, true, T>, Is_dropout, Is_causal>(params, stream);
                break;
        }
    });
#endif
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim160(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim192(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim224(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 224;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}
