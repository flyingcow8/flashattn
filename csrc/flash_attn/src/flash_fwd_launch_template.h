/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>

#include "flash_parameter.h"
#include "static_switch.h"
#include "flash_fwd_kernel.h"
#include "flash_fwd_split_kernel.h"

using namespace mcFlashAttn;

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask,
         bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Return_lse>
__global__ void flash_fwd_kernel(const Flash_fwd_params params) {
    static_assert(!(Is_causal && Is_local));
    flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, Is_even_MN, Is_even_K,
                        Return_softmax, Return_lse, Flash_fwd_params>(params);
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Split, bool Append_KV>
__global__ void flash_fwd_splitkv_kernel(const Flash_fwd_params params) {
    flash::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Split, Append_KV>(params);
}

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K>
__global__ void flash_fwd_splitkv_combine_kernel(const Flash_fwd_params params) {
    static_assert(Log_max_splits >= 1);
    flash::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h, params.b);

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 &&
                            params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    const bool return_lse = params.softmax_lse_ptr != nullptr;

    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        LOCAL_SWITCH_AND_CONST_PRECOND((!Is_causal), (params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
            BOOL_SWITCH_AND_CONST_PRECOND((Is_dropout), return_softmax, ReturnSoftmaxConst, [&] {
                BOOL_SWITCH_AND_CONST_PRECOND((IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128), is_even_MN, IsEvenMNConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        BOOL_SWITCH_AND_CONST_PRECOND((!Is_causal && !Is_local && !Has_alibi), params.has_attn_mask, Has_attn_mask, [&] {
                            BOOL_SWITCH_OR_CONST_PRECOND((Is_dropout), return_lse, ReturnLseConst, [&] {
                                auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst,
                                                                ReturnSoftmaxConst, ReturnLseConst>;
                                if (smem_size >= 32 * 1024) {
                                    CUDA_CHECK(cudaFuncSetAttribute(
                                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                                }
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                CUDA_KERNEL_LAUNCH_CHECK();
                            });
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    dim3 grid(params.num_splits > 1 ? params.b * params.h : params.h, num_m_block, params.num_splits > 1 ? params.num_splits : params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 &&
                            params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        LOCAL_SWITCH_AND_CONST_PRECOND((!Is_causal), (params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                    BOOL_SWITCH_AND_CONST_PRECOND((!Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128), is_even_MN, IsEvenMNConst, [&] {
                        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                            auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Split, Append_KV>;
                            if (smem_size >= 32 * 1024) {
                                CUDA_CHECK(cudaFuncSetAttribute(
                                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                            }
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
    if (params.num_splits > 1) {
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 8 : (Kernel_traits::kHeadDim % 64 == 0 ? 16 : 32);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        const int kNThreads = 256;
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, kNThreads, 0, stream>>>(params);
            }
            CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
}

namespace mcFlashAttn {

    template<typename T, int Headdim, bool Is_causal>
    void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
        if constexpr (Headdim == 128) {
            BOOL_SWITCH(params.num_splits > 1, Is_splits, [&] {
                if (params.seqlen_k <= 1024) {
                    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, true, true, T, Is_splits>, Is_causal>(params, stream);
                    return;
                }
                run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T, Is_splits>, Is_causal>(params, stream);
            });
            return;
        }

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        BOOL_SWITCH(params.num_splits > 1, Is_splits, [&] {
            run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, true, true, T, Is_splits>, Is_causal>(params, stream);
        });
    }

}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        #if 0
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        #if 0
        if constexpr(!Is_dropout) {
            if constexpr(!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        }
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    #if 0
    auto dprops = flash::mcGetCurrentDeviceProperties();
    bool is_sm8x = dprops.major == 8 && dprops.minor > 0;
    #endif
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        #if 0
        if (is_sm8x) {
            if constexpr(!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    #if 0
    auto dprops = flash::mcGetCurrentDeviceProperties();
    bool is_sm8x = dprops.major == 8 && dprops.minor > 0;
    #endif
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        #if 0
        if constexpr(!Is_dropout) {
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim160(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    #if 0
    auto dprops = flash::mcGetCurrentDeviceProperties();
    bool is_sm8x = dprops.major == 8 && dprops.minor > 0;
    #endif
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        #if 0
        if (is_sm8x) {
            if constexpr(!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        #if 0
        if constexpr(!Is_dropout) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim224(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 224;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      CUDA_CHECK(status_);
    }

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        #if 0
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64)) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      CUDA_CHECK(status_);
    }

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        #if 0
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        #endif
    });
}
