#pragma once

#include "decoder_masked_multihead_attention.h"
#include "decoder_masked_multihead_attention_utils.h"
#include "cuda_bf16_wrapper.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

#include "decoder_masked_multihead_attention_template.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

#define MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE, THDS_PER_BLOCK, DO_CROSS_ATTENTION, stream)    \
    size_t smem_sz = mmha::smem_size_in_bytes<T, DO_CROSS_ATTENTION>(params, THDS_PER_VALUE, THDS_PER_BLOCK);          \
    auto kernel = mmha::masked_multihead_attention_kernel<T, Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE,                 \
                                                          THDS_PER_BLOCK, DO_CROSS_ATTENTION>;                         \
    if (smem_sz >= 48 * 1024) {                                                                                        \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);                            \
    }                                                                                                                  \
    dim3 grid(params.nnz_head_idx == nullptr ? params.num_heads : params.nnz_heads, params.batch_size);                \
    kernel<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params)

template<
    typename T,
    int Dh,
    int Dh_MAX,
    int THDS_PER_KEY,
    int THDS_PER_VALUE,
    int THDS_PER_BLOCK,
    typename KERNEL_PARAMS_TYPE,
    bool DO_CROSS_ATTENTION,
    bool UNROLL = false>
void mmha_launch_kernel2(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream)
{
    size_t smem_sz = mmha::smem_size_in_bytes<T, DO_CROSS_ATTENTION>(params, THDS_PER_VALUE, THDS_PER_BLOCK);
    dim3 grid(params.nnz_head_idx == nullptr ? params.num_heads : params.nnz_heads, params.batch_size);
    if constexpr(UNROLL){
        auto kernel = mmha::masked_multihead_attention_kernel_unroll<T,Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE,THDS_PER_BLOCK, DO_CROSS_ATTENTION>;
        if (smem_sz >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);
        }
        kernel<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params);
    } else {
        auto kernel = mmha::masked_multihead_attention_kernel<T,Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE,THDS_PER_BLOCK, DO_CROSS_ATTENTION>;
        if (smem_sz >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);
        }
        kernel<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// !!! Specialize the launcher for Cross attention
template<typename T, int Dh, int Dh_MAX, typename KERNEL_PARAMS_TYPE>
void mmha_launch_kernel(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream)
{
    constexpr int  THREADS_PER_VALUE  = Dh_MAX * sizeof(T) / 16;
    constexpr bool DO_CROSS_ATTENTION = std::is_same<KERNEL_PARAMS_TYPE, Cross_multihead_attention_params<T>>::value;
    int            tlength            = (DO_CROSS_ATTENTION) ? params.memory_max_len : params.timestep;
    // printf("tlength, CROSS_ATTENTION = %d, %d\n", tlength, DO_CROSS_ATTENTION);
    if (tlength < 32) {
        // MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 4, THREADS_PER_VALUE, 64, DO_CROSS_ATTENTION, stream);
        constexpr int THREADS_PER_KEY = 4;
        constexpr int THREADS_PER_BLOCK = 64;
        mmha_launch_kernel2<T,Dh,Dh_MAX,THREADS_PER_KEY,THREADS_PER_VALUE,THREADS_PER_BLOCK,KERNEL_PARAMS_TYPE,DO_CROSS_ATTENTION>(params,stream);
    }
    else if (tlength < 2048) {
        // MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 2, THREADS_PER_VALUE, 128, DO_CROSS_ATTENTION, stream);
        constexpr int THREADS_PER_KEY = 2;
        constexpr int THREADS_PER_BLOCK = 128;
        mmha_launch_kernel2<T,Dh,Dh_MAX,THREADS_PER_KEY,THREADS_PER_VALUE,THREADS_PER_BLOCK,KERNEL_PARAMS_TYPE,DO_CROSS_ATTENTION>(params,stream);
    }
    else {
        // MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 2, THREADS_PER_VALUE, 512, DO_CROSS_ATTENTION, stream);
        constexpr int THREADS_PER_KEY = 2;
        constexpr int THREADS_PER_BLOCK = 512;
        mmha_launch_kernel2<T,Dh,Dh_MAX,THREADS_PER_KEY,THREADS_PER_VALUE,THREADS_PER_BLOCK,KERNEL_PARAMS_TYPE,DO_CROSS_ATTENTION,true>(params,stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#undef MMHA_LAUNCH_KERNEL

template<int hid1,int hid2,typename KERNEL_PARAMS_TYPE>
void run_multihead_attention_(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream);