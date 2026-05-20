// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_parameter.h"
#include "flash_fwd_split_global.h"
#include "kernel_traits.h"
#include <mctlass/numeric_types.h>


template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, true, true, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, true, true, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, true, true, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, true, true, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, true, false, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, true, false, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, false, true, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, false, true, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, false, true, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, false, true, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, false, false, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, false, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, false, false, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, true, false, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, true, true, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, true, true, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, true, true, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, true, true, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, true, false, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, true, false, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, false, true, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, false, true, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, false, true, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, false, true, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, false, false, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, false, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, false, false, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, true, false, false, false, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, true, true, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, true, true, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, true, true, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, true, true, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, true, false, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, true, false, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, false, true, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, false, true, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, false, true, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, false, true, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, false, false, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, false, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, false, false, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, true, false, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, true, true, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, true, true, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, true, true, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, true, true, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, true, false, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, true, false, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, false, true, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, false, true, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, false, true, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, false, true, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, false, false, true, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, false, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, false, false, false, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    128,
                    16,
                    16,
                    1,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                false, true, false, false, false, false, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);

