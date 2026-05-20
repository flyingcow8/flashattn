// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_parameter.h"
#include "flash_fwd_global.h"
#include "kernel_traits.h"
#include <mctlass/numeric_types.h>


template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, true, false, false, true, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, true, false, false, true, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, true, false, true, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, true, false, true, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, true, false, false, false, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, true, false, false, false, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, true, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, true, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, true, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, true, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, false, false, false, true, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, false, false, false, true, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, false, false, true, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, false, false, true, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, false, false, false, false, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, false, false, false, false, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, false, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, false, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, false, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, true, false, false, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, true, false, false, true, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, true, false, false, true, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, true, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, true, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, true, false, false, false, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, true, false, false, false, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, true, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, true, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, false, false, false, true, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, false, false, false, true, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, false, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, false, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, false, false, false, false, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, false, false, false, false, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, false, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, true, false, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, true, false, false, true, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, true, false, false, true, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, true, false, true, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, true, false, true, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, true, false, false, false, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, true, false, false, false, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, true, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, true, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, true, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, true, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, true, false, true, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, true, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, true, false, true, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, true, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, true, true, false, false, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, true, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, true, true, false, false, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, true, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, false, false, true, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, false, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, false, false, true, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, false, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, false, false, false, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, false, false, false, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, true, false, false, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, true, false, false, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, true, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, false, false, true, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, false, false, true, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, false, true, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, false, true, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, false, false, false, false, true, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, false, false, false, false, true, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    64,
                    128,
                    64,
                    4,
                    true,
                    true,
                    mctlass::half_t,
                    64>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                true, false, false, false, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);

