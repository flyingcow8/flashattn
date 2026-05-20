// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_parameter.h"
#include "flash_fwd_global.h"
#include "kernel_traits.h"
#include <mctlass/numeric_types.h>


template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, true, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, true, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, true, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, true, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, false, false, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, false, false, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, false, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, false, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, true, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, true, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, true, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, true, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, true, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, false, false, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, false, false, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, false, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, false, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, true, false, false, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, true, false, false, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, true, false, false, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, true, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, true, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, true, false, false, false, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, true, false, false, false, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, true, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, true, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, false, false, false, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, false, false, false, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, false, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, false, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, false, false, false, false, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, false, false, false, false, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, false, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, true, false, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, true, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, true, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, true, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, true, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, false, false, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, false, false, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, false, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, false, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, true, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, true, true, true, false, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, true, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, true, true, true, false, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, true, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, true, true, false, false, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, true, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, true, true, false, false, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, true, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, false, true, false, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, false, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, false, true, false, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, false, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, false, false, false, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, false, false, false, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, true, true, false, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, true, true, false, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, true, false, false, 2, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, true, false, false, 1, true, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, true, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, true, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, true, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, true, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, true, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, false, false, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, false, false, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, false, false, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, false, false, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, false, true, true, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, false, true, true, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, false, true, false, false, 2, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);



template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    192,
                    128,
                    64,
                    8,
                    true,
                    true,
                    mctlass::half_t,
                    128>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                false, false, false, false, false, false, true, false, false, 1, false, Arch::xcore1000
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);

