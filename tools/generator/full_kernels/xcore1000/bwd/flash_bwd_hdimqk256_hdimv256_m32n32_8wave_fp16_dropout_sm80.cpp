// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_parameter.h"
#include "flash_bwd_global.h"
#include "kernel_traits.h"
#include <mctlass/numeric_types.h>


template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, true, true, false, false, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, true, true, false, false, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, true, true, false, false, false, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, true, true, false, false, false, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, true, false, false, false, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, true, false, false, false, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, true, false, false, false, false, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, true, false, false, false, false, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, true, false, true, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, true, false, true, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, true, false, false, false, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, true, false, false, false, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, true, false, false, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, true, false, false, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, true, true, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, true, true, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, true, false, false, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, true, false, false, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, true, false, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, true, false, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, false, true, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, false, true, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, false, false, false, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, false, false, false, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, false, false, true, false, true, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);



template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    256,
                    32,
                    32,
                    8,
                    2,
                    2,
                    2,
                    true,
                    true,
                    true,
                    mctlass::half_t,
                    256>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                true, false, false, false, false, false, true, false, false, false, Arch::xcore1000
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);

