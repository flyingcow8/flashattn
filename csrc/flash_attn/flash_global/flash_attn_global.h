#pragma once

#include "flash_parameter.h"

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_deterministic=false, bool Is_balance=false, Arch arch=Arch::xcore1000>
__global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel(mcFlashAttn::Flash_bwd_params params, int gridtype);

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, int Rowblock_Parallel_Num = 1, bool Merge_attn_mask_ldg = false, Arch arch = Arch::xcore1000>
__global__ void flash_fwd_kernel(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Is_page_attn, Arch arch = Arch::xcore1000>
__global__ void flash_fwd_splitkv_kernel(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type) ;
