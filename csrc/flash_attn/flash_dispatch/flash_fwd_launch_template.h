#pragma once

#include <cuda.h>

#include "flash_parameter.h"
#include "static_switch.h"
#include "flash_performance_mode.h"
#include "print_parameter.h"
#include "flash_attn_global.h"

using namespace mcFlashAttn;

#define CHECK_MSG(x, ...) do { if((x) == false) {throw std::invalid_argument(__VA_ARGS__);} }while(0)
#define CUDA_CHECK(expr) {auto x = (expr); CHECK_MSG(x == cudaSuccess, #expr + std::string(" check failed!"));}
#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, Arch arch>
void run_flash_fwd(Flash_fwd_params &params, const Flash_launch_params &launch_params,cudaStream_t stream) {
    size_t smem_size = Kernel_traits::kSmemSize;
    // int block_type = flash::get_grid_type("MHA_DEBUG_FWD_GTYPE", block_type_);
    if ((Kernel_traits::kHeadDim == 64 || (Kernel_traits::kHeadDim / 32 % 2 == 1)) && params.has_attn_mask) {
        // only fwd hdim64 and k32 will load attn_mask to smem now
        if constexpr (Kernel_traits::kSmemKSize + Kernel_traits::kSmemMaskSize > Kernel_traits::kSmemSize) {
            smem_size = Kernel_traits::kSmemKSize + Kernel_traits::kSmemMaskSize;
        }
    }

    if constexpr (arch == Arch::xcore1500) {
        constexpr int KV_smem_size = Kernel_traits::kSmemVSize + Kernel_traits::kSmemKSize_k_stage;
        constexpr int Q_smem_size = Kernel_traits::kSmemQSize;
        smem_size = Kernel_traits::Share_Q_K_smem ? std::max(Q_smem_size, KV_smem_size) : Q_smem_size + KV_smem_size;
    }

    const int block_type = launch_params.block_type;
    const int rowblock_parallel = launch_params.rowblock_parallel;

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid = flash_fwd_compute_grid_dim(num_m_block, params.h, params.b, rowblock_parallel, block_type);

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    /*merged impl require
        1. bias_col_shape % 4 == 0
        2. bias_col_stride == 1
    */
    bool attn_mask_merge_ldg = false;
    if((params.attn_mask_col_shape % 4) == 0 && params.attn_mask_col_stride == 1) {
        attn_mask_merge_ldg = true;
    }

    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        LOCAL_SWITCH((!Is_causal), (params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
            RETURN_SOFTMAX_SWITCH((Is_dropout), return_softmax, ReturnSoftmaxConst, [&] {
                EVENMN_SWITCH((IsEvenKConst && !Is_local && !ReturnSoftmaxConst), is_even_MN, IsEvenMNConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        ATTN_MASK_SWITCH((!Is_causal && !Is_local && !Has_alibi), params.has_attn_mask, Has_attn_mask, [&] {
                            MERGE_ATTN_MASK_LDG_SWITCH((Has_attn_mask), attn_mask_merge_ldg, Merge_attn_mask_ldg, [&] {
                                ROWNUM_SWITCH(rowblock_parallel, Rowblock_Parallel_Num, [&]  {
                                    SOFTCAP_SWITCH((!Is_dropout), params.softcap > 0.0, Is_softcap, [&]{
                                        // number of templates reduce conditions :
                                        //      If Is_causal, set Is_local to false
                                        //      If Is_causal, set Has_attn_mask to false
                                        //      If Is_local, set Has_attn_mask to false
                                        //      If Has_alibi, set Has_attn_mask to false
                                        //      If not IsEvenKConst, set IsEvenMNConst to false
                                        //      If Is_local, set IsEvenMNConst to false
                                        //      If ReturnSoftmaxConst, set IsEvenMNConst to false
                                        //      If not Is_dropout, set ReturnSoftmaxConst to false
                                        //      If Is_dropout, set Is_softcap to false
                                        if (std::getenv("MHA_PRINT_PARA")) {
                                            shape_print(static_cast<Flash_bwd_params&>(params), launch_params, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, "fwd");
                                        }
                                        if (std::getenv("MHA_DEBUG_PARA")){
                                            debug_print(static_cast<Flash_bwd_params&>(params), launch_params, "fwd");
                                        }
                                        auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, arch>;
                                        if (smem_size >= 32 * 1024) {
                                            CUDA_CHECK(cudaFuncSetAttribute(
                                                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                                        }
                                        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params, num_m_block, block_type);
                                        CUDA_KERNEL_LAUNCH_CHECK();
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    });
}
