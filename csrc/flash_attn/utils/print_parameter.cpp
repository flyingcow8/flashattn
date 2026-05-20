#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "flash_parameter.h"
#include "logger.h"
#include "process_str.h"
#include "print_parameter.h"

std::stringstream process_params(mcFlashAttn::Flash_bwd_params params, const mcFlashAttn::Flash_launch_params launch_params,
                                            bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask ,const std::string& debug_flag) {
    /*

        ==============================Parts that require special handling（cu_seqlens）==============================

    */
    std::stringstream cu_seqlen_q;
    std::stringstream cu_seqlen_k;
    if (params.cu_seqlens_q != nullptr) {
        std::vector<int> host_cuseq_q(params.b + 1);
        cudaMemcpy(host_cuseq_q.data(), params.cu_seqlens_q, (sizeof(int) * (params.b + 1)), cudaMemcpyDeviceToHost);
        for(int i = 0; i < (params.b + 1); ++i) {
            if(cu_seqlen_q.str().size() == 0)
            {
                cu_seqlen_q << "[";
            } else if(cu_seqlen_q.str().size() > 1) {
                cu_seqlen_q << "-";
            }
            cu_seqlen_q << std::to_string(host_cuseq_q[i]);
        }
        cu_seqlen_q << "]";
    } else {
        cu_seqlen_q << "[nil]";
    }

    if (params.cu_seqlens_k != nullptr) {
        // when debug_flag is kvcache, cu_seqlens_k_size == batch_size
        // when debug_flag is fwd & bwd, cu_seqlens_k_size == batch_size + 1
        const int seq_k_size = debug_flag == "kvcache" ? params.b : (params.b + 1);
        std::vector<int> host_cuseq_k(seq_k_size);
        cudaMemcpy(host_cuseq_k.data(), params.cu_seqlens_k, (sizeof(int) * (seq_k_size)), cudaMemcpyDeviceToHost);
        for(int i = 0; i < seq_k_size; ++i) {
            if(cu_seqlen_k.str().size() == 0)
            {
                cu_seqlen_k << "[";
            } else if(cu_seqlen_k.str().size() > 1) {
                cu_seqlen_k << "-";
            }
            cu_seqlen_k << std::to_string(host_cuseq_k[i]);
        }
        cu_seqlen_k << "]";
    } else {
        cu_seqlen_k << "[nil]";
    }

    /*

        ==============================The part where the Bool_switch is printed==============================

    */

    bool Merge_attn_mask_ldg = false;
    if( Has_attn_mask && (params.attn_mask_col_shape % 4) == 0 && params.attn_mask_col_stride == 1) {
        Merge_attn_mask_ldg = true;
    }

    bool deterministic = false;

    if(debug_flag == "bwd") {
        deterministic = params.deterministic;
    }

    bool Split = params.num_splits > 1;

    std::vector<std::string> bool_info{
                                /*================
                                The unique part of the fwd and shared parts.
                                ================*/
                                std::to_string(Is_dropout),
                                std::to_string(params.is_bf16),
                                std::to_string(Is_causal),
                                std::to_string(Is_local),
                                std::to_string(Has_attn_mask),
                                std::to_string(params.is_seqlens_k_cumulative),
                                std::to_string(params.is_rotary_interleaved),
                                std::to_string(params.unpadded_lse),
                                std::to_string(Has_alibi),
                                std::to_string(Merge_attn_mask_ldg),
                                /*================
                                The unique part of the kvcache
                                ================*/
                                std::to_string(Split),
                                /*================
                                The unique part of the bwd
                                =================*/
                                std::to_string(deterministic)
    };
    /*

        ==============================The part where the Dim_info is printed==============================

    */
    float dropout = (1 - params.p_dropout);
    const int perf_mode = (launch_params.is_balance << 2) + launch_params.block_type;
    std::vector<std::string> dim_info{
                                std::to_string(params.b),
                                std::to_string(params.seqlen_q),
                                std::to_string(params.seqlen_k),
                                std::to_string(params.h),
                                std::to_string(params.h_k),
                                std::to_string(params.d),
                                std::to_string(params.d_value),
                                std::to_string(perf_mode),
                                std::to_string(params.num_splits),
                                std::to_string(params.page_block_size),
                                std::to_string(params.window_size_left),
                                std::to_string(params.window_size_right),
                                std::to_string(dropout),
                                std::to_string(params.scale_softmax),
                                std::to_string(params.softcap),
                                std::to_string(params.seqlen_knew),
                                std::to_string(params.seqlen_q_rounded),
                                std::to_string(params.seqlen_k_rounded),
                                std::to_string(params.d_rounded),
                                std::to_string(params.rotary_dim),
                                std::to_string(params.attn_mask_batch_shape),
                                std::to_string(params.attn_mask_nheads_shape),
                                std::to_string(params.attn_mask_row_shape),
                                std::to_string(params.attn_mask_col_shape),
                                /*================
                                cu_seqlen
                                ================*/
                                cu_seqlen_q.str(),
                                cu_seqlen_k.str(),
    };

    return concat_total_strs("MHA", debug_flag, bool_info, dim_info);
}

void shape_print(mcFlashAttn::Flash_bwd_params params, const mcFlashAttn::Flash_launch_params launch_params,
                            bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask ,const std::string& debug_flag) {
    auto total_strs = process_params(params, launch_params, Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, debug_flag);
    /*
    Bool Switch:
                is_dropout, is_bf16, is_causal, Is_local, has_attn_mask, is_seqlens_k_cumulative, is_rotary_interleaved, unpadded_lse,
                Has_alibi, Split, Is_deterministic,
    Dim_int:
                h, h_k, h_h_k_ratio, b, d, seqlen_q, seqlen_k, seqlen_knew, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, window_size_left, window_size_right
                attn_mask_batch_shape, attn_mask_nheads_shape, attn_mask_row_shape, attn_mask_col_shape, num_split,
    Dim_float:
                dropout, scale_softmax, softcap
    cu_seqlens:
                cu_seqlens_q, cu_seqlens_k,

    (unpadded_lse, softcap) is added since 2024/12/02
    (attn_mask_batch_shape, attn_mask_nheads_shape, attn_mask_row_shape, attn_mask_col_shape) is added since 2024/12/23

    2025/01/13:
        Bool Switch:
                    is_dropout, is_bf16, is_causal, Is_local, has_attn_mask, is_seqlens_k_cumulative, is_rotary_interleaved, unpadded_lse,
                    Has_alibi, Split, Is_deterministic, Merge_attn_mask_ldg
        Dim_info:
                    b, seqlen_q, seqlen_k, h_q, h_k, d_qk, d_v, perf_mode, num_split, page_size, window_size_left, window_size_right, dropout, scale_softmax, softcap,
                    seqlen_knew, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim,
                    attn_mask_batch_shape, attn_mask_nheads_shape, attn_mask_row_shape, attn_mask_col_shape, cu_seqlen_q, cu_seqlen_k
    */
    LOG_SHAPE("%s\n", total_strs.str().c_str());
}

void debug_print(mcFlashAttn::Flash_bwd_params params, const mcFlashAttn::Flash_launch_params launch_params, const std::string& debug_flag, std::vector<int>* extra_flag) {
    const int perf_mode = (launch_params.is_balance << 2) + launch_params.block_type;
    printf("==============%s-debug parameters recored start...\n", debug_flag.c_str());
    if(debug_flag == "fwd") {
        printf("----rng_state_seed=%d\n",params.rng_state_seed);
        printf("----rng_state_offset=%d\n",params.rng_state_offset);
        printf("----attn_mask_ptr=%p\n", params.attn_mask_ptr);
        printf("----attn_mask_batch_shape=%ld\n", params.attn_mask_batch_shape);
        printf("----attn_mask_nheads_shape=%ld\n", params.attn_mask_nheads_shape);
        printf("----attn_mask_row_shape=%ld\n", params.attn_mask_row_shape);
        printf("----attn_mask_col_shape=%ld\n", params.attn_mask_col_shape);
        printf("----attn_mask_batch_stride=%ld\n", params.attn_mask_batch_stride);
        printf("----attn_mask_nheads_stride=%ld\n", params.attn_mask_nheads_stride);
        printf("----attn_mask_row_stride=%ld\n", params.attn_mask_row_stride);
        printf("----attn_mask_col_stride=%ld\n", params.attn_mask_col_stride);
        printf("----p_ptr=%p\n", params.p_ptr);
        printf("----p_dropout_in_uint8_t=%d\n",params.p_dropout_in_uint8_t);
        printf("----rp_dropout=%f\n",params.rp_dropout);
    }
    if(debug_flag == "kvcache") {
        printf("----rotary_cos_ptr=%p\n", params.rotary_cos_ptr);
        printf("----rotary_sin_ptr=%p\n", params.rotary_sin_ptr);
        printf("----cache_batch_idx=%p\n", params.cache_batch_idx);
        printf("----block_table=%p\n", params.block_table);
        printf("----block_table_batch_stride=%ld\n", params.block_table_batch_stride);
        printf("----page_block_size=%d\n", params.page_block_size);
        printf("----knew_ptr=%p\n",params.knew_ptr);
        printf("----vnew_ptr=%p\n",params.vnew_ptr);
        printf("----oaccum_ptr=%p\n", params.oaccum_ptr);
        printf("----knew_batch_stride=%ld\n",params.knew_batch_stride);
        printf("----vnew_batch_stride=%ld\n",params.vnew_batch_stride);
        printf("----knew_row_stride=%ld\n",params.knew_row_stride);
        printf("----vnew_row_stride=%ld\n",params.vnew_row_stride);
        printf("----knew_head_stride=%ld\n",params.knew_head_stride);
        printf("----vnew_head_stride=%ld\n",params.vnew_head_stride);
        printf("----num_splits=%d\n",params.num_splits);
        printf("----softmax_lseaccum_ptr=%p\n", params.softmax_lseaccum_ptr);
    }
    printf("----softmax_lse_ptr=%p\n", params.softmax_lse_ptr);
    printf("----seqused_k=%p\n",params.seqused_k);
    printf("----q_ptr=%p\n", params.q_ptr);
    printf("----k_ptr=%p\n", params.k_ptr);
    printf("----v_ptr=%p\n", params.v_ptr);
    printf("----o_ptr=%p\n", params.o_ptr);
    printf("----alibi_slopes_ptr=%p\n", params.alibi_slopes_ptr);
    printf("----alibi_slopes_batch_stride=%ld\n", params.alibi_slopes_batch_stride);
    printf("----window_size_left=%d\n", params.window_size_left);
    printf("----window_size_right=%d\n", params.window_size_right);
    printf("----scale_softmax=%f\n",params.scale_softmax);
    printf("----scale_softmax_log2=%f\n",params.scale_softmax_log2);
    printf("----o_batch_stride=%ld\n", params.o_batch_stride);
    printf("----o_row_stride=%ld\n",params.o_row_stride);
    printf("----o_head_stride=%ld\n",params.o_head_stride);
    printf("----q_batch_stride=%ld\n",params.q_batch_stride);
    printf("----k_batch_stride=%ld\n",params.k_batch_stride);
    printf("----v_batch_stride=%ld\n",params.v_batch_stride);
    printf("----q_row_stride=%ld\n",params.q_row_stride);
    printf("----k_row_stride=%ld\n",params.k_row_stride);
    printf("----v_row_stride=%ld\n",params.v_row_stride);
    printf("----q_head_stride=%ld\n",params.q_head_stride);
    printf("----k_head_stride=%ld\n",params.k_head_stride);
    printf("----v_head_stride=%ld\n",params.v_head_stride);
    printf("----unpadded lse=%d\n", params.unpadded_lse);
    printf("----perf_mode=%d\n", perf_mode);
    printf("----softcap=%f\n", params.softcap);
    printf("----attn_mask_batch_shape=%d\n", params.attn_mask_batch_shape);
    printf("----attn_mask_nheads_shape=%d\n", params.attn_mask_nheads_shape);
    printf("----attn_mask_row_shape=%d\n", params.attn_mask_row_shape);
    printf("----attn_mask_col_shape=%d\n", params.attn_mask_col_shape);

    if(debug_flag == "bwd") {
        printf("----grid_m(%d,%d,%d)\n",  (*extra_flag)[0], (*extra_flag)[1],  (*extra_flag)[2]);
        printf("----grid_n(%d,%d,%d)\n",  (*extra_flag)[3],  (*extra_flag)[4],  (*extra_flag)[5]);
        printf("----blocksize: %d\n",  (*extra_flag)[6]);
        printf("----shared memory size: %d\n",  (*extra_flag)[7]);
        printf("----p_dropout=%f\n",params.p_dropout);
        printf("----p_dropout_in_uint8_t=%d\n",params.p_dropout_in_uint8_t);
        printf("----rng_state_seed=%d\n",params.rng_state_seed);
        printf("----rng_state_offset=%d\n",params.rng_state_offset);
        printf("----rp_dropout=%f\n",params.rp_dropout);
        printf("----scale_softmax_rp_dropout=%f\n",params.scale_softmax_rp_dropout);
        printf("--------------------------bwd start--------------------------------\n");
        printf("----dq_ptr=%p\n", params.dq_ptr);
        printf("----dk_ptr=%p\n", params.dk_ptr);
        printf("----dv_ptr=%p\n", params.dv_ptr);
        printf("----do_ptr=%p\n", params.do_ptr);
        printf("----dq_accum_ptr=%p\n", params.dq_accum_ptr);
        printf("----dsoftmax_sum=%p\n", params.dsoftmax_sum);
        printf("----do_batch_stride=%ld\n", params.do_batch_stride);
        printf("----do_row_stride=%ld\n",params.do_row_stride);
        printf("----do_head_stride=%ld\n",params.do_head_stride);
        printf("----dq_batch_stride=%ld\n",params.dq_batch_stride);
        printf("----dk_batch_stride=%ld\n",params.dk_batch_stride);
        printf("----dv_batch_stride=%ld\n",params.dv_batch_stride);
        printf("----dq_row_stride=%ld\n",params.dq_row_stride);
        printf("----dk_row_stride=%ld\n",params.dk_row_stride);
        printf("----dv_row_stride=%ld\n",params.dv_row_stride);
        printf("----dq_head_stride=%ld\n",params.dq_head_stride);
        printf("----dk_head_stride=%ld\n",params.dk_head_stride);
        printf("----dv_head_stride=%ld\n",params.dv_head_stride);
        printf("----deterministic=%d\n",params.deterministic);
        printf("----dq_accum_split_stride=%ld\n",params.dq_accum_split_stride);
        printf("--------------------------bwd end--------------------------------\n");
    }
    printf("==============%s-debug parameters recored end...\n", debug_flag.c_str());
}
