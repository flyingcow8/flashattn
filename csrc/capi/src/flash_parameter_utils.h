#pragma once

#include "tensor.h"
#include "flash_parameter.h" // Parameter
#include <iostream>
#include <stdexcept>

#define ALIGNUP(m, n) (((m) + (n) - 1) / (n) * (n))

void set_params_fprop(mcFlashAttn::Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      const size_t d_v,
                      const size_t d_v_rounded,
                      // device pointers
                      Tensor_t q,
                      Tensor_t k,
                      Tensor_t v,
                      Tensor_t out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap=0.0f,
                      bool seqlenq_ngroups_swapped=false);

void set_params_dgrad(mcFlashAttn::Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      const size_t d_v,
                      const size_t d_v_rounded,
                      // device pointers
                      const Tensor_t q,
                      const Tensor_t k,
                      const Tensor_t v,
                      const Tensor_t out,
                      const Tensor_t dout,
                      Tensor_t dq,
                      Tensor_t dk,
                      Tensor_t dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool deterministic,
                      const float softcap=0.0f);

void set_params_alibi(mcFlashAttn::Flash_fwd_params &params, Tensor_t alibi_slopes, int batch_size, int num_heads);


// score ->[bs,      head_num,      q,      k  ]
// mask -> [bs_mask, head_num_mask, q_mask, k_mask]
// Mask shape should satisfy these rules
// 1. bs % bs_mask == 0
// 2. head_num % head_num_mask == 0
// 3. q_mask == 1 or q_mask == q
// 4. k_mask == 1 or k_mask == k or k_mask == (k + 3) / 4 * 4 (align k to multiples of 4)
std::vector<int64_t>
get_attn_mask_stride(std::vector<int64_t> &mask_shape, std::vector<int64_t> &score_shape);

void set_params_attn_mask(mcFlashAttn::Flash_fwd_params &params, Tensor_t attn_mask, int batch_size, int num_heads, int seqlen_q, int seqlen_k);

/*
* Set parameters of input
*/

mcflashattnStatus_t set_fwd_parameters(
    mcFlashAttn::Flash_fwd_params &params,
    const Tensor_t q,                 // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t k,         // batch_size x seqlen_k x num_heads_k x head_size
    const Tensor_t v,         // batch_size x seqlen_k x num_heads_k x head_size
    Tensor_t out,             // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t alibi_slopes, // num_heads or batch_size x num_heads
    const Tensor_t attn_mask,
    const Tensor_t softmax_lse,   // batch_size x num_heads x seqlen_q
    const Tensor_t p, // [optional return]softmax batch_size x num_heads x seqlen_q_rounded x seqlen_k_rounded
    const Tensor_t rng_state, // [optional input] 2 x Int64
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    mcflashattnExtendParameter_t extend_parameter_// extend paramerter
);

mcflashattnStatus_t set_varlen_fwd_parameters(
    mcFlashAttn::Flash_fwd_params& params,
    int64_t batch_size,
    const Tensor_t q,                // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const Tensor_t k,         // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const Tensor_t v,         // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    Tensor_t out,             // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const Tensor_t cu_seqlens_q,  // b+1
    const Tensor_t cu_seqlens_k,  // b+1
    const Tensor_t seqused_k,      // b. If given, only this many elements of each batch element's keys are used.
    const Tensor_t alibi_slopes, // num_heads or batch_size x num_heads
    const Tensor_t softmax_lse,   // batch_size x num_heads x seqlen_q
    const Tensor_t p, // [optional return]softmax batch_size x num_heads x seqlen_q_rounded x seqlen_k_rounded
    const Tensor_t rng_state, // [optional input] 2 x Int64
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    mcflashattnExtendParameter_t extend_parameter_// extend paramerter
);

mcflashattnStatus_t
set_fwd_kvcache_parameters(
    mcFlashAttn::Flash_fwd_params &params,
    const Tensor_t q,                 // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t kcache,      // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const Tensor_t vcache,      // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const Tensor_t k,                 // batch_size x seqlen_knew x num_heads_k x head_size
    const Tensor_t v,                // batch_size x seqlen_knew x num_heads_k x head_size
    const Tensor_t seqlens_k,        // batch_size
    const Tensor_t rotary_cos,       // seqlen_ro x (rotary_dim / 2)
    const Tensor_t rotary_sin,       // seqlen_ro x (rotary_dim / 2)
    const Tensor_t cache_batch_idx,  // indices to index into the KV cache
    const Tensor_t block_table,      // batch_size x max_num_blocks_per_seq
    const Tensor_t alibi_slopes,     // num_heads or batch_size x num_heads
    const Tensor_t softmax_lse,     // batch_size x num_heads x seqlen_q
    Tensor_t out,                  // batch_size x seqlen_q x num_heads x head_size
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    int num_splits,
    const Tensor_t softmax_lse_accum, // num_splits x batch_size x num_heads x seqlen_q
    const Tensor_t out_accum,  // num_splits x batch_size x num_heads x max_seqlen_q x head_size_rounded (32)
    mcflashattnExtendParameter_t extend_parameter_, // extend paramerter
    bool for_get_num_splits = false // if true, don't check out_accum and softmax_lse_accum
);

/*
// score ->[bs,      head_num,      q,      k  ]
// mask -> [bs_mask, head_num_mask, q_mask, k_mask]
// Mask shape should satisfy these rules
// 1. bs % bs_mask == 0
// 2. head_num % head_num_mask == 0
// 3. q_mask == 1 or q_mask == q
// 4. k_mask == 1 or k_mask == k or k_mask == (k + 3) / 4 * 4 (align k to multiples of 4)
*/
mcflashattnStatus_t
set_bwd_parameters(
    mcFlashAttn::Flash_bwd_params &params,
    const Tensor_t dout,  // batch_size x seqlen_q x num_heads, x head_size_og
    const Tensor_t q,   // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t k,   // batch_size x seqlen_k x num_heads_k x head_size
    const Tensor_t v,   // batch_size x seqlen_k x num_heads_k x head_size
    Tensor_t out,   // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t softmax_d, // batch_size x num_heads x seqlen_q_rounded
    const Tensor_t softmax_lse,     // b x h x seqlen_q
    const Tensor_t dq,   // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t dk,   // batch_size x seqlen_k x num_heads x head_size
    const Tensor_t dv,   // batch_size x seqlen_k x num_heads x head_size
    const Tensor_t dq_accum,   // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t alibi_slopes, // num_heads or batch_size x num_heads
    const Tensor_t attn_mask,
    const Tensor_t rng_state, // [optional input] 2 x Int64
    const float p_dropout,         // probability to drop
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const bool deterministic,
    mcflashattnExtendParameter_t extend_parameter_// extend paramerter
);

mcflashattnStatus_t
set_varlen_bwd_parameters(
    mcFlashAttn::Flash_bwd_params &params,
    const Tensor_t dout,  // total_q x num_heads, x head_size
    const Tensor_t q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const Tensor_t k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const Tensor_t v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    Tensor_t out,   // total_q x num_heads x head_size
    const Tensor_t softmax_d, // batch_size x num_heads x seqlen_q_rounded
    const Tensor_t softmax_lse,     // b x h x s   softmax logsumexp
    const Tensor_t dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const Tensor_t dk,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const Tensor_t dv,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const Tensor_t dq_accum,   // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t cu_seqlens_q,  // b+1
    const Tensor_t cu_seqlens_k,  // b+1
    const Tensor_t alibi_slopes, // num_heads or b x num_heads
    const Tensor_t rng_state, // [optional input] 2 x Int64
    const int max_seqlen_q,
    const int max_seqlen_k,          // max sequence length to choose the kernel
    const float p_dropout,         // probability to drop
    const float softmax_scale,
    // const bool zero_tensors, python set to false
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const bool deterministic,
    mcflashattnExtendParameter_t extend_parameter_ // extend paramerter
);
