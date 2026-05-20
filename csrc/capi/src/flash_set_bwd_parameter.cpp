#include "flash_parameter_utils.h"
#include "utils.h"

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
) {
    float softcap = 0.0f;
    if (extend_parameter_ != nullptr){
        softcap = get_extend_parameter_softcap(extend_parameter_);
    }
    const int batch_size_ = get_tensor_size(q,0);
    const int seqlen_q_ = get_tensor_size(q,1);
    const int num_heads_ = get_tensor_size(q,2);
    const int head_size_og_ = get_tensor_size(dout,3);
    const int head_size_ = get_tensor_size(q,3);
    const int head_size_v_ = get_tensor_size(v,3);
    const int seqlen_k_ =get_tensor_size(k,1);
    const int num_heads_k_ =get_tensor_size(k,2);

    TENSOR_CHECK(batch_size_ > 0, "batch size must be positive");
    TENSOR_CHECK(head_size_ % 8 == 0, "head_size should be a multiple of 8");
    TENSOR_CHECK(head_size_ <= 256, "FlashAttention backward only supports head dimension <= 256");
    TENSOR_CHECK(num_heads_ % num_heads_k_ == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // we do not have headdim=224 kernel, padding head_size_rounded to 256 to support theses headdim
    const int head_size_rounded = round_multiple(head_size_, 32) == 224 ? 256 : round_multiple(head_size_, 32);
    const int head_size_v_rounded = round_multiple(head_size_v_,32) == 224 ? 256 : round_multiple(head_size_v_,32);
    const int seqlen_q_rounded = round_multiple(seqlen_q_, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k_, 128);

    TENSOR_CHECK(head_size_ == round_multiple(head_size_og_, 8), "head_size must be head_size_og rounded to a multiple of 8");

    if (is_causal) { window_size_right = 0; }
    if (window_size_left >= seqlen_k_) { window_size_left = -1; }
    if (window_size_right >= seqlen_k_) { window_size_right = -1; }

    CHECK_SHAPE(q, {batch_size_, seqlen_q_, num_heads_, head_size_});
    CHECK_SHAPE(k, {batch_size_, seqlen_k_, num_heads_k_, head_size_});
    CHECK_SHAPE(v, {batch_size_, seqlen_k_, num_heads_k_, head_size_});
    CHECK_SHAPE(out, {batch_size_, seqlen_q_, num_heads_, head_size_});
    CHECK_SHAPE(dout, {batch_size_, seqlen_q_, num_heads_, head_size_og_});
    CHECK_SHAPE(dq, {batch_size_, seqlen_q_, num_heads_, head_size_og_});
    //dk and dv should use shape (b, sk, h_q, d) and reduce_sum to (b, sk, h_k, d) in GQA/MQA
    CHECK_SHAPE(dk, {batch_size_, seqlen_k_, num_heads_, head_size_og_});
    CHECK_SHAPE(dv, {batch_size_, seqlen_k_, num_heads_, head_size_og_});

    set_params_dgrad(params,
                     batch_size_,
                     seqlen_q_, seqlen_k_,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads_, num_heads_k_,
                     head_size_, head_size_rounded,
                     head_size_v_,head_size_v_rounded,
                     q, k, v, out,
                     dout, dq, dk, dv,
                     nullptr,
                     nullptr,
                     get_tensor_data(dq_accum),
                     nullptr,
                     nullptr,
                     get_tensor_data(softmax_lse),
                     get_tensor_data(softmax_d),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     deterministic,
                     softcap);
    params.packed_seqlen = 0;
    params.dq_accum_split_stride = !deterministic ? 0 : get_tensor_stride(dq_accum,0);

    // in capi rng_state should input when dropout > 0.f
    if (p_dropout > 0.0)  {
        TENSOR_CHECK(rng_state != NULL, "when use dropout,  rng_state is necessary");
        CHECK_CONTIGUOUS(rng_state);
        CHECK_DTYPE(rng_state, InternalTensor::DataType::INT64);
        CHECK_SHAPE(rng_state, {2});

        auto rng_state_p = static_cast<int64_t*>(get_tensor_data(rng_state));
        params.rng_state_seed = rng_state_p[0];
        params.rng_state_offset = rng_state_p[1];
    }

    set_params_alibi(params, alibi_slopes, batch_size_, num_heads_);

    set_params_attn_mask(params, attn_mask, batch_size_, num_heads_, seqlen_q_, seqlen_k_);

    return MCFLASHATTN_STATUS_SUCCESS;
}

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
){
    float softcap = 0.0f;
    if (extend_parameter_ != nullptr){
        softcap = get_extend_parameter_softcap(extend_parameter_);
    }
    const int total_q_ = get_tensor_size(q,0);
    const int batch_size_ = get_tensor_size(cu_seqlens_q,0) - 1; // b
    const int num_heads_ = get_tensor_size(q,1);
    const int head_size_og_ = get_tensor_size(dout,2);
    const int head_size_ = get_tensor_size(q,2);
    const int head_size_v_ = get_tensor_size(v,2);
    const int total_k_ = get_tensor_size(k,0);
    const int num_heads_k_ = get_tensor_size(k,1);

    TENSOR_CHECK(batch_size_ > 0, "batch size must be positive");
    TENSOR_CHECK(head_size_ % 8 == 0, "head_size should be a multiple of 8");
    TENSOR_CHECK(head_size_ <= 256, "FlashAttention backward only supports head dimension <= 256");
    TENSOR_CHECK(num_heads_ % num_heads_k_ == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // we do not have headdim=224 kernel, padding head_size_rounded to 256 to support theses headdim
    const int head_size_rounded = round_multiple(head_size_, 32) == 224 ? 256 : round_multiple(head_size_, 32);
    const int head_size_v_rounded = round_multiple(head_size_v_,32) == 224 ? 256 : round_multiple(head_size_v_,32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    TENSOR_CHECK(head_size_ == round_multiple(head_size_og_, 8), "head_size must be head_size_og rounded to a multiple of 8");

    if (is_causal) { window_size_right = 0; }
    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, {total_q_, num_heads_, head_size_});
    CHECK_SHAPE(k, {total_k_, num_heads_k_, head_size_});
    CHECK_SHAPE(v, {total_k_, num_heads_k_, head_size_});
    CHECK_SHAPE(out, {total_q_, num_heads_, head_size_});
    CHECK_SHAPE(dout, {total_q_, num_heads_, head_size_og_});
    CHECK_SHAPE(dq, {total_q_, num_heads_, head_size_og_});

    //dk and dv should use shape (total_sk, h_q, d) and reduce_sum to  (total_sk, h_k, d) in GQA/MQA
    CHECK_SHAPE(dk, {total_k_, num_heads_, head_size_og_});
    CHECK_SHAPE(dv, {total_k_, num_heads_, head_size_og_});
    CHECK_SHAPE(cu_seqlens_q, {batch_size_ + 1});
    CHECK_SHAPE(cu_seqlens_k, {batch_size_ + 1});

    set_params_dgrad(params,
                     batch_size_,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads_, num_heads_k_,
                     head_size_, head_size_rounded,
                     head_size_v_, head_size_v_rounded,
                     q, k, v, out,
                     dout, dq, dk, dv,
                     get_tensor_data(cu_seqlens_q),
                     get_tensor_data(cu_seqlens_k),
                     get_tensor_data(dq_accum),
                     nullptr,
                     nullptr,
                     get_tensor_data(softmax_lse),
                     get_tensor_data(softmax_d),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     deterministic,
                     softcap);
    params.packed_seqlen = total_q_;  // asm mha_bwd kernel needs packed_seqlen
    params.dq_accum_split_stride = !deterministic ? 0 : get_tensor_stride(dq_accum,0);

    // in capi rng_state should input when dropout > 0.f
    if (p_dropout > 0.0)  {
        TENSOR_CHECK(rng_state != NULL, "when use dropout,  rng_state is necessary");
        CHECK_CONTIGUOUS(rng_state);
        CHECK_DTYPE(rng_state, InternalTensor::DataType::INT64);
        CHECK_SHAPE(rng_state, {2});

        auto rng_state_p = static_cast<int64_t*>(get_tensor_data(rng_state));
        params.rng_state_seed = rng_state_p[0];
        params.rng_state_offset = rng_state_p[1];
    }

    set_params_alibi(params, alibi_slopes, batch_size_, num_heads_);
    return MCFLASHATTN_STATUS_SUCCESS;
}
