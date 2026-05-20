#include "flash_parameter_utils.h"
#include "utils.h"

using namespace mcFlashAttn;

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
){
    // get shape and stride
    const int batch_size_ = get_tensor_size(q,0);
    int seqlen_q_ = get_tensor_size(q,1);
    int num_heads_ = get_tensor_size(q,2);
    const int head_size_pad = get_tensor_size(q,3); // pad to 8 by user
    const int head_size_v_pad = get_tensor_size(v,3); // pad to 8 by user
    float softcap = 0.0f;
    if (extend_parameter_ != nullptr){
        softcap = get_extend_parameter_softcap(extend_parameter_);
    }

    if(head_size_pad % 8 != 0)  return MCFLASHATTN_STATUS_FAILED; // Need pad

    const int seqlen_k_ = get_tensor_size(k,1);
    const int num_heads_k_ = get_tensor_size(k,2);

    if (window_size_left >= seqlen_k_) { window_size_left = -1; }
    if (window_size_right >= seqlen_k_) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q_ == 1 && !alibi_slopes) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // const int head_size = round_multiple(head_size_og, 8);
    // we do not have headdim=224 kernel, padding head_size_rounded to 256 to support theses headdim
    const int head_size_rounded = round_multiple(head_size_pad, 32) == 224 ? 256 : round_multiple(head_size_pad, 32);
    const int head_size_v_rounded = round_multiple(head_size_v_pad, 32) == 224 ? 256 : round_multiple(head_size_v_pad, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q_, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k_, 128);

    set_params_fprop(params,
                     batch_size_,
                     seqlen_q_, seqlen_k_,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads_, num_heads_k_,
                     head_size_pad, head_size_rounded,
                     head_size_v_pad, head_size_v_rounded,
                     q, k, v, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     (p != NULL) ? get_tensor_data(p) : nullptr,
                     (softmax_lse != NULL) ? get_tensor_data(softmax_lse) : nullptr,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap);

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
){
    float softcap = 0.0f;
    if (extend_parameter_ != nullptr){
        softcap = get_extend_parameter_softcap(extend_parameter_);
    }
    const int batch_size_ = batch_size;
    int num_heads_ = get_tensor_size(q,1);
    const int head_size_og_ = get_tensor_size(q,2);
    const int head_size_v_og_ = get_tensor_size(v,2);
    const int total_k_ = get_tensor_size(k,0);
    const int num_heads_k_ = get_tensor_size(k,1);

    if (max_seqlen_q == 1 && alibi_slopes == nullptr) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) { window_size_right = 0; }

    const int total_q_ = get_tensor_size(q,0);

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_ = round_multiple(head_size_og_, 8);
    const int head_size_v = round_multiple(head_size_v_og_,8);
    // we do not have headdim=224 kernel, padding head_size_rounded to 256 to support theses headdim
    const int head_size_rounded = round_multiple(head_size_, 32) == 224 ? 256 : round_multiple(head_size_, 32);
    const int head_size_v_rounded = round_multiple(head_size_v,32) == 224 ? 256 : round_multiple(head_size_v,32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    set_params_fprop(params,
                     batch_size_,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads_, num_heads_k_,
                     head_size_, head_size_rounded,
                     head_size_v,head_size_v_rounded,
                     q, k, v, out,
                     cu_seqlens_q != nullptr ? get_tensor_data(cu_seqlens_q) : nullptr,
                     cu_seqlens_k != nullptr ? get_tensor_data(cu_seqlens_k) : nullptr,
                     seqused_k != nullptr ? get_tensor_data(seqused_k) : nullptr,
                     p != nullptr ? get_tensor_data(p) : nullptr,
                     softmax_lse != nullptr ? get_tensor_data(softmax_lse) : nullptr,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap);

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
