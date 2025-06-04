#pragma once

#include "flash_attn.h"
#include "py_export_capi_utils.h"

std::vector<at::Tensor> mha_fwd_capi_test(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                          c10::optional<at::Tensor> &out_, c10::optional<at::Tensor> &alibi_slopes_,
                                          c10::optional<at::Tensor> &attn_mask_, const float p_dropout,
                                          const float softmax_scale, bool is_causal, int window_size_left,
                                          int window_size_right, const bool return_softmax,
                                          c10::optional<at::Generator> gen_);

std::vector<at::Tensor> mha_varlen_fwd_capi_test(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                                 c10::optional<at::Tensor> &out_, const at::Tensor &cu_seqlens_q,
                                                 const at::Tensor &cu_seqlens_k, c10::optional<at::Tensor> &seqused_k,
                                                 c10::optional<at::Tensor> &alibi_slopes_, int max_seqlen_q,
                                                 const int max_seqlen_k, const float p_dropout,
                                                 const float softmax_scale, const bool zero_tensors, bool is_causal,
                                                 int window_size_left, int window_size_right, const bool return_softmax,
                                                 c10::optional<at::Generator> gen_);

std::vector<at::Tensor> mha_bwd_capi_test(const at::Tensor &dout, const at::Tensor &q, const at::Tensor &k,
                                          const at::Tensor &v, const at::Tensor &out, const at::Tensor &softmax_lse,
                                          c10::optional<at::Tensor> &dq_, c10::optional<at::Tensor> &dk_,
                                          c10::optional<at::Tensor> &dv_, c10::optional<at::Tensor> &alibi_slopes_,
                                          c10::optional<at::Tensor> &attn_mask_, const float p_dropout,
                                          const float softmax_scale, const bool is_causal, int window_size_left,
                                          int window_size_right, const bool deterministic,
                                          c10::optional<at::Generator> gen_, c10::optional<at::Tensor> &rng_state_);

std::vector<at::Tensor> mha_varlen_bwd_capi_test(
    const at::Tensor &dout, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &out,
    const at::Tensor &softmax_lse, c10::optional<at::Tensor> &dq_, c10::optional<at::Tensor> &dk_,
    c10::optional<at::Tensor> &dv_, const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_k,
    c10::optional<at::Tensor> &alibi_slopes_, const int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, const bool is_causal, int window_size_left,
    int window_size_right, const bool deterministic, c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor> &rng_state_);