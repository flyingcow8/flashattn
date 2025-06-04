#pragma once

#include "flash_attn.h"
#include "py_export_capi_utils.h"

std::vector<at::Tensor> mha_fwd_inference_test(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                               c10::optional<at::Tensor> &out_,
                                               c10::optional<at::Tensor> &alibi_slopes_,
                                               c10::optional<at::Tensor> &attn_mask_, const float softmax_scale,
                                               bool is_causal, int window_size_left, int window_size_right);

std::vector<at::Tensor> mha_varlen_fwd_inference_test(
    at::Tensor &q, const at::Tensor &k, const at::Tensor &v, c10::optional<at::Tensor> &out_,
    const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_k, c10::optional<at::Tensor> &seqused_k,
    c10::optional<at::Tensor> &alibi_slopes_, int max_seqlen_q, const int max_seqlen_k, const float softmax_scale,
    const bool zero_tensors, bool is_causal, int window_size_left, int window_size_right);

std::vector<at::Tensor> mha_fwd_kvcache_test(
    at::Tensor &q, const at::Tensor &kcache, const at::Tensor &vcache, c10::optional<at::Tensor> &k_,
    c10::optional<at::Tensor> &v_, c10::optional<at::Tensor> &seqlens_k_, c10::optional<at::Tensor> &rotary_cos_,
    c10::optional<at::Tensor> &rotary_sin_, c10::optional<at::Tensor> &cache_batch_idx_,
    c10::optional<at::Tensor> &block_table_, c10::optional<at::Tensor> &alibi_slopes_, c10::optional<at::Tensor> &out_,
    const float softmax_scale, bool is_causal, int window_size_left, int window_size_right, bool is_rotary_interleaved,
    int num_splits);
