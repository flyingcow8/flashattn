#pragma once

#include "flash_attn.h"
#include "py_export_capi_utils.h"


std::vector<at::Tensor>
mha_fwd_kvcache_test(at::Tensor &q,
                const at::Tensor &kcache,
                const at::Tensor &vcache,
                c10::optional<at::Tensor> &k_,
                c10::optional<at::Tensor> &v_,
                c10::optional<at::Tensor> &seqlens_k_,
                c10::optional<at::Tensor> &rotary_cos_,
                c10::optional<at::Tensor> &rotary_sin_,
                c10::optional<at::Tensor> &cache_batch_idx_,
                c10::optional<at::Tensor> &leftpad_k_,
                c10::optional<at::Tensor> &block_table_,
                c10::optional<at::Tensor> &alibi_slopes_,
                c10::optional<at::Tensor> &out_,
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,
                int num_splits,
                c10::optional<at::Tensor> &s_aux_ // (n_heads)
                );
