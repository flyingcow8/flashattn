#include "flash_attn.h"
#include "tensor.h"
#include "utils.h"
#include "flash_parameter.h"
#include "run_mha.h"
#include <iostream>
#include <stdexcept>

using namespace mcFlashAttn;

__forceinline__ __host__ int mcGetCurrentProcessorCount() {
    int deviceId{};
    mcGetDevice(&deviceId);
    mcDeviceProp_t dprops;
    mcGetDeviceProperties(&dprops, deviceId);
    return dprops.multiProcessorCount;
}

void set_params_fprop(Flash_fwd_params &params,

                      const size_t b, const size_t seqlen_q, const size_t seqlen_k, const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded, const size_t h, const size_t h_k, const size_t d,
                      const size_t d_rounded,

                      Tensor_t q, Tensor_t k, Tensor_t v, Tensor_t out, void *cu_seqlens_q_d, void *cu_seqlens_k_d,
                      void *seqused_k, void *p_d, void *softmax_lse_d, float p_dropout, float softmax_scale,
                      int window_size_left, int window_size_right, bool seqlenq_ngroups_swapped = false) {
    memset(&params, 0, sizeof(params));

    params.is_bf16 = (get_tensor_dtype(q) == MCFLASHATTN_DATATYPE_BF16);

    params.q_ptr = get_tensor_data(q);
    params.k_ptr = get_tensor_data(k);
    params.v_ptr = get_tensor_data(v);

    params.q_row_stride = get_tensor_stride(q, -3);
    params.k_row_stride = get_tensor_stride(k, -3);
    params.v_row_stride = get_tensor_stride(v, -3);
    params.q_head_stride = get_tensor_stride(q, -2);
    params.k_head_stride = get_tensor_stride(k, -2);
    params.v_head_stride = get_tensor_stride(v, -2);

    params.o_ptr = get_tensor_data(out);

    params.o_row_stride = get_tensor_stride(out, -3);
    params.o_head_stride = get_tensor_stride(out, -2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = get_tensor_stride(q, 0);
        params.k_batch_stride = get_tensor_stride(k, 0);
        params.v_batch_stride = get_tensor_stride(v, 0);
        params.o_batch_stride = get_tensor_stride(out, 0);
        if (seqlenq_ngroups_swapped) {
            params.q_batch_stride *= seqlen_q;
            params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    params.p_ptr = p_d;

    params.softmax_lse_ptr = softmax_lse_d;

    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.p_dropout = 1.f - p_dropout;

    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) {
        window_size_left = seqlen_k;
    }
    if (window_size_left >= 0 && window_size_right < 0) {
        window_size_right = seqlen_k;
    }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;

    params.num_splits = 1;
}

void set_params_dgrad(Flash_bwd_params &params,

                      const size_t b, const size_t seqlen_q, const size_t seqlen_k, const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded, const size_t h, const size_t h_k, const size_t d,
                      const size_t d_rounded,

                      const Tensor_t q, const Tensor_t k, const Tensor_t v, const Tensor_t out, const Tensor_t dout,
                      Tensor_t dq, Tensor_t dk, Tensor_t dv, void *cu_seqlens_q_d, void *cu_seqlens_k_d,
                      void *dq_accum_d, void *dk_accum_d, void *dv_accum_d, void *softmax_lse_d, void *dsoftmax_sum_d,
                      float p_dropout, float softmax_scale, int window_size_left, int window_size_right,
                      bool deterministic) {
    set_params_fprop(params, b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded, q, k, v,
                     out, cu_seqlens_q_d, cu_seqlens_k_d, nullptr, nullptr, softmax_lse_d, p_dropout, softmax_scale,
                     window_size_left, window_size_right);

    params.do_ptr = get_tensor_data(dout);
    params.do_row_stride = get_tensor_stride(dout, -3);
    params.do_head_stride = get_tensor_stride(dout, -2);
    params.dq_ptr = get_tensor_data(dq);
    params.dk_ptr = get_tensor_data(dk);
    params.dv_ptr = get_tensor_data(dv);
    params.dq_row_stride = get_tensor_stride(dq, -3);
    params.dk_row_stride = get_tensor_stride(dk, -3);
    params.dv_row_stride = get_tensor_stride(dv, -3);
    params.dq_head_stride = get_tensor_stride(dq, -2);
    params.dk_head_stride = get_tensor_stride(dk, -2);
    params.dv_head_stride = get_tensor_stride(dv, -2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = get_tensor_stride(dout, 0);
        params.dq_batch_stride = get_tensor_stride(dq, 0);
        params.dk_batch_stride = get_tensor_stride(dk, 0);
        params.dv_batch_stride = get_tensor_stride(dv, 0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}

inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});

    if (max_splits < 64 || batch_nheads_mblocks / 64 > 10) {
        return 1;
    }
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);

            if (eff > max_efficiency) {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 2; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            continue;
        }
        if (efficiency[num_splits - 1] >= 0.96 * max_efficiency) {
            return num_splits;
        }
    }
    return 1;
}

void set_params_splitkv(Flash_fwd_params &params, const int batch_size, const int num_heads, const int head_size,
                        const int max_seqlen_k, const int max_seqlen_q, const int head_size_rounded,
                        const float p_dropout, const int num_splits, const int process_count,
                        InternalTensor::DataType data_type) {
    const int block_n = 32;
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;

    const int num_m_blocks = (max_seqlen_q + 32 - 1) / 32;
    params.num_splits = num_splits;
    if (p_dropout == 0.0f) {
        if (num_splits < 1) {
            params.num_splits = 1;
        }
        if (params.num_splits > 1) {
            CHECK_MSG(false, "params.num_splits > 1 is not support in cpai for now");
        }
    }
}

void set_params_alibi(Flash_fwd_params &params, Tensor_t alibi_slopes, int batch_size, int num_heads) {
#ifdef FLASHATTENTION_DISABLE_ALIBI

    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes != NULL) {
        auto alibi_slopes_ = alibi_slopes;

        params.alibi_slopes_ptr = get_tensor_data(alibi_slopes_);
        params.alibi_slopes_batch_stride =
            get_tensor_dims(alibi_slopes_) == 2 ? get_tensor_stride(alibi_slopes_, 0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

std::vector<int64_t> get_attn_mask_stride(Tensor_t &attn_mask, std::vector<int64_t> &score_shape) {
    auto attn_mask_dim = get_tensor_dims(attn_mask);

    CHECK_MSG(score_shape.size() == 4, "score_shape must be 4-dim");
    CHECK_MSG(attn_mask_dim <= score_shape.size(), "attn_mask should have dim less than score_mask");

    int64_t accum_stride = 1;
    std::vector<int64_t> mask_stride = {0, 0, 0, 0};

    bool is_shape_valid = false;
    for (int i = 0; i < attn_mask_dim; i++) {
        int64_t cur_dim_size = get_tensor_size(attn_mask, (attn_mask_dim - 1 - i));

        if (score_shape[3 - i] == cur_dim_size) {
            is_shape_valid = true;
            mask_stride[3 - i] = accum_stride;
            accum_stride *= cur_dim_size;

        } else if (cur_dim_size == 1) {
            is_shape_valid = true;
            mask_stride[3 - i] = 0;

        } else {
            is_shape_valid = false;
            break;
        }
    }

    CHECK_MSG(is_shape_valid,
              "attn_mask must have shape [bs/1, hdim/1, q/1, k/1] or [hdim/1, q/1, k/1] or [q/1, k/1] or [k/1]");

    return mask_stride;
}

void set_params_attn_mask(Flash_fwd_params &params, Tensor_t attn_mask, int batch_size, int num_heads, int seqlen_q,
                          int seqlen_k) {
    if (attn_mask == nullptr) {
        params.has_attn_mask = false;
    } else {
        auto attn_mask_ = attn_mask;

        std::vector<int64_t> score_shape = {batch_size, num_heads, seqlen_q, seqlen_k};
        auto mask_stride = get_attn_mask_stride(attn_mask_, score_shape);

        params.has_attn_mask = true;
        params.attn_mask_ptr = get_tensor_data(attn_mask_);
        params.attn_mask_batch_stride = mask_stride[0];
        params.attn_mask_hdim_stride = mask_stride[1];
        params.attn_mask_row_stride = mask_stride[2];
        params.attn_mask_col_stride = mask_stride[3];
    }
}

#ifdef __cplusplus
extern "C" {
#endif

mcflashattnStatus_t mha_fwd_inference(int64_t batch_size, int64_t seqlen_q, int64_t num_heads, int64_t seqlen_k,
                                      int64_t num_heads_k, int64_t head_size, const Tensor_t q, const Tensor_t k,
                                      const Tensor_t v, const Tensor_t out, const Tensor_t alibi_slopes,
                                      const Tensor_t attn_mask, const float softmax_scale, bool is_causal,
                                      int window_size_left, int window_size_right, mcStream_t stream,
                                      mcflashattnExtendParameter_t extend_parameter_) {
    if (!check_tensor({q, k, v, out})) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    auto dtype = ((InternalTensor *)q->data)->dtype;
    if (!check_tensor_type({k, v}, dtype)) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    const float p_dropout = 0.0f;

    const int batch_size_ = get_tensor_size(q, 0);
    int seqlen_q_ = get_tensor_size(q, 1);
    int num_heads_ = get_tensor_size(q, 2);
    const int head_size_pad = get_tensor_size(q, 3);

    if (head_size_pad % 8 != 0) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    const int seqlen_k_ = get_tensor_size(k, 1);
    const int num_heads_k_ = get_tensor_size(k, 2);

    if (window_size_left >= seqlen_k_) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k_) {
        window_size_right = -1;
    }

    if (seqlen_q_ == 1 && !alibi_slopes) {
        is_causal = false;
    }
    if (is_causal) {
        window_size_right = 0;
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

    const int head_size_rounded = round_multiple(head_size_pad, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q_, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k_, 128);

    Flash_fwd_params params;

    set_params_fprop(params, batch_size_, seqlen_q_, seqlen_k_, seqlen_q_rounded, seqlen_k_rounded, num_heads_,
                     num_heads_k_, head_size, head_size_rounded, q, k, v, out, nullptr, nullptr, nullptr, nullptr,
                     nullptr, p_dropout, softmax_scale, window_size_left, window_size_right);

    params.num_splits = 1;

    params.rng_state_seed = 0;
    params.rng_state_offset = 0;

    set_params_alibi(params, alibi_slopes, batch_size_, num_heads_);

    set_params_attn_mask(params, attn_mask, batch_size_, num_heads_, seqlen_q_, seqlen_k_);

    if (seqlen_k_ > 0) {
        run_mha_fwd(params, stream);
    }

    return MCFLASHATTN_STATUS_SUCCESS;
}

mcflashattnStatus_t mha_varlen_fwd_inference(int64_t batch_size, int64_t total_q, int64_t num_heads, int64_t total_k,
                                             int64_t num_heads_k, int64_t head_size, const Tensor_t q, const Tensor_t k,
                                             const Tensor_t v, Tensor_t out, const Tensor_t cu_seqlens_q,
                                             const Tensor_t cu_seqlens_k, const Tensor_t seqused_k,
                                             const Tensor_t alibi_slopes, int max_seqlen_q, const int max_seqlen_k,
                                             const float softmax_scale,

                                             bool is_causal, int window_size_left, int window_size_right,
                                             mcStream_t stream, mcflashattnExtendParameter_t extend_parameter_) {
    if (!check_tensor({q, k, v, out})) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    auto dtype = ((InternalTensor *)q->data)->dtype;
    if (!check_tensor_type({k, v}, dtype)) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    const float p_dropout = 0.0f;

    const int batch_size_ = batch_size;
    int num_heads_ = get_tensor_size(q, 1);
    const int head_size_og_ = get_tensor_size(q, 2);
    const int total_k_ = get_tensor_size(k, 0);
    const int num_heads_k_ = get_tensor_size(k, 1);

    if (max_seqlen_q == 1 && alibi_slopes == nullptr) {
        is_causal = false;
    }
    if (is_causal) {
        window_size_right = 0;
    }

    void *cu_seqlens_q_d = get_tensor_data(cu_seqlens_q);

    const int total_q_ = get_tensor_size(q, 0);

    if (window_size_left >= max_seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= max_seqlen_k) {
        window_size_right = -1;
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_ = round_multiple(head_size_og_, 8);
    const int head_size_rounded = round_multiple(head_size_, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    const bool seqlenq_ngroups_swapped = false;

    Flash_fwd_params params;
    set_params_fprop(params, batch_size_, max_seqlen_q, max_seqlen_k, seqlen_q_rounded, seqlen_k_rounded, num_heads,
                     num_heads_k, head_size, head_size_rounded, q, k, v, out, cu_seqlens_q_d,
                     get_tensor_data(cu_seqlens_k), seqused_k != nullptr ? get_tensor_data(seqused_k) : nullptr,
                     nullptr, nullptr, p_dropout, softmax_scale, window_size_left, window_size_right,
                     seqlenq_ngroups_swapped);

    params.num_splits = 1;

    params.rng_state_seed = 0;
    params.rng_state_offset = 0;

    set_params_alibi(params, alibi_slopes, batch_size_, num_heads_);

    if (max_seqlen_k > 0) {
        run_mha_fwd(params, stream);
    }

    return MCFLASHATTN_STATUS_SUCCESS;
}

mcflashattnStatus_t mha_fwd_kvcache(const Tensor_t q, const Tensor_t kcache, const Tensor_t vcache, const Tensor_t k,
                                    const Tensor_t v, const Tensor_t seqlens_k, const Tensor_t rotary_cos,
                                    const Tensor_t rotary_sin, const Tensor_t cache_batch_idx,
                                    const Tensor_t block_table, const Tensor_t alibi_slopes, const Tensor_t softmax_lse,
                                    Tensor_t out, const float softmax_scale, bool is_causal, int window_size_left,
                                    int window_size_right, bool is_rotary_interleaved, mcStream_t stream,
                                    int num_splits, const Tensor_t softmax_lse_accum, const Tensor_t out_accum,
                                    mcflashattnExtendParameter_t extend_parameter_) {
    if (!check_tensor({q, kcache, vcache, out})) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    auto dtype = ((InternalTensor *)q->data)->dtype;
    if (!check_tensor_type({kcache, vcache}, dtype)) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    const bool paged_KV = (block_table != nullptr);

    Tensor_t block_table_;
    if (paged_KV) {
        if (cache_batch_idx != nullptr) {
            std::cerr << "Paged KVcache does not support cache_batch_idx" << std::endl;
            return MCFLASHATTN_STATUS_FAILED;
        }

        if (get_tensor_dtype(block_table) != MCFLASHATTN_DATATYPE_INT32) {
            std::cerr << "block_table must have dtype torch.int32" << std::endl;
            return MCFLASHATTN_STATUS_FAILED;
        }
        if (get_tensor_stride(block_table, 1) != 1) {
            std::cerr << "block_table must have contiguous last dimension" << std::endl;
            return MCFLASHATTN_STATUS_FAILED;
        }

        block_table_ = block_table;
    }

    const int batch_size_ = get_tensor_size(q, 0);
    int seqlen_q_ = get_tensor_size(q, 1);
    int num_heads_ = get_tensor_size(q, 2);
    const int head_size_og_ = get_tensor_size(q, 3);

    const int max_num_blocks_per_seq = !paged_KV ? 0 : get_tensor_size(block_table_, 1);
    const int num_blocks = !paged_KV ? 0 : get_tensor_size(kcache, 0);
    const int page_block_size = !paged_KV ? 1 : get_tensor_size(kcache, 1);

    const int seqlen_k_ = !paged_KV ? get_tensor_size(kcache, 1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k_ = get_tensor_size(kcache, 2);
    const int batch_size_c = !paged_KV ? get_tensor_size(kcache, 0) : batch_size_;

    if (seqlen_q_ == 1 && alibi_slopes == nullptr) {
        is_causal = false;
    }
    if (is_causal) {
        window_size_right = 0;
    }

    if (window_size_left >= seqlen_k_) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k_) {
        window_size_right = -1;
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_ = round_multiple(head_size_og_, 8);
    const int head_size_rounded = round_multiple(head_size_, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q_, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k_, 128);

    Flash_fwd_params params;
    set_params_fprop(params, batch_size_, seqlen_q_, seqlen_k_, seqlen_q_rounded, seqlen_k_rounded, num_heads_,
                     num_heads_k_, head_size_, head_size_rounded, q, kcache, vcache, out, nullptr, nullptr, nullptr,
                     nullptr,

                     get_tensor_data(softmax_lse), 0.f, softmax_scale, window_size_left, window_size_right);

    Tensor_t k_ = nullptr;
    Tensor_t v_ = nullptr;
    if (k != nullptr) {
        k_ = k;
        v_ = v;
        if (!check_tensor({k_, v_})) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;
        if (!check_tensor_type({k_, v_}, dtype)) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

        int seqlen_knew = get_tensor_size(k_, 1);

        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = get_tensor_data(k_);
        params.vnew_ptr = get_tensor_data(v_);

        params.knew_batch_stride = get_tensor_stride(k_, 0);
        params.vnew_batch_stride = get_tensor_stride(v_, 0);
        params.knew_row_stride = get_tensor_stride(k_, 1);
        params.vnew_row_stride = get_tensor_stride(k_, 1);
        params.knew_head_stride = get_tensor_stride(k_, 2);
        params.vnew_head_stride = get_tensor_stride(k_, 2);
    }

    if (seqlens_k != nullptr) {
        auto seqlens_k_ = seqlens_k;
        params.cu_seqlens_k = static_cast<int *>(get_tensor_data(seqlens_k_));
    }

    params.is_seqlens_k_cumulative = !(seqlens_k != nullptr);

    if (rotary_cos != nullptr) {
        auto rotary_cos_ = rotary_cos;
        params.rotary_dim = get_tensor_size(rotary_cos_, 1) * 2;
        const int seqlen_ro = get_tensor_size(rotary_cos_, 0);

        auto rotary_sin_ = rotary_sin;
        params.rotary_cos_ptr = get_tensor_data(rotary_cos_);
        params.rotary_sin_ptr = get_tensor_data(rotary_sin_);
        params.is_rotary_interleaved = is_rotary_interleaved;
    } else {
        params.rotary_dim = 0;
    }

    if (cache_batch_idx != nullptr) {
        auto cache_batch_idx_ = cache_batch_idx;
        params.cache_batch_idx = reinterpret_cast<int *>(get_tensor_data(cache_batch_idx_));
    }

    if (num_splits < 1) return MCFLASHATTN_STATUS_ILLEGAL_NUM_SPLITS;
    if (num_splits > 128) return MCFLASHATTN_STATUS_ILLEGAL_NUM_SPLITS;
    if (num_splits != 1) {
        if (softmax_lse_accum == nullptr || get_tensor_data(softmax_lse_accum) == nullptr)
            return MCFLASHATTN_STATUS_ILLEGAL_SPLIT_PARA;
        if (out_accum == nullptr || get_tensor_data(out_accum) == nullptr) return MCFLASHATTN_STATUS_ILLEGAL_SPLIT_PARA;
    }

    params.num_splits = num_splits;

    if (params.num_splits > 1) {
        params.softmax_lseaccum_ptr = get_tensor_data(softmax_lse_accum);
        params.oaccum_ptr = get_tensor_data(out_accum);
    }

    if (paged_KV) {
        params.block_table = reinterpret_cast<int *>(get_tensor_data(block_table_));
        params.block_table_batch_stride = get_tensor_stride(block_table_, 0);
    }
    params.page_block_size = page_block_size;

    set_params_alibi(params, alibi_slopes, batch_size_, num_heads_);

    run_mha_fwd(params, stream, k != nullptr || cache_batch_idx != nullptr || paged_KV);

    return MCFLASHATTN_STATUS_SUCCESS;
}

/*
 *@attn_mask support [batch_size or 1, num_heads or 1, seqlen_q or 1, seqlen_k or 1]
 *                                     [num_heads or 1, seqlen_q or 1, seqlen_k or 1]
 *                                                     [seqlen_q or 1, seqlen_k or 1]
 *                                                                    [seqlen_k or 1]
 */
mcflashattnStatus_t mha_fwd(int64_t batch_size, int64_t seqlen_q, int64_t num_heads_q, int64_t seqlen_k,
                            int64_t num_heads_k, int64_t head_size_og, const Tensor_t q, const Tensor_t k,
                            const Tensor_t v, Tensor_t out, const Tensor_t alibi_slopes, const Tensor_t attn_mask,
                            const Tensor_t softmax_lse, const Tensor_t p, const Tensor_t rng_state,
                            const float p_dropout, const float softmax_scale, bool is_causal, int window_size_left,
                            int window_size_right, mcStream_t stream, mcflashattnExtendParameter_t extend_parameter_) {
    if (!check_tensor({q, k, v, out})) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    auto dtype = ((InternalTensor *)q->data)->dtype;
    if (!check_tensor_type({k, v}, dtype)) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    const int batch_size_ = get_tensor_size(q, 0);
    int seqlen_q_ = get_tensor_size(q, 1);
    int num_heads_ = get_tensor_size(q, 2);
    const int head_size_pad = get_tensor_size(q, 3);

    if (head_size_pad % 8 != 0) return MCFLASHATTN_STATUS_FAILED;

    const int seqlen_k_ = get_tensor_size(k, 1);
    const int num_heads_k_ = get_tensor_size(k, 2);

    if (window_size_left >= seqlen_k_) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k_) {
        window_size_right = -1;
    }

    if (seqlen_q_ == 1 && !alibi_slopes) {
        is_causal = false;
    }
    if (is_causal) {
        window_size_right = 0;
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

    const int head_size_rounded = round_multiple(head_size_pad, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q_, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k_, 128);

    Flash_fwd_params params;
    set_params_fprop(params, batch_size_, seqlen_q_, seqlen_k_, seqlen_q_rounded, seqlen_k_rounded, num_heads_,
                     num_heads_k_, head_size_og, head_size_rounded, q, k, v, out, nullptr, nullptr, nullptr,
                     (p != NULL) ? get_tensor_data(p) : nullptr,
                     (softmax_lse != NULL) ? get_tensor_data(softmax_lse) : nullptr, p_dropout, softmax_scale,
                     window_size_left, window_size_right);

    if (p_dropout > 0.0) {
        CHECK_MSG(rng_state != NULL, "when use dropout, must input rng_state");
        CHECK_MSG(check_tensor({rng_state}), "rng_state should be continues at last-dim");
        CHECK_MSG(check_tensor_type({rng_state}, InternalTensor::DataType::INT64), "rng_state should have dtype int64");
        CHECK_MSG(check_tensor_shape({rng_state}, {2}), "rng_state should have shape {2}");

        auto rng_state_p = static_cast<int64_t *>(get_tensor_data(rng_state));
        params.rng_state_seed = rng_state_p[0];
        params.rng_state_offset = rng_state_p[1];
    }

    set_params_alibi(params, alibi_slopes, batch_size_, num_heads_);

    set_params_attn_mask(params, attn_mask, batch_size_, num_heads_, seqlen_q_, seqlen_k_);

    if (seqlen_k_ > 0) {
        run_mha_fwd(params, stream);
    }

    return MCFLASHATTN_STATUS_SUCCESS;
}

mcflashattnStatus_t mha_varlen_fwd(int64_t batch_size, int64_t total_q, int64_t num_heads_q, int64_t total_k,
                                   int64_t num_heads_k, int64_t head_size_og, const Tensor_t q, const Tensor_t k,
                                   const Tensor_t v, Tensor_t out, const Tensor_t cu_seqlens_q,
                                   const Tensor_t cu_seqlens_k, const Tensor_t seqused_k, const Tensor_t alibi_slopes,
                                   const Tensor_t softmax_lse, const Tensor_t p, const Tensor_t rng_state,
                                   int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
                                   const float softmax_scale,

                                   bool is_causal, int window_size_left, int window_size_right, mcStream_t stream,
                                   mcflashattnExtendParameter_t extend_parameter_) {
    if (!check_tensor({q, k, v, out, cu_seqlens_q, cu_seqlens_k})) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    auto dtype = ((InternalTensor *)q->data)->dtype;
    if (!check_tensor_type({k, v}, dtype)) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    if (!check_tensor_type({cu_seqlens_q, cu_seqlens_k}, InternalTensor::DataType::INT32))
        return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    const int batch_size_ = batch_size;
    int num_heads_ = get_tensor_size(q, 1);
    const int head_size_og_ = get_tensor_size(q, 2);
    const int total_k_ = get_tensor_size(k, 0);
    const int num_heads_k_ = get_tensor_size(k, 1);

    if (max_seqlen_q == 1 && alibi_slopes == nullptr) {
        is_causal = false;
    }
    if (is_causal) {
        window_size_right = 0;
    }

    const int total_q_ = get_tensor_size(q, 0);

    if (window_size_left >= max_seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= max_seqlen_k) {
        window_size_right = -1;
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_ = round_multiple(head_size_og_, 8);
    const int head_size_rounded = round_multiple(head_size_, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    Flash_fwd_params params;
    ;

    set_params_fprop(
        params, batch_size_, max_seqlen_q, max_seqlen_k, seqlen_q_rounded, seqlen_k_rounded, num_heads_, num_heads_k_,
        head_size_, head_size_rounded, q, k, v, out, cu_seqlens_q != nullptr ? get_tensor_data(cu_seqlens_q) : nullptr,
        cu_seqlens_k != nullptr ? get_tensor_data(cu_seqlens_k) : nullptr,
        seqused_k != nullptr ? get_tensor_data(seqused_k) : nullptr, p != nullptr ? get_tensor_data(p) : nullptr,
        softmax_lse != nullptr ? get_tensor_data(softmax_lse) : nullptr, p_dropout, softmax_scale, window_size_left,
        window_size_right);

    if (p_dropout > 0.0) {
        CHECK_MSG(rng_state != NULL, "when use dropout, must input rng_state");
        CHECK_MSG(check_tensor({rng_state}), "rng_state should be continues at last-dim");
        CHECK_MSG(check_tensor_type({rng_state}, InternalTensor::DataType::INT64), "rng_state should have dtype int64");
        CHECK_MSG(check_tensor_shape({rng_state}, {2}), "rng_state should have shape {2}");

        auto rng_state_p = static_cast<int64_t *>(get_tensor_data(rng_state));
        params.rng_state_seed = rng_state_p[0];
        params.rng_state_offset = rng_state_p[1];
    }

    set_params_alibi(params, alibi_slopes, batch_size_, num_heads_);

    if (max_seqlen_k > 0) {
        run_mha_fwd(params, stream);
    }

    return MCFLASHATTN_STATUS_SUCCESS;
}

/*
 *@attn_mask support [batch_size or 1, num_heads or 1, seqlen_q or 1, seqlen_k or 1]
 *                                     [num_heads or 1, seqlen_q or 1, seqlen_k or 1]
 *                                                     [seqlen_q or 1, seqlen_k or 1]
 *                                                                    [seqlen_k or 1]
 */
mcflashattnStatus_t mha_bwd(int64_t batch_size, int64_t seqlen_q, int64_t num_heads_q, int64_t seqlen_k,
                            int64_t num_heads_k, int64_t head_size_og, const Tensor_t dout, const Tensor_t q,
                            const Tensor_t k, const Tensor_t v, Tensor_t out, const Tensor_t softmax_d,
                            const Tensor_t softmax_lse, const Tensor_t dq, const Tensor_t dk, const Tensor_t dv,
                            const Tensor_t dq_accum, const Tensor_t alibi_slopes, const Tensor_t attn_mask,
                            const Tensor_t rng_state, const float p_dropout, const float softmax_scale,
                            const bool is_causal, int window_size_left, int window_size_right, const bool deterministic,
                            mcStream_t stream, mcflashattnExtendParameter_t extend_parameter_) {
    if (!check_tensor({q, k, v, out, dout, softmax_lse, softmax_d, dq_accum})) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    auto dtype = ((InternalTensor *)q->data)->dtype;
    if (!check_tensor_type({k, v, out, dout}, dtype)) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    const int batch_size_ = get_tensor_size(q, 0);
    const int seqlen_q_ = get_tensor_size(q, 1);
    const int num_heads_ = get_tensor_size(q, 2);
    const int head_size_og_ = get_tensor_size(dout, 3);
    const int head_size_ = get_tensor_size(q, 3);
    const int seqlen_k_ = get_tensor_size(k, 1);
    const int num_heads_k_ = get_tensor_size(k, 2);

    CHECK_MSG(batch_size_ > 0, "batch size must be positive");
    CHECK_MSG(head_size_ % 8 == 0, "head_size should be a multiple of 8");
    CHECK_MSG(head_size_ <= 256, "FlashAttention backward only supports head dimension at most 256");
    CHECK_MSG(num_heads_ % num_heads_k_ == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size_, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q_, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k_, 128);

    CHECK_MSG(head_size_ == round_multiple(head_size_og_, 8),
              "head_size must be head_size_og rounded to a multiple of 8");

    if (is_causal) {
        window_size_right = 0;
    }
    if (window_size_left >= seqlen_k_) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k_) {
        window_size_right = -1;
    }

    CHECK_MSG(check_tensor_shape({q}, {batch_size_, seqlen_q_, num_heads_, head_size_}), "q shape invalid");
    CHECK_MSG(check_tensor_shape({k}, {batch_size_, seqlen_k_, num_heads_k_, head_size_}), "k shape invalid");
    CHECK_MSG(check_tensor_shape({v}, {batch_size_, seqlen_k_, num_heads_k_, head_size_}), "v shape invalid");
    CHECK_MSG(check_tensor_shape({out}, {batch_size_, seqlen_q_, num_heads_, head_size_}), "out shape invalid");
    CHECK_MSG(check_tensor_shape({dout}, {batch_size_, seqlen_q_, num_heads_, head_size_og_}), "dout shape invalid");

    Flash_bwd_params params;

    set_params_dgrad(params, batch_size_, seqlen_q_, seqlen_k_, seqlen_q_rounded, seqlen_k_rounded, num_heads_,
                     num_heads_k_, head_size_, head_size_rounded, q, k, v, out, dout, dq, dk, dv, nullptr, nullptr,
                     get_tensor_data(dq_accum), nullptr, nullptr, get_tensor_data(softmax_lse),
                     get_tensor_data(softmax_d), p_dropout, softmax_scale, window_size_left, window_size_right,
                     deterministic);
    params.packed_seqlen = 0;
    params.dq_accum_split_stride = !deterministic ? 0 : get_tensor_stride(dq_accum, 0);

    if (p_dropout > 0.0) {
        CHECK_MSG(rng_state != NULL, "when use dropout, must input rng_state");
        CHECK_MSG(check_tensor({rng_state}), "rng_state should be continues at last-dim");
        CHECK_MSG(check_tensor_type({rng_state}, InternalTensor::DataType::INT64), "rng_state should have dtype int64");
        CHECK_MSG(check_tensor_shape({rng_state}, {2}), "rng_state should have shape {2}");

        auto rng_state_p = static_cast<int64_t *>(get_tensor_data(rng_state));
        params.rng_state_seed = rng_state_p[0];
        params.rng_state_offset = rng_state_p[1];
    }

    set_params_alibi(params, alibi_slopes, batch_size_, num_heads_);

    set_params_attn_mask(params, attn_mask, batch_size_, num_heads_, seqlen_q_, seqlen_k_);

    if (seqlen_q_ > 0) {
        run_mha_bwd(params, stream);
    }

    return MCFLASHATTN_STATUS_SUCCESS;
}

mcflashattnStatus_t mha_varlen_bwd(int64_t batch_size, int64_t total_q, int64_t num_heads_q, int64_t total_k,
                                   int64_t num_heads_k, int64_t head_size_og, const Tensor_t dout, const Tensor_t q,
                                   const Tensor_t k, const Tensor_t v, Tensor_t out, const Tensor_t softmax_d,
                                   const Tensor_t softmax_lse, const Tensor_t dq, const Tensor_t dk, const Tensor_t dv,
                                   const Tensor_t dq_accum, const Tensor_t cu_seqlens_q, const Tensor_t cu_seqlens_k,
                                   const Tensor_t alibi_slopes, const Tensor_t rng_state, const int max_seqlen_q,
                                   const int max_seqlen_k, const float p_dropout, const float softmax_scale,

                                   const bool is_causal, int window_size_left, int window_size_right,
                                   const bool deterministic, mcStream_t stream,
                                   mcflashattnExtendParameter_t extend_parameter_) {
    if (!check_tensor({q, k, v, out, dout, softmax_lse, softmax_d, dq_accum})) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    if (is_causal) {
        window_size_right = 0;
    }

    auto dtype = ((InternalTensor *)q->data)->dtype;
    if (!check_tensor_type({k, v, out, dout}, dtype)) return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    if (!check_tensor_type({cu_seqlens_q, cu_seqlens_k}, InternalTensor::DataType::INT32))
        return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;

    const int total_q_ = get_tensor_size(q, 0);
    const int batch_size_ = batch_size;
    const int num_heads_ = get_tensor_size(q, 1);
    const int head_size_og_ = get_tensor_size(dout, 2);
    const int head_size_ = get_tensor_size(q, 2);
    const int total_k_ = get_tensor_size(k, 0);
    const int num_heads_k_ = get_tensor_size(k, 1);

    CHECK_MSG(batch_size_ > 0, "batch size must be positive");
    CHECK_MSG(head_size_ % 8 == 0, "head_size should be a multiple of 8");
    CHECK_MSG(head_size_ <= 256, "FlashAttention backward only supports head dimension at most 256");
    CHECK_MSG(num_heads_ % num_heads_k_ == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size_, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    CHECK_MSG(head_size_ == round_multiple(head_size_og_, 8),
              "head_size must be head_size_og rounded to a multiple of 8");

    if (is_causal) {
        window_size_right = 0;
    }
    if (window_size_left >= max_seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= max_seqlen_k) {
        window_size_right = -1;
    }

    CHECK_MSG(check_tensor_shape({q}, {total_q_, num_heads_, head_size_}), "q shape invalid");
    CHECK_MSG(check_tensor_shape({k}, {total_k_, num_heads_k_, head_size_}), "k shape invalid");
    CHECK_MSG(check_tensor_shape({v}, {total_k_, num_heads_k_, head_size_}), "v shape invalid");
    CHECK_MSG(check_tensor_shape({out}, {total_q_, num_heads_, head_size_}), "out shape invalid");
    CHECK_MSG(check_tensor_shape({dout}, {total_q_, num_heads_, head_size_og_}), "dout shape invalid");
    CHECK_MSG(check_tensor_shape({cu_seqlens_q}, {batch_size_ + 1}), "cu_seqlens_q shape invalid");
    CHECK_MSG(check_tensor_shape({cu_seqlens_k}, {batch_size_ + 1}), "cu_seqlens_k shape invalid");

    Flash_bwd_params params;

    set_params_dgrad(params, batch_size_, max_seqlen_q, max_seqlen_k, seqlen_q_rounded, seqlen_k_rounded, num_heads_,
                     num_heads_k_, head_size_, head_size_rounded, q, k, v, out, dout, dq, dk, dv,
                     get_tensor_data(cu_seqlens_q), get_tensor_data(cu_seqlens_k), get_tensor_data(dq_accum), nullptr,
                     nullptr, get_tensor_data(softmax_lse), get_tensor_data(softmax_d), p_dropout, softmax_scale,
                     window_size_left, window_size_right, deterministic);
    params.packed_seqlen = total_q_;
    params.dq_accum_split_stride = !deterministic ? 0 : get_tensor_stride(dq_accum, 0);

    auto launch = &run_mha_bwd;

    if (p_dropout > 0.0) {
        CHECK_MSG(rng_state != NULL, "when use dropout, must input rng_state");
        CHECK_MSG(check_tensor({rng_state}), "rng_state should be continues at last-dim");
        CHECK_MSG(check_tensor_type({rng_state}, InternalTensor::DataType::INT64), "rng_state should have dtype int64");
        CHECK_MSG(check_tensor_shape({rng_state}, {2}), "rng_state should have shape {2}");

        auto rng_state_p = static_cast<int64_t *>(get_tensor_data(rng_state));
        params.rng_state_seed = rng_state_p[0];
        params.rng_state_offset = rng_state_p[1];
    }

    set_params_alibi(params, alibi_slopes, batch_size_, num_heads_);

    if (max_seqlen_q > 0) {
        run_mha_bwd(params, stream);
    }

    return MCFLASHATTN_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif