

#include "flash_parameter_utils.h"

using namespace mcFlashAttn;

void set_params_fprop(Flash_fwd_params &params,
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
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped,
                      bool unpadded_lse,
                      int d_v,
                      int d_v_rounded) {

    // Reset the parameters
    // do not reset, as we have default construct for params
    memset(&params, 0, sizeof(params));

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
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
    params.d_value = d_v;
    params.d_value_rounded = d_v_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif

    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else {
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f -p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;
    // only for varlen
    params.total_q = 0;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_dgrad(Flash_bwd_params &params,
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
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
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
                      const bool unpadded_lse,
                      const float softcap,
                      int d_v,
                      int d_v_rounded) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     /*seqlenq_ngroups_swapped*/false,
                     unpadded_lse,
                     d_v,
                     d_v_rounded);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}

void set_params_alibi(Flash_fwd_params &params, c10::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

void set_params_rng_state(Flash_fwd_params &params, at::Tensor &rng_state) {
    TORCH_CHECK(rng_state.dtype() == torch::kInt64, "rng_state tensor must have dtype Int64");
    TORCH_CHECK(rng_state.stride(-1) == 1, "rng_state tensor must have contiguous last dimension");
    CHECK_HOST(rng_state);
    CHECK_SHAPE(rng_state, 2);

    uint64_t *rng_state_p;
    rng_state_p = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
    params.rng_state_seed = rng_state_p[0];
    params.rng_state_offset = rng_state_p[1];
}

void get_philox_state(c10::optional<at::Generator> &gen_, at::Tensor &rng_state, int64_t counter_offset) {
    TORCH_CHECK(rng_state.dtype() == torch::kInt64, "rng_state tensor must have dtype Int64");
    TORCH_CHECK(rng_state.stride(-1) == 1, "rng_state tensor must have contiguous last dimension");
    CHECK_SHAPE(rng_state, 2);

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    // auto gen = at::cuda::detail::getDefaultCUDAGenerator().get<at::CUDAGeneratorImpl>();

    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);

    auto philox_args = gen->philox_cuda_state(counter_offset);
    auto seeds = at::cuda::philox::unpack(philox_args);

    uint64_t *rng_state_p;
    rng_state_p = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
    rng_state_p[0] = std::get<0>(seeds);
    rng_state_p[1] = std::get<1>(seeds);
}

// score ->[bs,      head_num,      q,      k  ]
// mask -> [bs_mask, head_num_mask, q_mask, k_mask]
// Mask shape should satisfy these rules
// 1. bs % bs_mask == 0
// 2. head_num % head_num_mask == 0
// 3. q_mask == 1 or q_mask == q
// 4. k_mask == 1 or k_mask == k or k_mask == (k + 3) / 4 * 4 (align k to multiples of 4)
std::vector<int64_t>
get_attn_mask_stride(std::vector<int64_t> &mask_shape, std::vector<int64_t> &score_shape) {
    TORCH_CHECK(score_shape.size() == 4, "score_shape must be 4-dim");
    TORCH_CHECK(1 <= mask_shape.size() && mask_shape.size() <= score_shape.size(), "attn_mask should have dim less than score_mask and at least have 1 dim");

    int64_t accum_stride = 1; // dim-0 * ... * dim-n ,should be stride of dim-(n+1)
    std::vector<int64_t> mask_stride = {0, 0, 0, 0};

    bool is_shape_valid = true;
    for(int i = mask_shape.size() - 1; i >= 0; i--) {
        if (i <= 1) {
            // batch and nheads dim
            if (mask_shape[i] == 1) {
                mask_stride[i] = 0;
            } else if (score_shape[i] % mask_shape[i] == 0) {
                mask_stride[i] = accum_stride;
                accum_stride *= mask_shape[i];
            } else {
                is_shape_valid = false;
                break;
            }
        } else {
            // seqlenq and seqlenk dim
            if (mask_shape[i] == 1) {
                mask_stride[i] = 0;
            } else if (score_shape[i] == mask_shape[i]) {
                mask_stride[i] = accum_stride;
                accum_stride *= mask_shape[i];
            } else {
                is_shape_valid = false;
                break;
            }
        }
    }

    TORCH_CHECK(is_shape_valid, "the attn_mask shape is not support, please check it!");

    return mask_stride;
}

void set_params_attn_mask(Flash_fwd_params &params, bool is_causal, at::Tensor &attn_mask,
                            int batch_size, int num_heads, int seqlen_q, int seqlen_k) {

    // full attn_mask 
    TORCH_CHECK(!is_causal, "when attn_mask is true, causal should be false");
    TORCH_CHECK((attn_mask.dtype() == torch::kBFloat16) || (attn_mask.dtype() == torch::kFloat16), "attn_mask must have dtype fp16/bf16");
    CHECK_DEVICE(attn_mask);
    TORCH_CHECK(attn_mask.stride(-1) == 1, "attn_mask tensor must have contiguous last dimension");

    // CHECK_SHAPE(attn_mask, seqlen_q, seqlen_k)
    auto attn_mask_shape = attn_mask.sizes();
    // the attn_mask last dim will be padded to multiples of 4
    int seqlen_k_rounded = (seqlen_k + 3) / 4 * 4;
    std::vector<int64_t> score_shape = {batch_size, num_heads, seqlen_q, seqlen_k_rounded};
    std::vector<int64_t> mask_shape = {1, 1, 1, 1};
    for (int i = attn_mask_shape.size() - 1, j = 3; i >= 0; i--, j--) {
        mask_shape[j] = attn_mask_shape[i];
    }
    auto mask_stride = get_attn_mask_stride(mask_shape, score_shape);

    params.has_attn_mask = true;
    params.attn_mask_ptr = attn_mask.data_ptr();
    params.attn_mask_batch_stride = mask_stride[0];
    params.attn_mask_nheads_stride = mask_stride[1];
    params.attn_mask_row_stride = mask_stride[2];
    params.attn_mask_col_stride = mask_stride[3];
    params.attn_mask_batch_shape = mask_shape[0];
    params.attn_mask_nheads_shape = mask_shape[1];
    params.attn_mask_row_shape = mask_shape[2];
    params.attn_mask_col_shape = mask_shape[3];
}
