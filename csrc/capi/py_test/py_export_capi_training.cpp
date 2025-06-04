#include "flash_attn.h"
#include "py_export_capi_utils.h"
#include "py_export_capi_training.h"

/*
 *@attn_mask_ support [batch_size or 1, num_heads or 1, seqlen_q or 1, seqlen_k or 1]
 *                                     [num_heads or 1, seqlen_q or 1, seqlen_k or 1]
 *                                                     [seqlen_q or 1, seqlen_k or 1]
 *                                                                    [seqlen_k or 1]
 */
std::vector<at::Tensor> mha_fwd_capi_test(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                          c10::optional<at::Tensor> &out_, c10::optional<at::Tensor> &alibi_slopes_,
                                          c10::optional<at::Tensor> &attn_mask_, const float p_dropout,
                                          const float softmax_scale, bool is_causal, int window_size_left,
                                          int window_size_right, const bool return_softmax,
                                          c10::optional<at::Generator> gen_) {
    auto dprops = at::cuda::getCurrentDeviceProperties();

    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k) {
        window_size_right = -1;
    }

    if (seqlen_q == 1 && !alibi_slopes_.has_value()) {
        is_causal = false;
    }
    if (is_causal) {
        window_size_right = 0;
    }

    const int seqlenq_ngroups_swapped = 0;
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) {
            out = torch::empty_like(q_padded);
        }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;

    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    }

    int64_t counter_offset = batch_size * num_heads * 32;
    auto rng_state = torch::empty({2}, torch::kInt64);

    if (p_dropout > 0.0) {
        get_philox_state(gen_, rng_state, counter_offset);
    }

    Tensor_t q_mcfa = convert_mcfa_tensor(q_padded);
    Tensor_t k_mcfa = convert_mcfa_tensor(k_padded);
    Tensor_t v_mcfa = convert_mcfa_tensor(v_padded);
    Tensor_t out_mcfa = convert_mcfa_tensor(out);
    Tensor_t alibi_slopes_mcfa = convert_mcfa_tensor(alibi_slopes_);
    Tensor_t attn_mask_mcfa = convert_mcfa_tensor(attn_mask_);
    Tensor_t softmax_lse_mcfa = convert_mcfa_tensor(softmax_lse);
    Tensor_t p_mcfa = convert_mcfa_tensor(p);
    Tensor_t rng_state_mcfa = convert_mcfa_tensor(rng_state);

    mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = mha_fwd(batch_size, seqlen_q, num_heads, seqlen_k, num_heads_k, head_size, q_mcfa, k_mcfa, v_mcfa,
                       out_mcfa, alibi_slopes_mcfa, attn_mask_mcfa, softmax_lse_mcfa, p_mcfa, rng_state_mcfa, p_dropout,
                       softmax_scale, is_causal, window_size_left, window_size_right, stream, NULL);
    if (ret != MCFLASHATTN_STATUS_SUCCESS) {
        std::cerr << "Call mha_fwd failed,ret = " << int(ret) << std::endl;
    }

    release_mcfa_tensor({q_mcfa, k_mcfa, v_mcfa, out_mcfa, alibi_slopes_mcfa, attn_mask_mcfa, softmax_lse_mcfa, p_mcfa,
                         rng_state_mcfa});

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) {
            out_.value().copy_(out);
        }
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        out_padded = out_padded.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        q_padded = q_padded.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
}

std::vector<at::Tensor> mha_varlen_fwd_capi_test(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                                 c10::optional<at::Tensor> &out_, const at::Tensor &cu_seqlens_q,
                                                 const at::Tensor &cu_seqlens_k, c10::optional<at::Tensor> &seqused_k,
                                                 c10::optional<at::Tensor> &alibi_slopes_, int max_seqlen_q,
                                                 const int max_seqlen_k, const float p_dropout,
                                                 const float softmax_scale, const bool zero_tensors, bool is_causal,
                                                 int window_size_left, int window_size_right, const bool return_softmax,
                                                 c10::optional<at::Generator> gen_) {
    auto dprops = at::cuda::getCurrentDeviceProperties();

    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);

    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) {
        is_causal = false;
    }
    if (is_causal) {
        window_size_right = 0;
    }

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    const int seqlenq_ngroups_swapped = 0;
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og})
                .transpose(1, 2)
                .reshape({batch_size * ngroups, num_heads_k, head_size_og});
        max_seqlen_q = ngroups;
        num_heads = num_heads_k;
        cu_seqlens_q_d = nullptr;
    }

    const int total_q = q.sizes()[0];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= max_seqlen_k) {
        window_size_right = -1;
    }

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    if (seqused_k.has_value()) {
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) {
            out = torch::empty_like(q_padded);
        }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;

    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {
            p.zero_();
        }
    }

    int64_t counter_offset = batch_size * num_heads * 32;
    auto rng_state = torch::empty({2}, torch::kInt64);

    if (p_dropout > 0.0) {
        get_philox_state(gen_, rng_state, counter_offset);
    }

    Tensor_t q_mcfa = convert_mcfa_tensor(q_padded);
    Tensor_t k_mcfa = convert_mcfa_tensor(k_padded);
    Tensor_t v_mcfa = convert_mcfa_tensor(v_padded);
    Tensor_t out_mcfa = convert_mcfa_tensor(out);
    Tensor_t cu_seqlens_q_mcfa = convert_mcfa_tensor(cu_seqlens_q);
    Tensor_t cu_seqlens_k_mcfa = convert_mcfa_tensor(cu_seqlens_k);
    Tensor_t seqused_k_mcfa = convert_mcfa_tensor(seqused_k);
    Tensor_t alibi_slopes_mcfa = convert_mcfa_tensor(alibi_slopes_);
    Tensor_t softmax_lse_mcfa = convert_mcfa_tensor(softmax_lse);
    Tensor_t p_mcfa = convert_mcfa_tensor(p);
    Tensor_t rng_state_mcfa = convert_mcfa_tensor(rng_state);

    mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = mha_varlen_fwd(batch_size, total_q, num_heads, total_k, num_heads_k, head_size, q_mcfa, k_mcfa, v_mcfa,
                              out_mcfa, cu_seqlens_q_mcfa, cu_seqlens_k_mcfa, seqused_k_mcfa, alibi_slopes_mcfa,
                              softmax_lse_mcfa, p_mcfa, rng_state_mcfa, max_seqlen_q, max_seqlen_k, p_dropout,
                              softmax_scale, is_causal, window_size_left, window_size_right, stream, NULL);
    if (ret != MCFLASHATTN_STATUS_SUCCESS) {
        std::cerr << "Call mha_varlen_fwd failed,ret = " << int(ret) << std::endl;
    }

    release_mcfa_tensor({q_mcfa, k_mcfa, v_mcfa, out_mcfa, cu_seqlens_q_mcfa, cu_seqlens_k_mcfa, seqused_k_mcfa,
                         alibi_slopes_mcfa, softmax_lse_mcfa, p_mcfa, rng_state_mcfa});

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) {
            out_.value().copy_(out);
        }
    }

    if (seqlenq_ngroups_swapped) {
        int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size_og};
        int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size_og};
        out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
        out_padded = out_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
        q_padded = q_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * max_seqlen_q, 1});
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
}

std::vector<at::Tensor> mha_bwd_capi_test(const at::Tensor &dout, const at::Tensor &q, const at::Tensor &k,
                                          const at::Tensor &v, const at::Tensor &out, const at::Tensor &softmax_lse,
                                          c10::optional<at::Tensor> &dq_, c10::optional<at::Tensor> &dk_,
                                          c10::optional<at::Tensor> &dv_, c10::optional<at::Tensor> &alibi_slopes_,
                                          c10::optional<at::Tensor> &attn_mask_, const float p_dropout,
                                          const float softmax_scale, const bool is_causal, int window_size_left,
                                          int window_size_right, const bool deterministic,
                                          c10::optional<at::Generator> gen_, c10::optional<at::Tensor> &rng_state_) {
#ifdef FLASHATTENTION_DISABLE_BACKWARD
    TORCH_CHECK(false, "This flash attention build does not support backward.");
#endif
    if (is_causal) {
        window_size_right = 0;
    }
    auto dprops = at::cuda::getCurrentDeviceProperties();

    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    bool is_dropout = p_dropout > 0.0;

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(out);
    CHECK_DEVICE(dout);
    CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = dout.size(3);
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    if (head_size > 192) {
        TORCH_CHECK(is_sm80 || is_sm90, "FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800");
    }
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    TORCH_CHECK(head_size == round_multiple(head_size_og, 8),
                "head_size must be head_size_og rounded to a multiple of 8");

    if (window_size_left >= seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k) {
        window_size_right = -1;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded =
            torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    bool loop = true;

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    if (loop) {
        if (!deterministic) {
            dq_accum =
                torch::empty({batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (dprops->multiProcessorCount + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded},
                                    opts.dtype(at::kFloat));
        }
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {
        dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    int64_t counter_offset = batch_size * num_heads * 32;
    auto rng_state = torch::empty({2}, torch::kInt64);

    if (rng_state_.has_value()) {
        rng_state = rng_state_.value();
    } else if (is_dropout) {
        get_philox_state(gen_, rng_state, counter_offset);
    }

    Tensor_t q_mcfa = convert_mcfa_tensor(q);
    Tensor_t k_mcfa = convert_mcfa_tensor(k);
    Tensor_t v_mcfa = convert_mcfa_tensor(v);
    Tensor_t out_mcfa = convert_mcfa_tensor(out);
    Tensor_t dout_mcfa = convert_mcfa_tensor(dout_padded);
    Tensor_t dq_mcfa = convert_mcfa_tensor(dq);
    Tensor_t dk_mcfa = convert_mcfa_tensor(dk_expanded);
    Tensor_t dv_mcfa = convert_mcfa_tensor(dv_expanded);
    Tensor_t dq_accum_mcfa = convert_mcfa_tensor(dq_accum);
    Tensor_t softmax_lse_mcfa = convert_mcfa_tensor(softmax_lse);
    Tensor_t softmax_d_mcfa = convert_mcfa_tensor(softmax_d);
    Tensor_t alibi_slopes_mcfa = convert_mcfa_tensor(alibi_slopes_);
    Tensor_t attn_mask_mcfa = convert_mcfa_tensor(attn_mask_);
    Tensor_t rng_state_mcfa = convert_mcfa_tensor(rng_state);

    mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = mha_bwd(batch_size, seqlen_q, num_heads, seqlen_k, num_heads_k, head_size, dout_mcfa, q_mcfa, k_mcfa,
                       v_mcfa, out_mcfa, softmax_d_mcfa, softmax_lse_mcfa, dq_mcfa, dk_mcfa, dv_mcfa, dq_accum_mcfa,
                       alibi_slopes_mcfa, attn_mask_mcfa, rng_state_mcfa, p_dropout, softmax_scale, is_causal,
                       window_size_left, window_size_right, deterministic, stream, NULL);
    if (ret != MCFLASHATTN_STATUS_SUCCESS) {
        std::cerr << "Call mha_bwd failed,ret = " << int(ret) << std::endl;
    }

    release_mcfa_tensor({q_mcfa, k_mcfa, v_mcfa, out_mcfa, dout_mcfa, dq_mcfa, dk_mcfa, dv_mcfa, dq_accum_mcfa,
                         softmax_lse_mcfa, softmax_d_mcfa, alibi_slopes_mcfa, attn_mask_mcfa, rng_state_mcfa});

    if (num_heads_k != num_heads) {
        at::sum_out(
            dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(
            dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }
    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    }

    return {dq, dk, dv, softmax_d};
}

std::vector<at::Tensor> mha_varlen_bwd_capi_test(
    const at::Tensor &dout, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &out,
    const at::Tensor &softmax_lse, c10::optional<at::Tensor> &dq_, c10::optional<at::Tensor> &dk_,
    c10::optional<at::Tensor> &dv_, const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_k,
    c10::optional<at::Tensor> &alibi_slopes_, const int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, const bool is_causal, int window_size_left,
    int window_size_right, const bool deterministic, c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor> &rng_state_) {
#ifdef FLASHATTENTION_DISABLE_BACKWARD
    TORCH_CHECK(false, "This flash attention build does not support backward.");
#endif

    if (is_causal) {
        window_size_right = 0;
    }
    auto dprops = at::cuda::getCurrentDeviceProperties();

    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    bool is_dropout = p_dropout > 0.0;

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(out);
    CHECK_DEVICE(dout);
    CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_og = dout.size(2);
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    if (head_size > 192) {
        TORCH_CHECK(is_sm80 || is_sm90, "FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800");
    }
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    TORCH_CHECK(head_size == round_multiple(head_size_og, 8),
                "head_size must be head_size_og rounded to a multiple of 8");

    if (window_size_left >= max_seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= max_seqlen_k) {
        window_size_right = -1;
    }

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded =
            torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    bool loop = true;

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    if (loop) {
        if (!deterministic) {
            dq_accum = torch::empty({total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (dprops->multiProcessorCount + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, total_q + 128 * batch_size, num_heads, head_size_rounded},
                                    opts.dtype(at::kFloat));
        }
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {
        dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if (zero_tensors) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    int64_t counter_offset = batch_size * num_heads * 32;
    auto rng_state = torch::empty({2}, torch::kInt64);

    if (rng_state_.has_value()) {
        rng_state = rng_state_.value();
    } else if (is_dropout) {
        get_philox_state(gen_, rng_state, counter_offset);
    }

    Tensor_t q_mcfa = convert_mcfa_tensor(q);
    Tensor_t k_mcfa = convert_mcfa_tensor(k);
    Tensor_t v_mcfa = convert_mcfa_tensor(v);
    Tensor_t out_mcfa = convert_mcfa_tensor(out);
    Tensor_t dout_mcfa = convert_mcfa_tensor(dout_padded);
    Tensor_t softmax_lse_mcfa = convert_mcfa_tensor(softmax_lse);
    Tensor_t softmax_d_mcfa = convert_mcfa_tensor(softmax_d);
    Tensor_t dq_mcfa = convert_mcfa_tensor(dq);
    Tensor_t dk_mcfa = convert_mcfa_tensor(dk_expanded);
    Tensor_t dv_mcfa = convert_mcfa_tensor(dv_expanded);
    Tensor_t dq_accum_mcfa = convert_mcfa_tensor(dq_accum);
    Tensor_t cu_seqlens_q_mcfa = convert_mcfa_tensor(cu_seqlens_q);
    Tensor_t cu_seqlens_k_mcfa = convert_mcfa_tensor(cu_seqlens_k);
    Tensor_t alibi_slopes_mcfa = convert_mcfa_tensor(alibi_slopes_);
    Tensor_t rng_state_mcfa = convert_mcfa_tensor(rng_state);

    mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = mha_varlen_bwd(batch_size, total_q, num_heads, total_k, num_heads_k, head_size, dout_mcfa, q_mcfa,
                              k_mcfa, v_mcfa, out_mcfa, softmax_d_mcfa, softmax_lse_mcfa, dq_mcfa, dk_mcfa, dv_mcfa,
                              dq_accum_mcfa, cu_seqlens_q_mcfa, cu_seqlens_k_mcfa, alibi_slopes_mcfa, rng_state_mcfa,
                              max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale, is_causal, window_size_left,
                              window_size_right, deterministic, stream, NULL);
    if (ret != MCFLASHATTN_STATUS_SUCCESS) {
        std::cerr << "Call mha_varlen_bwd failed,ret = " << int(ret) << std::endl;
    }

    release_mcfa_tensor({q_mcfa, k_mcfa, v_mcfa, out_mcfa, dout_mcfa, softmax_lse_mcfa, softmax_d_mcfa, dq_mcfa,
                         dk_mcfa, dv_mcfa, dq_accum_mcfa, cu_seqlens_q_mcfa, cu_seqlens_k_mcfa, alibi_slopes_mcfa,
                         rng_state_mcfa});

    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
    }
    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    }

    return {dq, dk, dv, softmax_d};
}