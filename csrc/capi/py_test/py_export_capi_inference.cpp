#include "flash_attn.h"
#include "py_export_capi_utils.h"
#include "py_export_capi_inference.h"

std::vector<at::Tensor> mha_fwd_inference_test(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                               c10::optional<at::Tensor> &out_,
                                               c10::optional<at::Tensor> &alibi_slopes_,
                                               c10::optional<at::Tensor> &attn_mask_, const float softmax_scale,
                                               bool is_causal, int window_size_left, int window_size_right) {
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

    Tensor_t q_mcfa = convert_mcfa_tensor(q_padded);
    Tensor_t k_mcfa = convert_mcfa_tensor(k_padded);
    Tensor_t v_mcfa = convert_mcfa_tensor(v_padded);
    Tensor_t out_mcfa = convert_mcfa_tensor(out);
    Tensor_t alibi_slopes_mcfa = convert_mcfa_tensor(alibi_slopes_);
    Tensor_t attn_mask_mcfa = convert_mcfa_tensor(attn_mask_);

    mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = mha_fwd_inference(batch_size, seqlen_q, num_heads, seqlen_k, num_heads_k, head_size, q_mcfa, k_mcfa,
                                 v_mcfa, out_mcfa, alibi_slopes_mcfa, attn_mask_mcfa, softmax_scale, is_causal,
                                 window_size_left, window_size_right, stream);
    if (ret != MCFLASHATTN_STATUS_SUCCESS) {
        std::cerr << "Call mha_fwd_inference failed,ret = " << int(ret) << std::endl;
    }

    release_mcfa_tensor({q_mcfa, k_mcfa, v_mcfa, out_mcfa, alibi_slopes_mcfa, attn_mask_mcfa});

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) {
            out_.value().copy_(out);
        }
    }

    return {out, q_padded, k_padded, v_padded, out_padded};
}

std::vector<at::Tensor> mha_varlen_fwd_inference_test(
    at::Tensor &q, const at::Tensor &k, const at::Tensor &v, c10::optional<at::Tensor> &out_,
    const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_k, c10::optional<at::Tensor> &seqused_k,
    c10::optional<at::Tensor> &alibi_slopes_, int max_seqlen_q, const int max_seqlen_k, const float softmax_scale,
    const bool zero_tensors, bool is_causal, int window_size_left, int window_size_right) {
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

    Tensor_t q_mcfa = convert_mcfa_tensor(q_padded);
    Tensor_t k_mcfa = convert_mcfa_tensor(k_padded);
    Tensor_t v_mcfa = convert_mcfa_tensor(v_padded);
    Tensor_t out_mcfa = convert_mcfa_tensor(out);
    Tensor_t seqlens_q_mcfa = convert_mcfa_tensor(cu_seqlens_q);
    Tensor_t seqlens_k_mcfa = convert_mcfa_tensor(cu_seqlens_k);
    Tensor_t seqused_k_mcfa = convert_mcfa_tensor(seqused_k);
    Tensor_t alibi_slopes_mcfa = convert_mcfa_tensor(alibi_slopes_);

    mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = mha_varlen_fwd_inference(batch_size, total_q, num_heads, total_k, num_heads_k, head_size_og, q_mcfa,
                                        k_mcfa, v_mcfa, out_mcfa, seqlens_q_mcfa, seqlens_k_mcfa, seqused_k_mcfa,
                                        alibi_slopes_mcfa, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal,
                                        window_size_left, window_size_right, stream);
    if (ret != MCFLASHATTN_STATUS_SUCCESS) {
        std::cerr << "Call mha_varlen_fwd_inference failed,ret = " << int(ret) << std::endl;
    }

    release_mcfa_tensor(
        {q_mcfa, k_mcfa, v_mcfa, out_mcfa, seqlens_q_mcfa, seqlens_k_mcfa, seqused_k_mcfa, alibi_slopes_mcfa});

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) {
            out_.value().copy_(out);
        }
    }

    return {out, q_padded, k_padded, v_padded, out_padded};
}

std::vector<at::Tensor> mha_fwd_kvcache_test(
    at::Tensor &q, const at::Tensor &kcache, const at::Tensor &vcache, c10::optional<at::Tensor> &k_,
    c10::optional<at::Tensor> &v_, c10::optional<at::Tensor> &seqlens_k_, c10::optional<at::Tensor> &rotary_cos_,
    c10::optional<at::Tensor> &rotary_sin_, c10::optional<at::Tensor> &cache_batch_idx_,
    c10::optional<at::Tensor> &block_table_, c10::optional<at::Tensor> &alibi_slopes_, c10::optional<at::Tensor> &out_,
    const float softmax_scale, bool is_causal, int window_size_left, int window_size_right, bool is_rotary_interleaved,
    int num_splits) {
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
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(kcache);
    CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : kcache.size(0);
    const int page_block_size = !paged_KV ? 1 : kcache.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);
    const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q == 1 && !alibi_slopes_.has_value()) {
        is_causal = false;
    }
    if (is_causal) {
        window_size_right = 0;
    }

    if (window_size_left >= seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k) {
        window_size_right = -1;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    if (!paged_KV) {
        CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    at::Tensor q_padded, kcache_padded, vcache_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        kcache_padded =
            torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        vcache_padded =
            torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        kcache_padded = kcache;
        vcache_padded = vcache;
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

    if (num_splits < 1) {
        num_splits = compute_num_splits(batch_size, num_heads, head_size, seqlen_k, seqlen_q);
    }

    at::Tensor k, v, k_padded, v_padded;
    if (k_.has_value()) {
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k = k_.value();
        v = v_.value();
        TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
        TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        CHECK_DEVICE(k);
        CHECK_DEVICE(v);
        TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        int seqlen_knew = k.size(1);
        CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        if (head_size_og % 8 != 0) {
            k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            k_padded = k;
            v_padded = v;
        }
    }

    Tensor_t q_mcfa = convert_mcfa_tensor(q_padded);
    Tensor_t out_mcfa = convert_mcfa_tensor(out);

    Tensor_t kcache_mcfa = convert_mcfa_tensor(kcache_padded);
    Tensor_t vcache_mcfa = convert_mcfa_tensor(vcache_padded);

    Tensor_t k_mcfa = nullptr;
    Tensor_t v_mcfa = nullptr;
    if (k_.has_value()) {
        k_mcfa = convert_mcfa_tensor(k_padded);
        v_mcfa = convert_mcfa_tensor(v_padded);
    }

    Tensor_t seqlens_k_mcfa = convert_mcfa_tensor(seqlens_k_);
    Tensor_t rotary_cos_mcfa = convert_mcfa_tensor(rotary_cos_);
    Tensor_t rotary_sin_mcfa = convert_mcfa_tensor(rotary_sin_);
    Tensor_t cache_batch_idx_mcfa = convert_mcfa_tensor(cache_batch_idx_);
    Tensor_t block_table_mcfa = convert_mcfa_tensor(block_table_);
    Tensor_t alibi_slopes_mcfa = convert_mcfa_tensor(alibi_slopes_);
    Tensor_t softmax_lse_mcfa = convert_mcfa_tensor(softmax_lse);

    Tensor_t softmax_lse_accum_mcfa = nullptr;
    Tensor_t out_accum_mcfa = nullptr;

    if (num_splits > 1) {
        auto softmax_lse_accum = torch::empty({num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
        auto out_accum =
            torch::empty({num_splits, batch_size, num_heads, seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));

        softmax_lse_accum_mcfa = convert_mcfa_tensor(softmax_lse_accum);
        out_accum_mcfa = convert_mcfa_tensor(out_accum);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret =
        mha_fwd_kvcache(q_mcfa, kcache_mcfa, vcache_mcfa, k_mcfa, v_mcfa, seqlens_k_mcfa, rotary_cos_mcfa,
                        rotary_sin_mcfa, cache_batch_idx_mcfa, block_table_mcfa, alibi_slopes_mcfa, softmax_lse_mcfa,
                        out_mcfa, softmax_scale, is_causal, window_size_left, window_size_right, is_rotary_interleaved,
                        stream, num_splits, softmax_lse_accum_mcfa, out_accum_mcfa);

    release_mcfa_tensor({q_mcfa, out_mcfa, kcache_mcfa, vcache_mcfa, k_mcfa, v_mcfa, seqlens_k_mcfa, rotary_cos_mcfa,
                         rotary_sin_mcfa, cache_batch_idx_mcfa, block_table_mcfa, alibi_slopes_mcfa, softmax_lse_mcfa,
                         softmax_lse_accum_mcfa, out_accum_mcfa});

    if (ret != MCFLASHATTN_STATUS_SUCCESS) {
        std::cerr << "Call mha_fwd_kvcache failed,ret = " << int(ret) << std::endl;
    }

    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) {
            out_.value().copy_(out);
        }
        if (k_.has_value()) {
            kcache.copy_(kcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
            vcache.copy_(vcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
        }
    }

    return {out, softmax_lse};
}
