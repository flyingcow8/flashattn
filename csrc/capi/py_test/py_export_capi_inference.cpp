#include "flash_attn.h"
#include "py_export_capi_utils.h"
#include "py_export_capi_inference.h"

//------------------ infer --------------------

std::vector<at::Tensor>
mha_fwd_kvcache_test(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                c10::optional<at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<at::Tensor> &seqlens_k_, // batch_size
                c10::optional<at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                c10::optional<at::Tensor> &leftpad_k_, // batch_size
                c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits,
                c10::optional<at::Tensor> &s_aux_ // (n_heads)
                ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only supports fp16 and bf16 data type");
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

    TORCH_CHECK(!paged_KV || (page_block_size > 0 && (page_block_size & (page_block_size - 1)) == 0), "Paged KV cache block size must be a power of 2!");
    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);
    const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension <= 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    mcflashattnExtendParameter_t extend_parameter_ = make_extend_param();
    set_extend_parameter_softcap(extend_parameter_, softcap);

    if (leftpad_k_.has_value()) {
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
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
        kcache_padded = torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        vcache_padded = torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
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
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    // we do not have headdim=224 kernel, padding head_size_rounded to 256 to support theses headdim
    const int head_size_rounded = round_multiple(head_size, 32) == 224 ? 256 : round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    // compute num_splits
    if(num_splits < 1){
        num_splits = compute_num_splits(batch_size,num_heads,head_size,seqlen_k,seqlen_q);
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
        CHECK_DEVICE(k); CHECK_DEVICE(v);
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
    if(k_.has_value()){
        k_mcfa = convert_mcfa_tensor(k_padded);
        v_mcfa = convert_mcfa_tensor(v_padded);
    }

    Tensor_t seqlens_k_mcfa = convert_mcfa_tensor(seqlens_k_);
    Tensor_t rotary_cos_mcfa = convert_mcfa_tensor(rotary_cos_);
    Tensor_t rotary_sin_mcfa = convert_mcfa_tensor(rotary_sin_);
    Tensor_t cache_batch_idx_mcfa = convert_mcfa_tensor(cache_batch_idx_);
    Tensor_t block_table_mcfa = convert_mcfa_tensor(block_table_);
    Tensor_t leftpad_k_mcfa = convert_mcfa_tensor(leftpad_k_);
    Tensor_t alibi_slopes_mcfa = convert_mcfa_tensor(alibi_slopes_);
    Tensor_t softmax_lse_mcfa = convert_mcfa_tensor(softmax_lse);

    extend_parameter_->leftpad_k_ = leftpad_k_mcfa;

    // process splitkv
    Tensor_t softmax_lse_accum_mcfa = nullptr;
    Tensor_t out_accum_mcfa = nullptr;

    if(num_splits > 1){
        auto softmax_lse_accum = torch::empty({num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
        auto out_accum = torch::empty({num_splits, batch_size, num_heads, seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));

        softmax_lse_accum_mcfa = convert_mcfa_tensor(softmax_lse_accum);
        out_accum_mcfa = convert_mcfa_tensor(out_accum);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = mha_fwd_kvcache(
                q_mcfa,
                kcache_mcfa,
                vcache_mcfa,
                k_mcfa,
                v_mcfa,
                seqlens_k_mcfa,
                rotary_cos_mcfa,
                rotary_sin_mcfa,
                cache_batch_idx_mcfa,
                block_table_mcfa,
                alibi_slopes_mcfa,
                softmax_lse_mcfa,
                out_mcfa,
                softmax_scale,
                is_causal,
                window_size_left,
                window_size_right,
                is_rotary_interleaved,
                stream,
                num_splits,
                softmax_lse_accum_mcfa,
                out_accum_mcfa,
                extend_parameter_
            );

    release_mcfa_tensor({q_mcfa, out_mcfa, kcache_mcfa, vcache_mcfa, k_mcfa, v_mcfa, seqlens_k_mcfa, rotary_cos_mcfa,
                         rotary_sin_mcfa, cache_batch_idx_mcfa, block_table_mcfa, alibi_slopes_mcfa, softmax_lse_mcfa,
                         softmax_lse_accum_mcfa, out_accum_mcfa});
    release_extend_param(extend_parameter_);

    if(ret != MCFLASHATTN_STATUS_SUCCESS){
        std::cerr << "Call mha_fwd_kvcache failed, ret = " << int(ret) << std::endl;
    }

    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
        if (k_.has_value()) {
            // It's expensive to copy the KV cache here for the case where head size not divisible by 8,
            // but we don't expect to get this case in practice. This is just so that the code works for that case.
            kcache.copy_(kcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
            vcache.copy_(vcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
        }
    }

    return {out, softmax_lse};
}
//---------------------------------------------------------------------------------------------
