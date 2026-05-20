#include "flash_parameter_utils.h"
#include "utils.h"

mcflashattnStatus_t
set_fwd_kvcache_parameters(
    mcFlashAttn::Flash_fwd_params &params,
    const Tensor_t q,                 // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t kcache,      // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const Tensor_t vcache,      // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const Tensor_t k,                 // batch_size x seqlen_knew x num_heads_k x head_size
    const Tensor_t v,                // batch_size x seqlen_knew x num_heads_k x head_size
    const Tensor_t seqlens_k,        // batch_size
    const Tensor_t rotary_cos,       // seqlen_ro x (rotary_dim / 2)
    const Tensor_t rotary_sin,       // seqlen_ro x (rotary_dim / 2)
    const Tensor_t cache_batch_idx,  // indices to index into the KV cache
    const Tensor_t block_table,      // batch_size x max_num_blocks_per_seq
    const Tensor_t alibi_slopes,     // num_heads or batch_size x num_heads
    const Tensor_t softmax_lse,     // batch_size x num_heads x seqlen_q
    Tensor_t out,                  // batch_size x seqlen_q x num_heads x head_size
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    int num_splits,
    const Tensor_t softmax_lse_accum, // num_splits x batch_size x num_heads x seqlen_q
    const Tensor_t out_accum,  // num_splits x batch_size x num_heads x max_seqlen_q x head_size_rounded (32)
    mcflashattnExtendParameter_t extend_parameter_, // extend paramerter
    bool for_get_num_splits // if true, don't check out_accum and softmax_lse_accum
){

    auto dtype = ((InternalTensor*)q->data)->dtype;
    const bool paged_KV = (block_table != nullptr);
    float softcap = 0.0f;
    if (extend_parameter_ != nullptr){
        softcap = get_extend_parameter_softcap(extend_parameter_);
    }

    Tensor_t block_table_ = nullptr;
    if(paged_KV){
        if(cache_batch_idx != nullptr) {
            std::cerr << "Paged KVcache does not support cache_batch_idx" << std::endl;
            return MCFLASHATTN_STATUS_FAILED; // Paged KVcache does not support cache_batch_idx
        }

        if(get_tensor_dtype(block_table) != MCFLASHATTN_DATATYPE_INT32) {
            std::cerr << "block_table must have dtype torch.int32" <<std::endl;
            return MCFLASHATTN_STATUS_FAILED;
        }
        if(get_tensor_stride(block_table,1) != 1) {
            std::cerr << "block_table must have contiguous last dimension" << std::endl;
            return MCFLASHATTN_STATUS_FAILED;
        }

        block_table_ = block_table;
    }

    const int batch_size_ = get_tensor_size(q,0);
    int seqlen_q_ = get_tensor_size(q,1);
    int num_heads_ = get_tensor_size(q,2);
    const int head_size_og_ = get_tensor_size(q,3);
    const int head_size_v_og_ = get_tensor_size(v,3);

    const int max_num_blocks_per_seq = !paged_KV ? 0 : get_tensor_size(block_table_,1);
    const int num_blocks = !paged_KV ? 0 : get_tensor_size(kcache,0);
    const int page_block_size = !paged_KV ? 0 :get_tensor_size(kcache,1);

    // TORCH_CHECK(!paged_KV || page_block_size % 8 == 0 || page_block_size == 1, "Paged KV cache block size must be divisible by 8 or equal to 1");
    const int seqlen_k_ = !paged_KV ? get_tensor_size(kcache,1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k_ = get_tensor_size(kcache,2);
    const int batch_size_c = !paged_KV ? get_tensor_size(kcache,0) : batch_size_;
    // TORCH_CHECK(batch_size > 0, "batch size must be positive");
    // TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension <= 256");
    // TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q_ == 1 && alibi_slopes == nullptr) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    // const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && alibi_slopes == nullptr;

    if (window_size_left >= seqlen_k_) { window_size_left = -1; }
    if (window_size_right >= seqlen_k_) { window_size_right = -1; }

    // CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    // if (!paged_KV) {
    //     CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
    //     CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
    // } else {
    //     CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_og);
    //     CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_og);
    //     CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    // }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_ = round_multiple(head_size_og_, 8);
    const int head_size_v = round_multiple(head_size_v_og_,8);
    // we do not have headdim=224 kernel, padding head_size_rounded to 256 to support theses headdim
    const int head_size_rounded = round_multiple(head_size_, 32) == 224 ? 256 : round_multiple(head_size_, 32);
    const int head_size_v_rounded = round_multiple(head_size_v,32) == 224 ? 256 : round_multiple(head_size_v,32);
    const int seqlen_q_rounded = round_multiple(seqlen_q_, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k_, 128);

    set_params_fprop(params,
                     batch_size_,
                     seqlen_q_, seqlen_k_,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads_, num_heads_k_,
                     head_size_, head_size_rounded,
                     head_size_v,head_size_v_rounded,
                     q, kcache, vcache, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                    //  /*softmax_lse*/ nullptr,
                    get_tensor_data(softmax_lse),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right);

    Tensor_t k_ = nullptr;
    Tensor_t v_ = nullptr;
    if(k != nullptr){
        // TORCH_CHECK(v.has_value(), "If key is supplied, value must also be passed in");
        // TORCH_CHECK(seqlens_k.has_value(), "If key is supplied, seqlens_k must also be passed in");
        // TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k_ = k;
        v_ = v;
        CHECK_CONTIGUOUS(k_);
        CHECK_CONTIGUOUS(v_);
        CHECK_DTYPE(k_, dtype);
        CHECK_DTYPE(v_, dtype);

        int seqlen_knew = get_tensor_size(k_,1);

        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = get_tensor_data(k_);
        params.vnew_ptr = get_tensor_data(v_);
        // All stride are in elements, not bytes.
        params.knew_batch_stride = get_tensor_stride(k_,0);
        params.vnew_batch_stride = get_tensor_stride(v_,0);
        params.knew_row_stride = get_tensor_stride(k_,1);
        params.vnew_row_stride = get_tensor_stride(k_,1);
        params.knew_head_stride = get_tensor_stride(k_,2);
        params.vnew_head_stride = get_tensor_stride(k_,2);
    }

    if(seqlens_k != nullptr){
        auto seqlens_k_ = seqlens_k;
        params.cu_seqlens_k = static_cast<int *>(get_tensor_data(seqlens_k_));
    }

    params.is_seqlens_k_cumulative = !(seqlens_k != nullptr);
    params.leftpad_k = nullptr;

    if(rotary_cos != nullptr){
        auto rotary_cos_ = rotary_cos;
        params.rotary_dim = get_tensor_size(rotary_cos_,1) * 2;
        const int seqlen_ro = get_tensor_size(rotary_cos_,0);

        auto rotary_sin_ = rotary_sin;
        params.rotary_cos_ptr = get_tensor_data(rotary_cos_);
        params.rotary_sin_ptr = get_tensor_data(rotary_sin_);
        params.is_rotary_interleaved = is_rotary_interleaved;
    }else {
        params.rotary_dim = 0;
    }

    if (cache_batch_idx != nullptr) {
        auto cache_batch_idx_ = cache_batch_idx;
        params.cache_batch_idx = reinterpret_cast<int *>(get_tensor_data(cache_batch_idx_));
    }

    ///////////////////////////////////////////////////////////////////////////////////
    //
    // process splitkv
    //  check valid , num_splits must in 1 ~ 128 , 2024/7/10
    //  if num_splits = 1 , softmax_lse_accum and out_accum can be null
    //  if num_splits > 1 , softmax_lse_aucc and out_accum must be not null
    //
    //////////////////////////////////////////////////////////////////////////////////

    if(num_splits < 1) return MCFLASHATTN_STATUS_ILLEGAL_NUM_SPLITS;
    if(num_splits > 128) return MCFLASHATTN_STATUS_ILLEGAL_NUM_SPLITS;
    // don't check softmax_lse_accum and out_accum when get num splits
    if(!for_get_num_splits && num_splits != 1){
        if(softmax_lse_accum == nullptr || get_tensor_data(softmax_lse_accum) == nullptr) return MCFLASHATTN_STATUS_ILLEGAL_SPLIT_PARA;
        if(out_accum == nullptr || get_tensor_data(out_accum) == nullptr) return MCFLASHATTN_STATUS_ILLEGAL_SPLIT_PARA;
    }

    params.num_splits = num_splits;

    // Not need set softmax_lse_accum and out_accum when num_splits = 1
    // don't set softmax_lse_accum and out_accum when get num splits
    if (!for_get_num_splits && params.num_splits > 1) {
        params.softmax_lseaccum_ptr = get_tensor_data(softmax_lse_accum);
        params.oaccum_ptr = get_tensor_data(out_accum);
    }

    if (paged_KV) {
        params.block_table = reinterpret_cast<int *>(get_tensor_data(block_table_));
        params.block_table_batch_stride = get_tensor_stride(block_table_,0);
    }
    params.page_block_size = page_block_size;

    set_params_alibi(params, alibi_slopes, batch_size_, num_heads_);

    return MCFLASHATTN_STATUS_SUCCESS;

}
