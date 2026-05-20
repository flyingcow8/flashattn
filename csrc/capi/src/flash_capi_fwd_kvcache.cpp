#include <iostream>
#include "tensor.h"
#include "utils.h"
#include "flash_attn.h"
#include "flash_parameter_utils.h"
#include "run_mha.h"
#include "host_utils.h"


using namespace mcFlashAttn;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

mcflashattnStatus_t
mha_fwd_kvcache(const Tensor_t q,                 // batch_size x seqlen_q x num_heads x head_size
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
                mcStream_t stream,
                int num_splits,
                const Tensor_t softmax_lse_accum, // num_splits x batch_size x num_heads x seqlen_q
                const Tensor_t out_accum,  // num_splits x batch_size x num_heads x max_seqlen_q x head_size_rounded (32)
                mcflashattnExtendParameter_t extend_parameter_ // extend paramerter
                ){

    auto dprops = flash::mcGetCurrentDeviceProperties();
    int arch = dprops.major * 100 + dprops.minor;
    // check continues
    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(kcache);
    CHECK_CONTIGUOUS(vcache);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(softmax_lse);

    // check data type
    auto q_dtype = ((InternalTensor*)q->data)->dtype;
    TENSOR_CHECK(q_dtype == InternalTensor::DataType::FP16 || q_dtype == InternalTensor::DataType::BF16,
            "FlashAttention only supports fp16 and bf16 data type");
    CHECK_DTYPE(kcache, q_dtype);
    CHECK_DTYPE(vcache, q_dtype);
    mcFlashAttn::Flash_fwd_params params;
    auto ret = set_fwd_kvcache_parameters(params,q,kcache,vcache,k,v,seqlens_k,rotary_cos,rotary_sin,
        cache_batch_idx,block_table,alibi_slopes,softmax_lse,out,softmax_scale,is_causal,window_size_left,
        window_size_right,is_rotary_interleaved,num_splits,softmax_lse_accum,out_accum,extend_parameter_);
    params.arch = arch;
    if(ret != MCFLASHATTN_STATUS_SUCCESS) return ret;

    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    bool paged_KV = (block_table != nullptr);
    params.num_splits = num_splits;
    run_mha_fwd(params, stream, /*force_split_kernel=*/k != nullptr || cache_batch_idx != nullptr || paged_KV);

    return MCFLASHATTN_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
