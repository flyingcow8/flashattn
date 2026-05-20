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

/*-----------------------training api--------------------------------*/
/*
// score ->[bs,      head_num,      q,      k  ]
// mask -> [bs_mask, head_num_mask, q_mask, k_mask]
// Mask shape should satisfy these rules
// 1. bs % bs_mask == 0
// 2. head_num % head_num_mask == 0
// 3. q_mask == 1 or q_mask == q
// 4. k_mask == 1 or k_mask == k or k_mask == (k + 3) / 4 * 4 (align k to multiples of 4)
*/
mcflashattnStatus_t
mha_fwd(int64_t batch_size,
        int64_t seqlen_q,
        int64_t num_heads_q,
        int64_t seqlen_k,
        int64_t num_heads_k,
        int64_t head_size_og,
        const Tensor_t q,                 // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t k,         // batch_size x seqlen_k x num_heads_k x head_size
        const Tensor_t v,         // batch_size x seqlen_k x num_heads_k x head_size
        Tensor_t out,             // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t alibi_slopes, // num_heads or batch_size x num_heads
        const Tensor_t attn_mask,
        const Tensor_t softmax_lse,   // batch_size x num_heads x seqlen_q
        const Tensor_t p, // [optional return]softmax batch_size x num_heads x seqlen_q_rounded x seqlen_k_rounded
        const Tensor_t rng_state, // [optional input] 2 x Int64
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        mcStream_t stream,
        mcflashattnExtendParameter_t extend_parameter_// extend paramerter
        ) {

    auto dprops = flash::mcGetCurrentDeviceProperties();
    int arch = dprops.major * 100 + dprops.minor;
    // check data continues
    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(softmax_lse);

    // check data type
    auto q_dtype = ((InternalTensor*)q->data)->dtype;
    TENSOR_CHECK(q_dtype == InternalTensor::DataType::FP16 || q_dtype == InternalTensor::DataType::BF16,
            "FlashAttention only supports fp16 and bf16 data type");
    CHECK_DTYPE(k, q_dtype);
    CHECK_DTYPE(v, q_dtype);

    // check others
    int64_t head_size_og_v = ((InternalTensor*)v->data)->sizes[3];

    TENSOR_CHECK(batch_size > 0, "batch size must be positive");
    TENSOR_CHECK(head_size_og <= 512, "FlashAttention forward only supports head dimension <= 512");
    TENSOR_CHECK(head_size_og_v <= 512, "FlashAttention forward only supports head dimension <= 512");
    TENSOR_CHECK(num_heads_q % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TENSOR_CHECK(head_size_og == head_size_og_v, "capi only supports head_size_og == head_size_og_v");

    CHECK_SHAPE(q, {batch_size, seqlen_q, num_heads_q, head_size_og});
    CHECK_SHAPE(k, {batch_size, seqlen_k, num_heads_k, head_size_og});
    CHECK_SHAPE(v, {batch_size, seqlen_k, num_heads_k, head_size_og_v});


    mcFlashAttn::Flash_fwd_params params;
    auto ret = set_fwd_parameters(params, q,k,v,out,alibi_slopes,attn_mask,softmax_lse,p,rng_state,p_dropout,softmax_scale,
        is_causal,window_size_left,window_size_right,extend_parameter_);
    params.arch = arch;

    if(ret != MCFLASHATTN_STATUS_SUCCESS) return ret;

    bool force_split_kernel = false;
    if (params.d == 80 && params.seqlen_q == 1024 && params.seqlen_k == 77) {
        force_split_kernel = true;
    }

    if (params.seqlen_k > 0) {
        params.num_splits = 1;
        run_mha_fwd(params, stream, force_split_kernel);
    }

    return MCFLASHATTN_STATUS_SUCCESS;
}

mcflashattnStatus_t
mha_varlen_fwd(
        int64_t batch_size,
        int64_t total_q,
        int64_t num_heads_q,
        int64_t total_k,
        int64_t num_heads_k,
        int64_t head_size_og,
        const Tensor_t q,                // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const Tensor_t k,         // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
        const Tensor_t v,         // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
        Tensor_t out,             // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const Tensor_t cu_seqlens_q,  // b+1
        const Tensor_t cu_seqlens_k,  // b+1
        const Tensor_t seqused_k,      // b. If given, only this many elements of each batch element's keys are used.
        const Tensor_t alibi_slopes, // num_heads or batch_size x num_heads
        const Tensor_t softmax_lse,   // batch_size x num_heads x seqlen_q
        const Tensor_t p, // [optional return]softmax batch_size x num_heads x seqlen_q_rounded x seqlen_k_rounded
        const Tensor_t rng_state, // [optional input] 2 x Int64
        int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        // const bool zero_tensors, python set to false
        bool is_causal,
        int window_size_left,
        int window_size_right,
        mcStream_t stream,
        mcflashattnExtendParameter_t extend_parameter_// extend paramerter
        ) {

    auto dprops = flash::mcGetCurrentDeviceProperties();
    int arch = dprops.major * 100 + dprops.minor;

    // check data continues
    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    CHECK_CONTIGUOUS(softmax_lse);

    // check data type
    auto q_dtype = ((InternalTensor*)q->data)->dtype;
    TENSOR_CHECK(q_dtype == InternalTensor::DataType::FP16 || q_dtype == InternalTensor::DataType::BF16,
            "FlashAttention only supports fp16 and bf16 data type");
    CHECK_DTYPE(k, q_dtype);
    CHECK_DTYPE(v, q_dtype);

    CHECK_DTYPE(cu_seqlens_q, InternalTensor::DataType::INT32);
    CHECK_DTYPE(cu_seqlens_k, InternalTensor::DataType::INT32);

    // check others
    int64_t head_size_og_v = ((InternalTensor*)v->data)->sizes[2];

    TENSOR_CHECK(batch_size > 0, "batch size must be positive");
    TENSOR_CHECK(head_size_og <= 512, "FlashAttention forward only supports head dimension at most 512");
    TENSOR_CHECK(num_heads_q % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TENSOR_CHECK(head_size_og == head_size_og_v, "capi only supports head_size_og == head_size_og_v");

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);


    mcFlashAttn::Flash_fwd_params params;
    auto ret = set_varlen_fwd_parameters(params,batch_size,q,k,v,out,cu_seqlens_q,cu_seqlens_k,seqused_k,
        alibi_slopes,softmax_lse, p, rng_state,max_seqlen_q,max_seqlen_k,p_dropout,softmax_scale,
        is_causal,window_size_left,window_size_right,extend_parameter_);
    params.arch = arch;

    if (max_seqlen_k > 0) {
        params.num_splits = 1;
        run_mha_fwd(params, stream);
    }

    return MCFLASHATTN_STATUS_SUCCESS;
}



#ifdef __cplusplus
}
#endif /* __cplusplus */
