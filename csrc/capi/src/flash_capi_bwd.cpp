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
mha_bwd(int64_t batch_size,
        int64_t seqlen_q,
        int64_t num_heads_q,
        int64_t seqlen_k,
        int64_t num_heads_k,
        int64_t head_size_og,
        const Tensor_t dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const Tensor_t q,   // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t k,   // batch_size x seqlen_k x num_heads_k x head_size
        const Tensor_t v,   // batch_size x seqlen_k x num_heads_k x head_size
        Tensor_t out,   // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t softmax_d, // batch_size x num_heads x seqlen_q_rounded
        const Tensor_t softmax_lse,     // b x h x seqlen_q
        const Tensor_t dq,   // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t dk,   // batch_size x seqlen_k x num_heads x head_size
        const Tensor_t dv,   // batch_size x seqlen_k x num_heads x head_size
        const Tensor_t dq_accum,   // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t alibi_slopes, // num_heads or batch_size x num_heads
        const Tensor_t attn_mask,
        const Tensor_t rng_state, // [optional input] 2 x Int64
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const bool deterministic,
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
    CHECK_CONTIGUOUS(dout);
    CHECK_CONTIGUOUS(softmax_lse);
    CHECK_CONTIGUOUS(softmax_d);
    CHECK_CONTIGUOUS(dq_accum);

    // check data type
    auto q_dtype = ((InternalTensor*)q->data)->dtype;
    TENSOR_CHECK(q_dtype == InternalTensor::DataType::FP16 || q_dtype == InternalTensor::DataType::BF16,
            "FlashAttention only supports fp16 and bf16 data type");
    CHECK_DTYPE(k, q_dtype);
    CHECK_DTYPE(v, q_dtype);
    CHECK_DTYPE(out, q_dtype);
    CHECK_DTYPE(dout, q_dtype);

    // check others
    int64_t head_size = ((InternalTensor*)q->data)->sizes[3];
    int64_t head_size_og_v = ((InternalTensor*)v->data)->sizes[3];

    TENSOR_CHECK(batch_size > 0, "batch size must be positive");
    TENSOR_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TENSOR_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension <= 256");
    TENSOR_CHECK(head_size_og_v <= 256, "FlashAttention backward only supports head dimension <= 256");
    TENSOR_CHECK(num_heads_q % num_heads_k == 0, "num_heads of key/value must divide num_heads of query");
    TENSOR_CHECK(head_size == head_size_og_v, "capi only supports head_size == head_size_og_v");

    mcFlashAttn::Flash_bwd_params params;
    auto ret = set_bwd_parameters(params,dout,q,k,v,out,softmax_d,softmax_lse,dq,dk,dv,dq_accum,
        alibi_slopes,attn_mask,rng_state,p_dropout,softmax_scale,is_causal,window_size_left,
        window_size_right,deterministic,extend_parameter_);
    params.arch = arch;
    if(ret != MCFLASHATTN_STATUS_SUCCESS) return ret;

    if (params.seqlen_q > 0) {
        run_mha_bwd(params, stream);
    }

    return MCFLASHATTN_STATUS_SUCCESS;
}

mcflashattnStatus_t
mha_varlen_bwd(int64_t batch_size,
               int64_t total_q,
               int64_t num_heads_q,
               int64_t total_k,
               int64_t num_heads_k,
               int64_t head_size_og,
               const Tensor_t dout,  // total_q x num_heads, x head_size
               const Tensor_t q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const Tensor_t k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const Tensor_t v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               Tensor_t out,   // total_q x num_heads x head_size
               const Tensor_t softmax_d, // batch_size x num_heads x seqlen_q_rounded
               const Tensor_t softmax_lse,     // b x h x s   softmax logsumexp
               const Tensor_t dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const Tensor_t dk,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const Tensor_t dv,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const Tensor_t dq_accum,   // batch_size x seqlen_q x num_heads x head_size
               const Tensor_t cu_seqlens_q,  // b+1
               const Tensor_t cu_seqlens_k,  // b+1
               const Tensor_t alibi_slopes, // num_heads or b x num_heads
               const Tensor_t rng_state, // [optional input] 2 x Int64
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               // const bool zero_tensors, python set to false
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const bool deterministic,
               mcStream_t stream,
               mcflashattnExtendParameter_t extend_parameter_ // extend paramerter
               ) {
    auto dprops = flash::mcGetCurrentDeviceProperties();
    int arch = dprops.major * 100 + dprops.minor;
    // check data continues
    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(dout);
    CHECK_CONTIGUOUS(softmax_lse);
    CHECK_CONTIGUOUS(softmax_d);
    CHECK_CONTIGUOUS(dq_accum);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    if (is_causal) { window_size_right = 0; }

    // check data type
    auto q_dtype = ((InternalTensor*)q->data)->dtype;
    TENSOR_CHECK(q_dtype == InternalTensor::DataType::FP16 || q_dtype == InternalTensor::DataType::BF16,
            "FlashAttention only supports fp16 and bf16 data type");
    CHECK_DTYPE(k, q_dtype);
    CHECK_DTYPE(v, q_dtype);
    CHECK_DTYPE(out, q_dtype);
    CHECK_DTYPE(dout, q_dtype);

    CHECK_DTYPE(cu_seqlens_q, InternalTensor::DataType::INT32);
    CHECK_DTYPE(cu_seqlens_k, InternalTensor::DataType::INT32);

    // check others
    int64_t head_size = ((InternalTensor*)q->data)->sizes[2];
    int64_t head_size_og_v = ((InternalTensor*)v->data)->sizes[2];

    TENSOR_CHECK(batch_size > 0, "batch size must be positive");
    TENSOR_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TENSOR_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension <= 256");
    TENSOR_CHECK(head_size_og_v % 8 == 0, "head_size_og_v should be a multiple of 8");
    TENSOR_CHECK(head_size_og_v <= 256, "FlashAttention backward only supports head dimension <= 256");
    TENSOR_CHECK(num_heads_q % num_heads_k == 0, "num_heads of key/value must divide num_heads of query");
    TENSOR_CHECK(head_size == head_size_og_v, "capi only supports head_size == head_size_og_v");

    mcFlashAttn::Flash_bwd_params params;
    auto ret = set_varlen_bwd_parameters(params,dout,q,k,v,out,softmax_d,softmax_lse,dq,dk,dv,dq_accum,
        cu_seqlens_q,cu_seqlens_k,alibi_slopes,rng_state,max_seqlen_q,max_seqlen_k,p_dropout,softmax_scale,
        is_causal,window_size_left,window_size_right,deterministic,extend_parameter_);
    params.arch = arch;
    if(ret != MCFLASHATTN_STATUS_SUCCESS) return ret;

    if (max_seqlen_q > 0) {
        run_mha_bwd(params, stream);
    }
    return MCFLASHATTN_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
