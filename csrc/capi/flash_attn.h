#pragma once

#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>

#include <mcr/mc_runtime.h>

#define MCFLASH_MAJOR_VERSION 2
#define MCFLASH_MINOR_VERSION 0
#define MCFLASH_PATCH_VERSION 0
#define MCFLASHATTN_VERSION MCFLASH_MAJOR_VERSION * 10000 + MCFLASH_MINOR_VERSION * 100 + MCFLASH_PATCH_VERSION

typedef enum {
    MCFLASHATTN_STATUS_SUCCESS = 0,
    MCFLASHATTN_STATUS_FAILED,
    MCFLASHATTN_STATUS_ILLEGAL_PARAMETER,
    MCFLASHATTN_STATUS_ILLEGAL_TENSOR,
    MCFLASHATTN_STATUS_ILLEGAL_NUM_SPLITS,
    MCFLASHATTN_STATUS_ILLEGAL_SPLIT_PARA
}mcflashattnStatus_t;

typedef enum {
    MCFLASHATTN_DATATYPE_FP16 = 0,
    MCFLASHATTN_DATATYPE_BF16,
    MCFLASHATTN_DATATYPE_INT8,
    MCFLASHATTN_DATATYPE_INT32,
    MCFLASHATTN_DATATYPE_INT64,
    MCFLASHATTN_DATATYPE_FP32,
    MCFLASHATTN_DATATYPE_FP64,
    MCFLASHATTN_DATATYPE_NONE,
}mcflashattnDataType_t;

struct Tensor{
    void *data;
    int version;
};
typedef struct Tensor *Tensor_t;

inline struct Tensor* make_tensor(){
    struct Tensor* tensor = (struct Tensor*)malloc(sizeof(struct Tensor));
    tensor->data = NULL;
    tensor->version = MCFLASHATTN_VERSION;

    return tensor;
}


struct McFlashExtendParameter {
    void * data;
    int version;
};
typedef struct McFlashExtendParameter *mcflashattnExtendParameter_t;

inline mcflashattnExtendParameter_t make_extend_param(){
    mcflashattnExtendParameter_t extend_param = (struct McFlashExtendParameter*)malloc(sizeof(struct McFlashExtendParameter));
    extend_param->data = NULL;
    extend_param->version = MCFLASHATTN_VERSION;

    return extend_param;
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

//////////////////////////////////////////////////////////////////////////////////////
//
// Tensor
//
/////////////////////////////////////////////////////////////////////////////////////

/*
    batch,    dim:0 get_stride(0) get_shape(0)  batch_stride
    seqlen,   dim:1 get_stride(1) get_shape(1)  row_stride
    num_heads,dim:2 get_stride(2) get_shape(2)  head_stride
    head_size,dim:3 get_stride(3) get_shape(3)  stride = 1
*/
Tensor_t make_contiguous_tensor1d(
    void *data,
    mcflashattnDataType_t dtype,
    int head_size
);

Tensor_t make_contiguous_tensor2d(
    void *data,
    mcflashattnDataType_t dtype,
    int head_num,
    int head_size
);

Tensor_t make_contiguous_tensor3d(
    void *data,
    mcflashattnDataType_t dtype,
    int seqlen,
    int head_num,
    int head_size
);

Tensor_t make_contiguous_tensor4d(
    void *data,
    mcflashattnDataType_t dtype,
    int batch,
    int seqlen,
    int head_num,
    int head_size
);


Tensor_t make_tensor1d(
    void *data,
    mcflashattnDataType_t dtype,
    int size0,
    int stride0
);

Tensor_t make_tensor2d(
    void *data,
    mcflashattnDataType_t dtype,
    int size0,
    int size1,
    int stride0,
    int stride1
);

Tensor_t make_tensor3d(
    void *data,
    mcflashattnDataType_t dtype,
    int size0,
    int size1,
    int size2,
    int stride0,
    int stride1,
    int stride2
);

Tensor_t make_tensor4d(
    void *data,
    mcflashattnDataType_t dtype,
    int size0,
    int size1,
    int size2,
    int size3,
    int stride0,
    int stride1,
    int stride2,
    int stride3
);

Tensor_t make_tensor5d(
    void *data,
    mcflashattnDataType_t dtype,
    int size0,
    int size1,
    int size2,
    int size3,
    int size4,
    int stride0,
    int stride1,
    int stride2,
    int stride3,
    int stride4
);



void* get_tensor_data(Tensor_t tensor);
int get_tensor_size(Tensor_t tensor,int index);
int get_tensor_stride(Tensor_t tensor,int index);
int get_tensor_dims(Tensor_t tensor);
mcflashattnDataType_t get_tensor_dtype(Tensor_t tensor);

void print_tensor_info(Tensor_t tensor);

void release_tensor(Tensor_t tensor);
void release_extend_param(mcflashattnExtendParameter_t extend_param);

int compute_num_splits(int batch_size,int num_heads,int head_size,int seqlen_k,int seqlen_q);
int head_size_pad(int head_size_og);


////////////////////////////////////////////////////////////////////////////////////
//
// Flash Attention2 API
//
///////////////////////////////////////////////////////////////////////////////////

mcflashattnStatus_t
mha_fwd_inference(
    int64_t batch_size,
    int64_t seqlen_q,
    int64_t num_heads_q,
    int64_t seqlen_k,
    int64_t num_heads_k,
    int64_t head_size_og,
    const Tensor_t q,                 // batch_size x seqlen_q x num_heads_q x head_size
    const Tensor_t k,         // batch_size x seqlen_k x num_heads_k x head_size
    const Tensor_t v,         // batch_size x seqlen_k x num_heads_k x head_size
    Tensor_t out,            // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t alibi_slopes, // num_heads or batch_size x num_heads
    const Tensor_t attn_mask,    // batch_size x seqlen_q
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    mcStream_t stream,
    mcflashattnExtendParameter_t extend_parameter_ = NULL// extend paramerter
);

mcflashattnStatus_t
mha_varlen_fwd_inference(
    int64_t batch_size,
    int64_t total_q,
    int64_t num_heads_q,
    int64_t total_k,
    int64_t num_heads_k,
    int64_t head_size_og,
    const Tensor_t q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const Tensor_t k,         // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const Tensor_t v,         // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    Tensor_t out,             // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const Tensor_t cu_seqlens_q,  // b+1
    const Tensor_t cu_seqlens_k,  // b+1
    const Tensor_t seqused_k,      // b. If given, only this many elements of each batch element's keys are used.
    const Tensor_t alibi_slopes, // num_heads or batch_size x num_heads
    int max_seqlen_q,
    const int max_seqlen_k,
    const float softmax_scale,
    // const bool zero_tensors, python set to false
    bool is_causal,
    int window_size_left,
    int window_size_right,
    mcStream_t stream,
    mcflashattnExtendParameter_t extend_parameter_ = NULL// extend paramerter
);

mcflashattnStatus_t
mha_fwd_kvcache(
    const Tensor_t q,           // batch_size x seqlen_q x num_heads x head_size
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
    Tensor_t out,             // batch_size x seqlen_q x num_heads x head_size
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    mcStream_t stream,
    int num_splits = 1,
    const Tensor_t softmax_lse_accum = NULL, // num_splits x batch_size x num_heads x seqlen_q
    const Tensor_t out_accum = NULL, // num_splits x batch_size x num_heads x max_seqlen_q x head_size_rounded (32)
    mcflashattnExtendParameter_t extend_parameter_ = NULL// extend paramerter
);


// training
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
        mcflashattnExtendParameter_t extend_parameter_ = NULL// extend paramerter
        );

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
        mcflashattnExtendParameter_t extend_parameter_ = NULL// extend paramerter
        );

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
        const Tensor_t dk,   // batch_size x seqlen_k x num_heads_k x head_size
        const Tensor_t dv,   // batch_size x seqlen_k x num_heads_k x head_size
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
        mcflashattnExtendParameter_t extend_parameter_ = NULL// extend paramerter
        );

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
               const Tensor_t dk,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const Tensor_t dv,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
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
               mcflashattnExtendParameter_t extend_parameter_ = NULL// extend paramerter
               );

#ifdef __cplusplus
}
#endif /* __cplusplus */


