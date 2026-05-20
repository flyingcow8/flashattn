## Overriew

Flash attention C APIs提供了Flash attention2的C接口，它不依赖torch库，可用于推理和训练。
1. mha_fwd
2. mha_varlen_fwd
3. mha_bwd
4. mha_varlen_bwd
5. mha_fwd_kvcache

## 数据结构

### Version
```
#define MCFLASH_MAJOR_VERSION 2
#define MCFLASH_MINOR_VERSION 0
#define MCFLASH_PATCH_VERSION 0
#define MCFLASHATTN_VERSION MCFLASH_MAJOR_VERSION * 10000 + MCFLASH_MINOR_VERSION * 100 + MCFLASH_PATCH_VERSION
```

1. Major，主版本，该版本号增加时，表示进行重大的更新或重构，这些更新可能包含**不兼容的API变更**、功能的大幅增加或减少，或者是对软件架构的彻底重写，通常需要用户重新编译。
2. Minor，次版本，次版本号增加时，在保持**向后兼容性**的前提下，增加了新的功能。用户升级到新的次版本时，不需要担心与旧版本不兼容的问题。
3. Patch，修订号，用于标识对API的微小修改，主要是修复bug、改善性能或进行小的功能调整。用户升级到新的修订版本时，不需要担心功能上的变化和兼容性问题。

### mcflashattnStatus_t
C API执行后的返回结果，表示执行的状态。
```
typedef enum {
    MCFLASHATTN_STATUS_SUCCESS = 0,
    MCFLASHATTN_STATUS_FAILED,
    MCFLASHATTN_STATUS_ILLEGAL_PARAMETER,
    MCFLASHATTN_STATUS_ILLEGAL_TENSOR,
    MCFLASHATTN_STATUS_ILLEGAL_NUM_SPLITS,
    MCFLASHATTN_STATUS_ILLEGAL_SPLIT_PARA
}mcflashattnStatus_t;
```
- MCFLASHATTN_STATUS_SUCCESS， 成功
- MCFLASHATTN_STATUS_FAILED， 失败
- MCFLASHATTN_STATUS_ILLEGAL_PARAMETER， 传入的参数非法
- MCFLASHATTN_STATUS_ILLEGAL_TENSOR，传入的Tensor非法
- MCFLASHATTN_STATUS_ILLEGAL_NUM_SPLITS， 设置的num_splits <= 0
- MCFLASHATTN_STATUS_ILLEGAL_SPLIT_PARA， split相关的参数设置错误

### mcflashattnDataType_t
数据类型
```
typedef enum {
    MCFLASHATTN_DATATYPE_FP16 = 0,
    MCFLASHATTN_DATATYPE_BF16,
    MCFLASHATTN_DATATYPE_INT8,
    MCFLASHATTN_DATATYPE_INT32,
    MCFLASHATTN_DATATYPE_FP32,
    MCFLASHATTN_DATATYPE_FP64,
    MCFLASHATTN_DATATYPE_NONE,
}mcflashattnDataType_t;
```
### Tensor_t
```
struct Tensor{
    void *data;
    int version;
};
typedef struct Tensor *Tensor_t;
```
用来向C API传递Tensor数据的，仅仅用于传递数据信息，不会做任何修改内存的操作。
- data ，数据指针，需要是device端指针，由用户申请和释放。
- version， C API的version，用于version的校验。调用`make_tensor`时，会将头文件的version号传递到so内。

### mcflashattnExtendParameter_t
```
struct McFlashExtendParameter {
    float softcap;
    Tensor_t leftpad_k_;
    Tensor_t block_table_;
    int version;
};
typedef struct McFlashExtendParameter *mcflashattnExtendParameter_t;
```
扩展参数，用于参数的扩展，目前支持的扩展参数为softcap, leftpad_k_, block_table_

## Function

### Tensor创建函数

- make_tensor
```
inline struct Tensor* make_tensor(){
    struct Tensor* tensor = (struct Tensor*)malloc(sizeof(struct Tensor));
    tensor->data = NULL;
    tensor->version = MCFLASHATTN_VERSION;

    return tensor;
}
```
构建一个空的Tensor，是inline的，主要用于向so传递头文件的version，用户通常不需要主动调用该函数。

- make_contiguous_tensor*d
根据用户传入的指针以及相应的size参数，创建内存排布是连续的Tensor，只需要提供各个维度的size，各个维度的stride按照连续存储的方式计算得到。
```
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
```

- make_tensor*d

根据用户传入的指针以及相应的size和stride参数，创建Tensor。
```
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
```
注意，C API内部只能处理连续存储的Tensor，即如上每个API的最后一个stride值必须为1。

### 获取Tensor的函数
```
void* get_tensor_data(Tensor_t tensor);
int get_tensor_size(Tensor_t tensor,int index);
int get_tensor_stride(Tensor_t tensor,int index);
int get_tensor_dims(Tensor_t tensor);
mcflashattnDataType_t get_tensor_dtype(Tensor_t tensor);
```
用于获取Tensor的数据指针，各个维度信息以及数据类型。

### Tensor的释放
```
void release_tensor(Tensor_t tensor);
```
释放Tensor占用的资源。
注意，**该函数不会释放设置Tensor的数据指针（Tensor->data)指向的memory，这部分资源由用户管理。**

### ExtendParameter的创建和释放
```
inline mcflashattnExtendParameter_t make_extend_param(){
    mcflashattnExtendParameter_t extend_param = (struct McFlashExtendParameter*)malloc(sizeof(struct McFlashExtendParameter));
    extend_param->data = NULL;
    extend_param->version = MCFLASHATTN_VERSION;

    return extend_param;
}

void release_extend_param(mcflashattnExtendParameter_t extend_param);

```

### 设置ExterndParameter的值

```
void set_extend_parameter_softcap(mcflashattnExtendParameter_t extend_param, const float softcap);
void set_extend_parameter_leftpad(mcflashattnExtendParameter_t extend_param, void *data, int padding_size);
void set_extend_parameter_block_table(mcflashattnExtendParameter_t extend_param, void *data, int block_table_size_m,
                                     int block_table_size_n);
```

### 获取ExterndParameter的值

```
float get_extend_parameter_softcap(mcflashattnExtendParameter_t extend_param);
Tensor_t get_extend_parameter_leftpad(mcflashattnExtendParameter_t extend_param);
Tensor_t get_extend_parameter_block_table(mcflashattnExtendParameter_t extend_param);
```



### 辅助函数

```
int head_size_pad(int head_size_og);
int compute_num_splits(int batch_size,int num_heads,int head_size,int seqlen_k,int seqlen_q);
```

- head_size_pad, C API需要将head size pad到8的倍数，然后按照pad后的size申请显存。 例如head_size_og = 31，则返回32；head_size_og = 78，则返回80；head_size_og = 64，返回64。
- compute_num_splits, fwd_kvcache需要根据传入的参数信息，计算num_splits，用于根据该值分配相应的memory后传递给fwd_kvcache。如果用户不指定num_splits，程序中默认使用1.

### Usage of Split Kernel
To use the split kernel, the value of `num_splits` needs to be set in advance. There are two ways to set the value of `num_splits`:
1. Set directly by the user.
2. Call the corresponding `get_*_num_splits` method of the API to obtain the `num_splits` value recommended by the Flash Attn library.
The specific steps to use the split kernel are as follows:
1. First, the user calls the `get_fwd/var_fwd/fwd_kvcache_num_splits` interface or directly sets the value of `num_splits`.
2. The user allocates the space for `softmax_lse_accum` and `out_accum` according to the value of `num_splits`.
3. Call `fwd/var_fwd/fwd_kvcache` and pass `softmax_lse_accum` and `out_accum` as parameters.

After calling `get_fwd/var_fwd/fwd_kvcache_num_splits`, the returned value of `num_splits` should be checked.
 - If the return value is greater than 1, it is a valid `num_splits` value.
 - If the return value is 1, FlashAttn does not recommend calling the split kernel. However, if the user sets an arbitrary valid `num_splits` value, it may lead to performance not meeting expectations.
 - If the return value is less than 1, the parameter is illegal. Please check whether the parameter meets the requirements.

example codes
```cpp
    auto num_splits = get_fwd_kvcache_num_splits(
        tensor_q,tensor_kcache,tensor_vcache,tensor_k,tensor_v,tensor_seqlens_k,tensor_cos,tensor_sin,
        tensor_cache_batch_idx,tensor_block_table,nullptr,tensor_softmax_lse,tensor_out,softmax_scale,
        is_causal,window_size_left,window_size_right,false
    );

    // set num_splits to a special value
    // int num_splits = 4;
    std::cout << "get num_splits:" << num_splits << std::endl;
    if(num_splits < 1) {
        std::cerr << "get num splits error, num_splits must be greater than 1." << std::endl;
        std::cerr << "exit..." << std::endl;
        exit(1);
    }

    if(num_splits == 1) num_splits = 2; // set a speical value to use split kernel

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

    Tensor_t tensor_softmax_lse_accum = nullptr;
    Tensor_t tensor_out_accum = nullptr;
    void *softmax_lse_accum_ptr = nullptr;
    void *out_accum_ptr = nullptr;
    if(num_splits > 1){
        mcMalloc((void **)(&softmax_lse_accum_ptr), num_splits * batch_size * num_heads_q * seqlen_q * sizeof(float));
        tensor_softmax_lse_accum = make_contiguous_tensor4d(
            softmax_lse_accum_ptr,
            MCFLASHATTN_DATATYPE_FP32,
            num_splits,
            batch_size,
            num_heads_q,
            seqlen_q
        );
        const int head_size = round_multiple(head_size_og, 8);
        // we do not have headdim=224 kernel, padding head_size_rounded to 256 to support theses headdim
        const int head_size_rounded = round_multiple(head_size, 32) == 224 ? 256 : round_multiple(head_size, 32);
        mcMalloc((void **)(&out_accum_ptr), num_splits * batch_size * num_heads_q * seqlen_q * head_size_rounded * sizeof(float));

        tensor_out_accum = make_contiguous_tensor5d(
            out_accum_ptr,
            MCFLASHATTN_DATATYPE_FP32,
            num_splits,
            batch_size,
            num_heads_q,
            seqlen_q,
            head_size_rounded
        );

    }
```

## APIs

### mha_fwd
```
mcflashattnStatus_t
mha_fwd(int64_t batch_size_,
        int64_t seqlen_q_,
        int64_t num_heads_q_,
        int64_t seqlen_k_,
        int64_t num_heads_k_,
        int64_t head_size_og_,
        const Tensor_t q,                 // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t k,         // batch_size x seqlen_k x num_heads_k x head_size
        const Tensor_t v,         // batch_size x seqlen_k x num_heads_k x head_size
        Tensor_t out_,             // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t alibi_slopes_, // num_heads or batch_size x num_heads
        const Tensor_t attn_mask_,
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
```
Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
    1 1 1 1 0
    1 1 1 1 1
If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
    0 0
    0 0
    0 0
    1 0
    1 1
If the row of the mask is all zero, the output will be zero.

If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

Arguments:
    batch_size: int batch size
    seqlen_q: int seqlen of q
    num_heads_q: int number heads of q
    seqlen_k: int seqlen of k
    num_heads_k: int number heads of k
    head_size_og: int origin head size
    q: (batch_size, seqlen, nheads, headdim)
    k: (batch_size, seqlen, nheads_k, headdim)
    v: (batch_size, seqlen, nheads_k, headdim)
    out: (batch_size, seqlen, nheads, headdim).
    alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        is added to the attention score of query i and key j.
    attn_mask_: [optional] mask ->  [bs/1,  hdim/1,  q/1,   k/1]
                                            [hdim/1, q/1,   k/1]
                                                    [q/1,   k/1]
                                                           [k/1]
    softmax_lse: (batch_size, nheads, seqlen) , must input when use softmax
    p: (batch_size, nheads, seqlen, seqlen_k) , must input when use softmax
    rng_state: 2 x Int64 , stored random state {seed , offset}
    p_dropout: float. Dropout probability.
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    stream: stream
    extend_parameter_: extend parameter, support softcap and leftpad_k extend parameter
Return:
    mcflashattnStatus_t

### mha_varlen_fwd

```
mcflashattnStatus_t
mha_varlen_fwd(
        int64_t batch_size_,
        int64_t total_q_,
        int64_t num_heads_q_,
        int64_t total_k_,
        int64_t num_heads_k_,
        int64_t head_size_og_,
        const Tensor_t q,                // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const Tensor_t k,         // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
        const Tensor_t v,         // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
        Tensor_t out_,             // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const Tensor_t cu_seqlens_q,  // b+1
        const Tensor_t cu_seqlens_k,  // b+1
        const Tensor_t seqused_k,      // b. If given, only this many elements of each batch element's keys are used.
        const Tensor_t alibi_slopes_, // num_heads or batch_size x num_heads
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
```

Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
    1 1 1 1 0
    1 1 1 1 1
If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
    0 0
    0 0
    0 0
    1 0
    1 1
If the row of the mask is all zero, the output will be zero.

If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

Arguments:
    batch_size:   int64_t batch_size,
    total_q:      int64_t total_q,
    num_heads_q:  int64_t num_heads_q,
    total_k:      int64_t total_k,
    num_heads_k:  int64_t num_heads_k,
    head_size_og: int64_t head_size_og,
    q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
    k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
    v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
    out: (total, nheads, headdim).
    cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
        of the sequences in the batch, used to index into q.
    cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
        of the sequences in the batch, used to index into kv.
    seqused_k: b. If given, only this many elements of each batch element's keys are used.
    alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        is added to the attention score of query i and key j.
    softmax_lse: (batch_size, nheads, seqlen) , must input when use softmax
    p: (batch_size, nheads, seqlen, seqlen_k) , must input when use softmax
    rng_state: 2 x Int64 , stored random state {seed , offset}
    max_seqlen_q: int. Maximum query sequence length in the batch.
    max_seqlen_k: int. Maximum key sequence length in the batch.
    p_dropout: float. Dropout probability.
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    stream: stream
    extend_parameter_: extend parameter, support softcap extend parameter
Return:
    mcflashattnStatus_t

### mha_bwd

```
// NOTE: For MQA/GQA bwd, C API does not depend on Torch, so C API will not call torch reduce_sum op
//       and reduce_sum need to be called manually by the user.
//       reduce_sum for dk and dv from (b, sk, h_q, d) to (b, sk, h_k, d), sum along head_num dimension
mcflashattnStatus_t
mha_bwd(int64_t batch_size_,
        int64_t seqlen_q_,
        int64_t num_heads_q_,
        int64_t seqlen_k_,
        int64_t num_heads_k_,
        int64_t head_size_og_,
        const Tensor_t dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const Tensor_t q,   // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t k,   // batch_size x seqlen_k x num_heads_k x head_size
        const Tensor_t v,   // batch_size x seqlen_k x num_heads_k x head_size
        Tensor_t out,   // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t softmax_d, // batch_size x num_heads x seqlen_q_rounded
        const Tensor_t softmax_lse,     // batch_size x num_heads x seqlen_q
        const Tensor_t dq,   // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t dk,   // batch_size x seqlen_k x num_heads x head_size
        const Tensor_t dv,   // batch_size x seqlen_k x num_heads x head_size
        const Tensor_t dq_accum,   // batch_size x seqlen_q x num_heads x head_size
        const Tensor_t alibi_slopes_, // num_heads or batch_size x num_heads
        const Tensor_t attn_mask_,
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
```
Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

Arguments:
    batch_size: int batch size
    seqlen_q: int seqlen of q
    num_heads_q: int number heads of q
    seqlen_k: int seqlen of k
    num_heads_k: int number heads of k
    head_size_og: int origin head size
    dout: (batch_size, seqlen, nheads, headdim)
    q: (batch_size, seqlen, nheads, headdim)
    k: (batch_size, seqlen, nheads_k, headdim)
    v: (batch_size, seqlen, nheads_k, headdim)
    out: (batch_size, seqlen, nheads, headdim)
    softmax_d: (batch_size, nheads, seqlen) , must input when use softmax
    softmax_lse: (batch_size, nheads, seqlen) , must input when use softmax
    dq: (batch_size, seqlen, nheads, headdim)
    dk: (batch_size, seqlen, nheads, headdim)
    dv: (batch_size, seqlen, nheads, headdim)
    dq_accum: (batch_size, seqlen, nheads, headdim) , must input use to compute dq
    alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        is added to the attention score of query i and key j.
    attn_mask_: [optional] mask ->  [bs/1,  hdim/1,  q/1,   k/1]
                                            [hdim/1, q/1,   k/1]
                                                    [q/1,   k/1]
                                                            [k/1]
    rng_state: 2 x Int64 , stored random state {seed , offset}
    p_dropout: float. Dropout probability.
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    deterministic: bool. Whether to use the deterministic implementation of the backward pass,
        which is slightly slower and uses more memory. The forward pass is always deterministic.
    stream: stream
    extend_parameter_: extend parameter, support softcap extend parameter
Return:
    mcflashattnStatus_t

### mha_varlen_bwd

```
// NOTE: For MQA/GQA bwd, C API does not depend on Torch, so C API will not call torch reduce_sum op
//       and reduce_sum need to be called manually by the user.
//       reduce_sum for dk and dv from (total_k, h_q, d) to (total_k, h_k, d), sum along head_num dimension
mcflashattnStatus_t
mha_varlen_bwd(int64_t batch_size_,
               int64_t total_q_,
               int64_t num_heads_q_,
               int64_t total_k_,
               int64_t num_heads_k_,
               int64_t head_size_og_,
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
               mcflashattnExtendParameter_t extend_parameter_ = NULL// extend paramerter
               );
```
Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

Arguments:
    batch_size: int batch size
    seqlen_q: int seqlen of q
    num_heads_q: int number heads of q
    seqlen_k: int seqlen of k
    num_heads_k: int number heads of k
    head_size_og: int origin head size
    dout: (total, nheads, headdim)
    q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
    k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
    v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
    out: (total, nheads, headdim).
    softmax_d: (batch_size, nheads, seqlen) , must input when use softmax
    softmax_lse: (batch_size, nheads, seqlen) , must input when use softmax
    dq: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
    dk: (total_k, nheads, headdim), where total_k = total number of key tokens in the batch.
    dv: (total_k, nheads, headdim), where total_k = total number of key tokens in the batch.
    dq_accum: (batch_size, seqlen, nheads, headdim) , must input use to compute dq
    cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
        of the sequences in the batch, used to index into q.
    cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
        of the sequences in the batch, used to index into kv.
    alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        is added to the attention score of query i and key j.
    rng_state: 2 x Int64 , stored random state {seed , offset}
    max_seqlen_q: int. Maximum query sequence length in the batch.
    max_seqlen_k: int. Maximum key sequence length in the batch.
    p_dropout: float. Dropout probability.
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    deterministic: bool. Whether to use the deterministic implementation of the backward pass,
        which is slightly slower and uses more memory. The forward pass is always deterministic.
    stream: stream
    extend_parameter_: extend parameter, support softcap extend parameter
Return:
    mcflashattnStatus_t

### mha_fwd_kvcache

```
mcflashattnStatus_t
mha_fwd_kvcache(
    const Tensor_t q,           // batch_size x seqlen_q x num_heads x head_size
    const Tensor_t kcache,      // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const Tensor_t vcache,      // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const Tensor_t k_,                 // batch_size x seqlen_knew x num_heads_k x head_size
    const Tensor_t v_,                // batch_size x seqlen_knew x num_heads_k x head_size
    const Tensor_t seqlens_k_,        // batch_size
    const Tensor_t rotary_cos_,       // seqlen_ro x (rotary_dim / 2)
    const Tensor_t rotary_sin_,       // seqlen_ro x (rotary_dim / 2)
    const Tensor_t cache_batch_idx_,  // indices to index into the KV cache
    const Tensor_t block_table_,      // batch_size x max_num_blocks_per_seq
    const Tensor_t alibi_slopes_,     // num_heads or batch_size x num_heads
    const Tensor_t softmax_lse,     // batch_size x num_heads x seqlen_q
    Tensor_t out_,                 // batch_size x seqlen_q x num_heads x head_size
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
```

If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
the previous step, and update them with the new keys/values from the current step, and do
attention with the updated cache, all in 1 kernel.

If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
For example, the KV cache could be pre-allocated with the max sequence length, and you can use
cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
    1 1 1 1 0
    1 1 1 1 1
If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
    0 0
    0 0
    0 0
    1 0
    1 1
If the row of the mask is all zero, the output will be zero.

If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

Note: Does not support backward pass.

Arguments:
    q: (batch_size, seqlen, nheads, headdim)
    k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
        or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        page_block_size must be a multiple of 256.
    v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
        or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
    k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
        k with k_cache, starting at the indices specified by cache_seqlens.
    v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
    seqlens_k_:
    rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
        to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
    rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
    cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
        KV cache.
    block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
    cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
        If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
        If the indices are not distinct, and k and v are provided, the values updated in the cache
                might come from any of the duplicate indices.
    alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        is added to the attention score of query i and key j.
    softmax_lse: (batch_size,num_heads, seqlen_q), float32, the buffer of softmax lse
    out: (batch_size, seqlen, nheads, headdim).
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
        If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
        rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
        (i.e. GPT-NeoX style).
    num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
        If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
        to automatically determine the number of splits.
        Don't change this unless you know what you are doing.
    stream: stream
    extend_parameter_: extend parameter, support softcap extend parameter
Return:
    mcflashattnStatus_t
