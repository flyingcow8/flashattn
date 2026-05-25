# FlashAttention
We provide the implementation of FlashAttention-2(version 2.6.3) from Tri Dao, based on MACA toolkit and C500 chips.

**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
Tri Dao

Paper: https://tridao.me/publications/flash2/flash2.pdf

FlashAttention-2 currently supports:
1. Datatype fp16 and bf16.
2. Most head dimensions up to 256 work correctly. Performance of headdim 64 and 128 are specially optimized.


## How to build FlashAttention

Requirements:
- Linux.
- MACA SDK(software development toolkit).
- mcPytorch2.0 and above. Conda virtual environment is highly recommended.
- cu-bridge(Need to install **separately** after maca2.33.0)
- PyYAML(For run kernel generator)

### Set environment variables
```bash
export MACA_PATH=/your/maca/path
export CUDA_PATH=$MACA_PATH/tools/cu-bridge
export MACA_CLANG_PATH=$MACA_PATH/mxgpu_llvm/bin
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:$LD_LIBRARY_PATH
```

### Install cu-bridge
```bash
# download cu-bridge deb and unpack the deb package
dpkg-deb -x your-cu-bridge-deb.deb ./
# copy the cu-bridge to your MACA_PATH
cp -r ./opt/maca-*/* $MACA_PATH
```

### Build and compile
```bash
# build python api wheel package
make python
# build c api shared object
make cplus_api
# only build device kernel
make kernel
```

### Fast build

Use `DEFAULT` to compile only the default dispatch MN tiles. Forward feature variants are disabled by default to reduce build time. Backward kernels are also disabled by default; because dropout forward is only useful together with backward in this build flow, dropout forward kernels are disabled when `BUILD_WITH_BWD_KERNEL=FALSE`.

Running `make python` with no extra options uses these defaults:

| Option | Default used by `make python` | Effect when changed |
| --- | --- | --- |
| `FLASHATTN_BUILD_PROJECTS` | unset | If unset, build both C500 and C600. Set `FLASHATTN_BUILD_PROJECTS=C500` or `C600` to build one architecture. |
| `HDIM_LIST` | `128 256` | Select the head dimensions to compile. |
| `DTYPE` | `BF16` | Select the dtype to compile. |
| `FWD_MN_LIST` | `DEFAULT` | Select xcore1000/xcore1500 fwd MN tiles; `DEFAULT` means the dispatch default tiles for each architecture. |
| `FWD_SPLIT_MN_LIST` | `DEFAULT` | Select xcore1000/xcore1500 fwd_split MN tiles; `DEFAULT` means the dispatch default tiles for each architecture. |
| `BUILD_WITH_BWD_KERNEL` | `FALSE` | Set to `TRUE` to build backward kernels and enable backward API support. |
| `FWD_ENABLE_LOCAL` | `FALSE` | Set to `TRUE` to build local/sliding-window forward variants. |
| `FWD_ENABLE_ALIBI` | `FALSE` | Set to `TRUE` to build ALiBi forward variants. |
| `FWD_ENABLE_SOFTCAP` | `FALSE` | Set to `TRUE` to build softcap forward variants. |
| `FWD_ENABLE_APPENDKV` | `FALSE` | Set to `TRUE` to build append-KV variants for `flash_attn_with_kvcache`. |
| `FWD_ENABLE_CAUSAL` | `FALSE` | Set to `TRUE` to build causal forward variants. |

With these defaults, `make python` builds BF16 forward-only kernels for hdim 128 and 256, uses only the default dispatch MN tiles, disables backward/dropout, and excludes local, ALiBi, softcap, append-KV, and causal forward variants. The default MN tiles are:

| arch | hdim | fwd default MN | fwd_split default MN |
| --- | --- | --- | --- |
| xcore1000 | 128 | 64x64 | 64x64 |
| xcore1000 | 256 | 64x32 | 64x64 |
| xcore1500 | 128 | 128x64 | 16x32, 128x64 |
| xcore1500 | 256 | 128x64 | 128x64 |

Enable only the variants needed by your workload:
```bash
# build only C500
FLASHATTN_BUILD_PROJECTS=C500 make python

# build backward and dropout-capable forward kernels
make python BUILD_WITH_BWD_KERNEL=TRUE

# support causal=True
make python FWD_ENABLE_CAUSAL=TRUE

# support local attention and ALiBi
make python FWD_ENABLE_LOCAL=TRUE FWD_ENABLE_ALIBI=TRUE

# support append KV in flash_attn_with_kvcache
make python FWD_ENABLE_APPENDKV=TRUE

# enable multiple variants together
make python BUILD_WITH_BWD_KERNEL=TRUE FWD_ENABLE_CAUSAL=TRUE FWD_ENABLE_LOCAL=TRUE
```

Override `FWD_MN_LIST` and `FWD_SPLIT_MN_LIST` to include more forward tiles:
```bash
make python FWD_MN_LIST=64x32,64x64 FWD_SPLIT_MN_LIST=64x64
```

The generated xcore1000 sources currently provide these MN choices:

| hdim | fwd MN choices | fwd_split MN choices | dispatch default |
| --- | --- | --- | --- |
| 32 | 128x64, 128x128 | 64x64 | fwd: 128x128; fwd_split: 64x64 |
| 64 | 16x16, 32x32, 64x64, 128x64, 128x128 | 16x16, 64x64 | fwd: 64x64; fwd_split: 64x64 |
| 96 | 64x64, 128x64 | 64x64 | fwd: 128x64; fwd_split: 64x64 |
| 128 | 64x32, 64x64, 128x32, 128x64 | 16x16, 32x32, 64x32, 64x64, 128x64 | fwd: 64x64; fwd_split: 64x64 |
| 160 | 64x32, 64x64, 128x64 | 64x64 | fwd: 64x32; fwd_split: 64x64 |
| 192 | 64x64; 128x64 for hdimv128 | 64x64 | fwd: 64x64 and 128x64 for hdimv128; fwd_split: 64x64 |
| 256 | 64x32, 64x64 | 64x32, 64x64 | fwd: 64x32 without dropout, 64x64 for dropout; fwd_split: 64x64 |
| 512 | 64x32 | 32x32 | fwd: 64x32; fwd_split: 32x32 |

The generated xcore1500 sources currently provide these MN choices:

| hdim | fwd MN choices | fwd_split MN choices | dispatch default |
| --- | --- | --- | --- |
| 32 | 128x64, 128x128 | 64x64 | fwd: 128x64 and 128x128; fwd_split: 64x64 |
| 64 | 128x64 | 128x64 | fwd: 128x64; fwd_split: 128x64 |
| 96 | 128x64 | 64x64 | fwd: 128x64; fwd_split: 64x64 |
| 128 | 128x64 | 16x32, 128x64 | fwd: 128x64; fwd_split: 16x32 for short seqlen_q, 128x64 otherwise |
| 160 | 128x64 | 64x64 | fwd: 128x64; fwd_split: 64x64 |
| 192 | 128x64 | 128x64 | fwd: 128x64; fwd_split: 128x64 |
| 256 | 128x64 | 128x64 | fwd: 128x64; fwd_split: 128x64 |

### ‌Multi-SKU build

The build process can be controlled through the environment variable `FLASHATTN_BUILD_PROJECTS`:

- By default, `FLASHATTN_BUILD_PROJECTS` is **not set**, and the build system will compile for multiple SDKs (C500 and C600).
- To build for a specific architecture, set `FLASHATTN_BUILD_PROJECTS` to the desired architecture:
  ```bash
  export FLASHATTN_BUILD_PROJECTS=C500
  # or
  export FLASHATTN_BUILD_PROJECTS=C600
  ```

**Note**: The submodule `sage_attn` currently only supports C500 architecture builds.

For more details, refer to these build scripts in `tools/build_scripts`:

- `build_projects_related.sh`
  Determines build projects based on:
  1. The `FLASHATTN_BUILD_PROJECTS` environment variable
  2. Build type (SDK vs PYTORCH)
- `torch_extension_related.sh`
  Handles modification/retrieval of architecture flags added in torch `cpp_extension.py` script.

## Installation

Requirements:
- Linux.
- MACA SDK(software development toolkit).
- mcPytorch2.0 and above. Conda virtual environment is highly recommended.

To install flash attn in conda env:
1. Make sure that mcPytorch is installed.
```
pip install torch-of-specific-version-in-maca-wheel.whl --force-reinstall --no-deps
```
2. Install flash-attn (`pip install your_packages`)
```
pip install flash-attn-of-specific-version-in-maca-wheel.whl
```


## How to use FlashAttention
### Set environment variables
```bash
export MACA_PATH=/your/maca/path
export CUDA_PATH=$MACA_PATH/tools/cu-bridge
export MACA_CLANG_PATH=$MACA_PATH/mxgpu_llvm/bin
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:$LD_LIBRARY_PATH
```

### Python APIs
The main functions implement scaled dot product attention (softmax(Q @ K^T * softmax_scale) @ V).
Interface: `flash_attn/flash_attention_interface.py`:

```python
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
```

```python
flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    attn_mask=None,
    deterministic=False,
    return_attn_probs=False,
    softcap=0.0,  # <=0.0 means deactivate
    s_aux=None,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        softcap: float. Anything > 0 activates softcapping attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
```

```python
flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    attn_mask=None,
    softcap=0.0,  # 0.0 means deactivated
    s_aux=None,
):
    """dropout_p should be set to 0.0 during evaluation
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
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        softcap: float. Anything > 0 activates softcapping attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
```

```python
flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    softcap=0.0,  # 0.0 means deactivated
    s_aux=None,
):
    """
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

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

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
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        softcap: float. Anything > 0 activates softcapping attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
```

```python

def flash_attn_with_kvcache_dequant(
    q,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    dequant_group=-1,
    softcap=0.0,  # 0.0 means deactivated
):

    """
    Performs attention with INT8 key/value cache, dequanting kv cache with fp16/bf16 key/value scale.

    the dequantization is applied as:
    k_cache = k_cache * kscale, v_cache = v_cache * vscale.
    Note:
        Currently only supports cached key/value (k and v must be None)
        Dequantization group size must be 8 elements for hardware acceleration compatibility

    Arguments:
        q: (batch_size, seqlen, nheads, headdim), torch.float16/bfloat16
        k_cache: (num_blocks, page_block_size, nheads_k, headdim) with block_table,
                torch.int8. Page block size must be multiple of 256.
        v_cache: (num_blocks, page_block_size, nheads_k, headdim) with block_table,
                torch.int8
        kscale: (num_blocks, page_block_size, nheads_k, headdim/dequant_group), torch.float16/bfloat16. Dequantization scale for k_cache
        vscale: (num_blocks, page_block_size, nheads_k, headdim/dequant_group), torch.float16/bfloat16. Dequantization scale for v_cache
        cache_seqlens: int or (batch_size,), torch.int32. Current sequence lengths in KV cache
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32
        softmax_scale: float. QK^T scaling before softmax. Default: 1/sqrt(headdim)
        causal: bool. Applies causal mask if True
        window_size: (left, right). Sliding window attention when != (-1, -1)
        num_splits: int. Key/value chunking control. 0 for auto-determination.

    Return:
        out: (batch_size, seqlen, nheads, headdim)
        softmax_lse [optional]: (batch_size, nheads, seqlen). Logsumexp values
    """

```

```python
flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    softcap=0.0,  # 0.0 means deactivated
    block_table=None,
    s_aux=None,
):
    """dropout_p should be set to 0.0 during evaluation
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
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        If block_table is not none, the shape changes to (num_blocks, page_block_size, nheads_k, headdim)
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        If block_table is not none, the shape changes to (num_blocks, page_block_size, nheads_k, headdim)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        softcap: float. Anything > 0 activates softcapping attention.
        block_table: Optional block_table of dtype int32 and shape [batch_size, num_blocks_per_seq] used for paged attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """

```

### Python tests
We test that FlashAttention produces the same output and gradient as a reference
implementation, up to some numerical tolerance. In particular, we check that the
maximum numerical error of FlashAttention is at most twice the numerical error
of a baseline implementation in Pytorch (for different head dimensions, input
dtype, sequence length, causal / non-causal).

To run the tests:
```sh
pytest -q -s tests/test_flash_attn.py
```


### Using C++ APIs
At the present release version, forward MHA and decoder attention support C++ APIs via header
 `maca/include/flash_attn/flash_attn.h`.

```
namespace mcFlashAttn {

  struct Qkv_params;
  struct Flash_fwd_params;

  // run_mha_fwd_ supported headdims:  [32, 64, 96, 128, 160, 192, 256]
  //              supported datatypes: [mctlass::half_t, mctlass::bfloat16_t]
  template <typename T, int Headdim>
  void run_mha_fwd_(Flash_fwd_params &params, mcStream_t stream);

  // run_mha_fwd_splitkv_dispatch supported headdims:  [32, 64, 96, 128, 160]
  //                              supported datatypes: [mctlass::half_t, mctlass::bfloat16_t]
  template <typename T, int Headdim>
  void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, mcStream_t stream);

} // namespace mcFlashAttn
```

NOTE:
1. Dropout is not supported yet.
2. If num_splits > 0, extra device memory is needed, refering to `set_params_splitkv` in `csrc/flash_attn/flash_api.cpp`.

### C++ APIs example
C++ APIs is used with namespace `mcFlashAttn`.
```
mcFlashAttn::Flash_fwd_params params;  // use struct with namespace
mcFlashAttn::run_mha_fwd_splitkv_dispatch<mctlass::bfloat16_t, 128>(params, nullptr);  // call APIs with namespace
```

For more detailed usage of C++ APIs, please refer to examples/cpp_sample/flash_attn_decoder_cpp_example.cpp.
This example is trimed from csrc/flash_attn/flash_api.cpp, and used to test that C++ APIs can be called from cpp file,
and it does not include any functional checking.
