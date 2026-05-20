#include <mcr/mc_runtime.h>
#include "flash_attn/flash_attn.h"
#include <common/maca_bfloat16.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using bfloat16_t = __maca_bfloat16;

uint16_t rand_uint16(){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::uniform_int_distribution<uint16_t> distribution(0, 256);

    return distribution(generator);
}

void random_vector(std::vector<bfloat16_t> &vec){
    for(int i = 0; i < vec.size(); ++i){
        uint16_t val = rand_uint16();
        vec[i] = *reinterpret_cast<bfloat16_t*>(&val);
    }
}


int main()
{
    // 1. Set probelm size and parameters
    bool is_causal = true;
    int num_heads_q = 6;
    int num_heads_k = num_heads_q;
    int head_size_og = 128;

    ///////////////////////////////////////////
    // Set varlen q and k length
    int batch_size = 2;
    int total_q = 1125; // 25 + 1100
    int total_k = 1342; // 42 + 1300

    std::vector<int> host_seqlens_q = {0,25,1125}; // batch_size + 1
    std::vector<int> host_seqlens_k = {0,42,1342}; // batch_size + 1

    int max_seqlen_q = 1100;
    int max_seqlen_k = 1300;
    ///////////////////////////////////////////////////

    float softmax_scale = float(1.0 / sqrt(float(head_size_og)));
    int window_size_left = -1;
    int window_size_right = -1;
    float p_dropout = 0.17;

    // 2. malloc the memory

    void *rng_state_ptr = nullptr;
    int64_t rng_state_size = 2;
    rng_state_ptr = malloc(rng_state_size * sizeof(int64_t));

    void *q_ptr = nullptr;
    void *k_ptr = nullptr;
    void *v_ptr = nullptr;
    void *out_ptr = nullptr;
    void *alibi_slopes_ptr = nullptr;
    void *cu_seqlens_q_ptr = nullptr;
    void *cu_seqlens_k_ptr = nullptr;
    void *p_ptr = nullptr;
    void *softmax_lse_ptr = nullptr;

    int64_t q_size = total_q * num_heads_q * head_size_og;
    int64_t k_size = total_k * num_heads_k * head_size_og;
    int64_t v_size = total_k * num_heads_k * head_size_og;
    int64_t out_size = total_q * num_heads_q * head_size_og;
    int64_t alibi_slopes_size = batch_size * num_heads_q; // or num_heads
    int64_t softmax_lse_size = batch_size * num_heads_q * max_seqlen_q;

    std::vector<bfloat16_t> host_q(q_size);
    std::vector<bfloat16_t> host_k(k_size);
    std::vector<bfloat16_t> host_v(v_size);
    random_vector(host_q);
    random_vector(host_k);
    random_vector(host_v);

    mcMalloc((void **)(&q_ptr), q_size * sizeof(bfloat16_t));
    mcMalloc((void **)(&k_ptr), k_size * sizeof(bfloat16_t));
    mcMalloc((void **)(&v_ptr), v_size * sizeof(bfloat16_t));
    mcMalloc((void **)(&out_ptr), out_size * sizeof(bfloat16_t));
    mcMalloc((void **)(&softmax_lse_ptr), softmax_lse_size * sizeof(float));
    // mcMalloc((void **)(&alibi_slopes_ptr), alibi_slopes_size * sizeof(float));

    mcMemcpy(q_ptr,host_q.data(),q_size * sizeof(bfloat16_t),mcMemcpyHostToDevice);
    mcMemcpy(k_ptr,host_k.data(),k_size * sizeof(bfloat16_t),mcMemcpyHostToDevice);
    mcMemcpy(v_ptr,host_v.data(),v_size * sizeof(bfloat16_t),mcMemcpyHostToDevice);

    int64_t cu_seqlen_size = batch_size + 1;
    mcMalloc((void **)(&cu_seqlens_q_ptr), cu_seqlen_size * sizeof(int32_t));
    mcMalloc((void **)(&cu_seqlens_k_ptr), cu_seqlen_size * sizeof(int32_t));

    mcMemcpy(cu_seqlens_q_ptr,host_seqlens_q.data(),cu_seqlen_size * sizeof(int32_t),mcMemcpyHostToDevice);
    mcMemcpy(cu_seqlens_k_ptr,host_seqlens_k.data(),cu_seqlen_size * sizeof(int32_t),mcMemcpyHostToDevice);

    // 3. Create the tensor of data pointer
    auto tensor_q = make_contiguous_tensor3d(
        q_ptr,
        MCFLASHATTN_DATATYPE_BF16,
        total_q,
        num_heads_q,
        head_size_og
    );

    auto tensor_k = make_contiguous_tensor3d(
        k_ptr,
        MCFLASHATTN_DATATYPE_BF16,
        total_k,
        num_heads_k,
        head_size_og
    );

    auto tensor_v = make_contiguous_tensor3d(
        v_ptr,
        MCFLASHATTN_DATATYPE_BF16,
        total_k,
        num_heads_k,
        head_size_og
    );

    auto tensor_out = make_contiguous_tensor3d(
        out_ptr,
        MCFLASHATTN_DATATYPE_BF16,
        total_q,
        num_heads_q,
        head_size_og
    );

    auto tensor_cu_seqlens_q = make_contiguous_tensor1d(
        cu_seqlens_q_ptr,
        MCFLASHATTN_DATATYPE_INT32,
        batch_size + 1
    );

    auto tensor_cu_seqlens_k = make_contiguous_tensor1d(
        cu_seqlens_k_ptr,
        MCFLASHATTN_DATATYPE_INT32,
        batch_size + 1
    );

    auto tensor_softmax_lse = make_contiguous_tensor3d(
        softmax_lse_ptr,
        MCFLASHATTN_DATATYPE_FP32,
        batch_size,
        num_heads_q,
        max_seqlen_q
    );

    auto tensor_rng_state = make_contiguous_tensor1d(
        rng_state_ptr,
        MCFLASHATTN_DATATYPE_INT64,
        rng_state_size
    );


    // 4. Call fwd
    // auto stream = mcGetCurrentStream();
    mcStream_t stream = nullptr;
    auto ret = mha_varlen_fwd(
        batch_size,
        total_q,
        num_heads_q,
        total_k,
        num_heads_k,
        head_size_og,
        tensor_q,
        tensor_k,
        tensor_v,
        tensor_out,
        tensor_cu_seqlens_q,
        tensor_cu_seqlens_k,
        nullptr, // seqused_k
        nullptr, // alibi_slopes
        tensor_softmax_lse,
        nullptr,  // p
        tensor_rng_state,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        stream
    );
    mcDeviceSynchronize();
    if (mcGetLastError() != mcSuccess) {
        throw std::runtime_error("run kernel failed");
    }

    std::vector<bfloat16_t> host_out(out_size);

    if(ret != MCFLASHATTN_STATUS_SUCCESS){
        std::cerr << "mha_varlen_fwd failed, ret:" << int(ret) << std::endl;
    }else {
        std::cout << "mha_varlen_fwd success" << std::endl;
        mcMemcpy(host_out.data(),out_ptr,out_size * sizeof(bfloat16_t),mcMemcpyDeviceToHost);
    }

    // 5. Release tensor
    // Don't free the device memory
    release_tensor(tensor_q);
    release_tensor(tensor_k);
    release_tensor(tensor_v);
    release_tensor(tensor_out);
    release_tensor(tensor_cu_seqlens_q);
    release_tensor(tensor_cu_seqlens_k);
    release_tensor(tensor_softmax_lse);
    release_tensor(tensor_rng_state);

    // 6. Free device memory and host memory
    mcFree(q_ptr);
    mcFree(k_ptr);
    mcFree(v_ptr);
    mcFree(out_ptr);
    mcFree(cu_seqlens_q_ptr);
    mcFree(cu_seqlens_k_ptr);
    mcFree(softmax_lse_ptr);

    free(rng_state_ptr);

    return int(ret);
}
