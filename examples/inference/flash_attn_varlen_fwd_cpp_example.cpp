#include <mcr/mc_runtime.h>
#include "flash_attn/flash_attn.h"
#include "mctlass/bfloat16.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

uint16_t rand_uint16(){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::uniform_int_distribution<uint16_t> distribution(0, 256);

    return distribution(generator);
}

void random_vector(std::vector<mctlass::bfloat16_t> &vec){
    for(int i = 0; i < vec.size(); ++i){
        uint16_t val = rand_uint16();
        vec[i] = *reinterpret_cast<mctlass::bfloat16_t*>(&val);
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

    // 2. malloc the memory
    void *q_ptr = nullptr;
    void *k_ptr = nullptr;
    void *v_ptr = nullptr;
    void *out_ptr = nullptr;
    void *alibi_slopes_ptr = nullptr;
    void *cu_seqlens_q_ptr = nullptr;
    void *cu_seqlens_k_ptr = nullptr;

    int64_t q_size = total_q * num_heads_q * head_size_og;
    int64_t k_size = total_k * num_heads_k * head_size_og;
    int64_t v_size = total_k * num_heads_k * head_size_og;
    int64_t out_size = total_q * num_heads_q * head_size_og;
    int64_t alibi_slopes_size = batch_size * num_heads_q; // or num_heads

    std::vector<mctlass::bfloat16_t> host_q(q_size);
    std::vector<mctlass::bfloat16_t> host_k(k_size);
    std::vector<mctlass::bfloat16_t> host_v(v_size);
    random_vector(host_q);
    random_vector(host_k);
    random_vector(host_v);

    mcMalloc((void **)(&q_ptr), q_size * sizeof(mctlass::bfloat16_t));
    mcMalloc((void **)(&k_ptr), k_size * sizeof(mctlass::bfloat16_t));
    mcMalloc((void **)(&v_ptr), v_size * sizeof(mctlass::bfloat16_t));
    mcMalloc((void **)(&out_ptr), out_size * sizeof(mctlass::bfloat16_t));
    // mcMalloc((void **)(&alibi_slopes_ptr), alibi_slopes_size * sizeof(float));

    mcMemcpy(q_ptr,host_q.data(),q_size * sizeof(mctlass::bfloat16_t),mcMemcpyHostToDevice);
    mcMemcpy(k_ptr,host_k.data(),k_size * sizeof(mctlass::bfloat16_t),mcMemcpyHostToDevice);
    mcMemcpy(v_ptr,host_v.data(),v_size * sizeof(mctlass::bfloat16_t),mcMemcpyHostToDevice);

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

    // 4. Call fwd
    // auto stream = mcGetCurrentStream();
    mcStream_t stream = nullptr;
    auto ret = mha_varlen_fwd_inference(
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
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        stream
    );

    std::vector<mctlass::bfloat16_t> host_out(out_size);

    if(ret != MCFLASHATTN_STATUS_SUCCESS){
        std::cerr << "mha_fwd_inference failed, ret:" << int(ret) << std::endl;
    }else {
        std::cout << "mha_fwd_inference success" << std::endl;
        mcMemcpy(host_out.data(),out_ptr,out_size * sizeof(mctlass::bfloat16_t),mcMemcpyDeviceToHost);
    }

    // 5. Release tensor
    // Don't free the device memory
    release_tensor(tensor_q);
    release_tensor(tensor_k);
    release_tensor(tensor_v);
    release_tensor(tensor_out);
    release_tensor(tensor_cu_seqlens_q);
    release_tensor(tensor_cu_seqlens_k);

    // 6. Free device memory
    mcFree(q_ptr);
    mcFree(k_ptr);
    mcFree(v_ptr);
    mcFree(out_ptr);
    mcFree(cu_seqlens_q_ptr);
    mcFree(cu_seqlens_k_ptr);
}