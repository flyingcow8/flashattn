#include <mcr/mc_runtime.h>
#include "flash_attn/flash_attn.h"
#include "mctlass/bfloat16.h"
#include <cmath>
#include <iostream>
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
    int batch_size = 2;
    int num_heads_q = 6;
    int num_heads_k = num_heads_q;
    int head_size_og = 128;
    int seqlen_q = 1248;
    int seqlen_k = 1324;

    float softmax_scale = float(1.0 / sqrt(float(head_size_og)));
    int window_size_left = -1;
    int window_size_right = -1;

    // 2. malloc the memory
    void *q_ptr = nullptr;
    void *k_ptr = nullptr;
    void *v_ptr = nullptr;
    void *out_ptr = nullptr;
    void *alibi_slopes_ptr = nullptr;
    void *attn_mask_ptr = nullptr;

    int64_t q_size = batch_size * seqlen_q * num_heads_q * head_size_og;
    int64_t k_size = batch_size * seqlen_k * num_heads_k * head_size_og;
    int64_t v_size = batch_size * seqlen_k * num_heads_k * head_size_og;
    int64_t out_size = batch_size * seqlen_q * num_heads_q * head_size_og;
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
    mcMalloc((void **)(&alibi_slopes_ptr), alibi_slopes_size * sizeof(mctlass::bfloat16_t));

    mcMemcpy(q_ptr,host_q.data(),q_size * sizeof(mctlass::bfloat16_t),mcMemcpyHostToDevice);
    mcMemcpy(k_ptr,host_k.data(),k_size * sizeof(mctlass::bfloat16_t),mcMemcpyHostToDevice);
    mcMemcpy(v_ptr,host_v.data(),v_size * sizeof(mctlass::bfloat16_t),mcMemcpyHostToDevice);

    // 3. Create the tensor of data pointer
    auto tensor_q = make_contiguous_tensor4d(
        q_ptr,
        MCFLASHATTN_DATATYPE_BF16,
        batch_size,
        seqlen_q,
        num_heads_q,
        head_size_og
    );

    auto tensor_k = make_contiguous_tensor4d(
        k_ptr,
        MCFLASHATTN_DATATYPE_BF16,
        batch_size,
        seqlen_k,
        num_heads_k,
        head_size_og
    );

    auto tensor_v = make_contiguous_tensor4d(
        v_ptr,
        MCFLASHATTN_DATATYPE_BF16,
        batch_size,
        seqlen_k,
        num_heads_k,
        head_size_og
    );

    auto tensor_out = make_contiguous_tensor4d(
        out_ptr,
        MCFLASHATTN_DATATYPE_BF16,
        batch_size,
        seqlen_q,
        num_heads_q,
        head_size_og
    );

    // 4. Call fwd
    // auto stream = mcGetCurrentStream();
    mcStream_t stream = nullptr;
    auto ret = mha_fwd_inference(
        batch_size,
        seqlen_q,
        num_heads_q,
        seqlen_k,
        num_heads_k,
        head_size_og,
        tensor_q,
        tensor_k,
        tensor_v,
        tensor_out,
        nullptr,
        nullptr,
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

    // 6. Free device memory
    mcFree(q_ptr);
    mcFree(k_ptr);
    mcFree(v_ptr);
    mcFree(out_ptr);
}