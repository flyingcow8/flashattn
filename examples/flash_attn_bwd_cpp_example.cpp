#include <mcr/mc_runtime.h>
#include "flash_attn/flash_attn.h"
#include <common/maca_bfloat16.h>
#include <cmath>
#include <iostream>
#include <vector>

using bfloat16_t = __maca_bfloat16;

int main()
{
    int err_code = 0;
    std::vector<std::string> attn_types{"MHA", "GQA", "MQA"};
    for (auto attn_type : attn_types) {
        // 1. Set probelm size and parameters
        bool is_causal = true;
        int batch_size = 1;
        int num_heads_q = 40;
        int num_heads_k = attn_type == "MHA" ? num_heads_q : (attn_type == "MQA" ? 1: 20);
        int head_size_og = 128;
        int seqlen_q = 2048;
        int seqlen_k = 2048;

        float p_dropout = 0.17;
        float softmax_scale = float(1.0 / sqrt(float(head_size_og)));
        int window_size_left = -1;
        int window_size_right = -1;

        //----------------------------------------------------------------
        // 2. malloc host memory
        void *rng_state_ptr = nullptr;
        int64_t rng_state_size = 2;
        rng_state_ptr = malloc(rng_state_size * sizeof(int64_t)); // rng_state is host buffer


        //----------------------------------------------------------------
        // 3. malloc device memory
        void *dout_ptr = nullptr;
        void *q_ptr = nullptr;
        void *k_ptr = nullptr;
        void *v_ptr = nullptr;
        void *out_ptr = nullptr;
        void *softmax_d_ptr = nullptr;
        void *softmax_lse_ptr = nullptr;
        void *dq_ptr = nullptr;
        void *dk_ptr = nullptr;
        void *dv_ptr = nullptr;
        void *dq_accum_ptr = nullptr;

        int64_t q_size = batch_size * seqlen_q * num_heads_q * head_size_og;
        int64_t k_size = batch_size * seqlen_k * num_heads_k * head_size_og;
        int64_t v_size = batch_size * seqlen_k * num_heads_k * head_size_og;
        int64_t out_size = batch_size * seqlen_q * num_heads_q * head_size_og;
        int64_t dout_size = batch_size * seqlen_q * num_heads_q * head_size_og;
        int64_t softmax_d_size = batch_size * num_heads_q * seqlen_q;
        int64_t softmax_lse_size = batch_size * num_heads_q * seqlen_q;
        int64_t dq_size = batch_size * seqlen_q * num_heads_q * head_size_og;
        int64_t dk_size = batch_size * seqlen_k * num_heads_q * head_size_og;
        int64_t dv_size = batch_size * seqlen_k * num_heads_q * head_size_og;
        int64_t dq_accum_size = batch_size * seqlen_q * num_heads_q * head_size_og;

        mcMalloc((void **)(&q_ptr), q_size * sizeof(bfloat16_t));
        mcMalloc((void **)(&k_ptr), k_size * sizeof(bfloat16_t));
        mcMalloc((void **)(&v_ptr), v_size * sizeof(bfloat16_t));
        mcMalloc((void **)(&out_ptr), out_size * sizeof(bfloat16_t));
        mcMalloc((void **)(&dout_ptr), dout_size * sizeof(bfloat16_t));
        mcMalloc((void **)(&softmax_d_ptr), softmax_d_size * sizeof(float));
        mcMalloc((void **)(&softmax_lse_ptr), softmax_lse_size * sizeof(float));
        mcMalloc((void **)(&dq_ptr), dq_size * sizeof(bfloat16_t));
        mcMalloc((void **)(&dk_ptr), dk_size * sizeof(bfloat16_t));
        mcMalloc((void **)(&dv_ptr), dv_size * sizeof(bfloat16_t));
        mcMalloc((void **)(&dq_accum_ptr), dq_accum_size * sizeof(float));

        //----------------------------------------------------------------
        // 4. Create the tensor of data pointer
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

        auto tensor_dout = make_contiguous_tensor4d(
            dout_ptr,
            MCFLASHATTN_DATATYPE_BF16,
            batch_size,
            seqlen_q,
            num_heads_q,
            head_size_og
        );

        auto tensor_softmax_d = make_contiguous_tensor3d(
            softmax_d_ptr,
            MCFLASHATTN_DATATYPE_FP32,
            batch_size,
            num_heads_q,
            seqlen_q
        );

        auto tensor_softmax_lse = make_contiguous_tensor3d(
            softmax_lse_ptr,
            MCFLASHATTN_DATATYPE_FP32,
            batch_size,
            num_heads_q,
            seqlen_q
        );

        auto tensor_dq = make_contiguous_tensor4d(
            dq_ptr,
            MCFLASHATTN_DATATYPE_BF16,
            batch_size,
            seqlen_q,
            num_heads_q,
            head_size_og
        );

        auto tensor_dk = make_contiguous_tensor4d(
            dk_ptr,
            MCFLASHATTN_DATATYPE_BF16,
            batch_size,
            seqlen_k,
            num_heads_q,
            head_size_og
        );

        auto tensor_dv = make_contiguous_tensor4d(
            dv_ptr,
            MCFLASHATTN_DATATYPE_BF16,
            batch_size,
            seqlen_k,
            num_heads_q,
            head_size_og
        );

        auto tensor_dq_accum = make_contiguous_tensor4d(
            dq_accum_ptr,
            MCFLASHATTN_DATATYPE_FP32,
            batch_size,
            seqlen_q,
            num_heads_q,
            head_size_og
        );

        auto tensor_rng_state = make_contiguous_tensor1d(
            rng_state_ptr,
            MCFLASHATTN_DATATYPE_INT64,
            rng_state_size
        );

        //----------------------------------------------------------------
        // 5. Call bwd
        auto ret = mha_bwd(
            batch_size,
            seqlen_q,
            num_heads_q,
            seqlen_k,
            num_heads_k,
            head_size_og,
            tensor_dout,
            tensor_q,
            tensor_k,
            tensor_v,
            tensor_out,
            tensor_softmax_d,
            tensor_softmax_lse,
            tensor_dq,
            tensor_dk,
            tensor_dv,
            tensor_dq_accum,
            /*alibi_slopes_=*/nullptr,
            /*attn_mask_=*/nullptr,
            tensor_rng_state,
            p_dropout,
            softmax_scale,
            is_causal,
            window_size_left,
            window_size_right,
            /*deterministic=*/false,
            /*stream=*/nullptr
        );
        mcDeviceSynchronize();
        if (mcGetLastError() != mcSuccess) {
            throw std::runtime_error("run kernel failed");
        }

        if(ret != MCFLASHATTN_STATUS_SUCCESS){
            std::cerr << attn_type + " mha_bwd failed, ret:" << int(ret) << std::endl;
        } else {
            std::cerr << attn_type + " mha_bwd end" << std::endl;
        }
        err_code += ret;

        //----------------------------------------------------------------
        // 6. Release tensors
        // Don't free the device memory
        release_tensor(tensor_q);
        release_tensor(tensor_k);
        release_tensor(tensor_v);
        release_tensor(tensor_out);
        release_tensor(tensor_dout);
        release_tensor(tensor_softmax_d);
        release_tensor(tensor_softmax_lse);
        release_tensor(tensor_dq);
        release_tensor(tensor_dk);
        release_tensor(tensor_dv);
        release_tensor(tensor_dq_accum);
        release_tensor(tensor_rng_state);

        //----------------------------------------------------------------
        // 7. Free device memory
        mcFree(q_ptr);
        mcFree(k_ptr);
        mcFree(v_ptr);
        mcFree(out_ptr);
        mcFree(dout_ptr);
        mcFree(softmax_d_ptr);
        mcFree(softmax_lse_ptr);
        mcFree(dq_ptr);
        mcFree(dk_ptr);
        mcFree(dv_ptr);
        mcFree(dq_accum_ptr);

        //----------------------------------------------------------------
        //8. Free host memory
        free(rng_state_ptr);
    }

    return err_code;
}
