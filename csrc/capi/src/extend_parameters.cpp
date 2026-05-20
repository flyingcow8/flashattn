#include "flash_attn.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void set_extend_parameter_softcap(mcflashattnExtendParameter_t extend_param, const float softcap) {
    extend_param->softcap = softcap;
    return;
}

void set_extend_parameter_leftpad(mcflashattnExtendParameter_t extend_param, void *data, int padding_size) {
    Tensor_t leftpad_k_ = make_tensor1d(data, MCFLASHATTN_DATATYPE_INT32, padding_size, 1);
    extend_param->leftpad_k_ = leftpad_k_;
    return;
}

void set_extend_parameter_block_table(mcflashattnExtendParameter_t extend_param, void *data, int block_table_size_m,
                                     int block_table_size_n) {
    Tensor_t block_table_ =
        make_tensor2d(data, MCFLASHATTN_DATATYPE_INT32, block_table_size_m, block_table_size_n, block_table_size_n, 1);
    extend_param->block_table_ = block_table_;
    return;
}

float get_extend_parameter_softcap(mcflashattnExtendParameter_t extend_param) {
    float softcap = extend_param->softcap;
    return softcap;
}

Tensor_t get_extend_parameter_leftpad(mcflashattnExtendParameter_t extend_param) {
    Tensor_t leftpad_k_ = extend_param->leftpad_k_;
    return leftpad_k_;
}

Tensor_t get_extend_parameter_block_table(mcflashattnExtendParameter_t extend_param) {
    Tensor_t block_table_ = extend_param->block_table_;
    return block_table_;
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
