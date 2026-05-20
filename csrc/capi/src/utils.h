#pragma once

#include "flash_attn.h"
#include "tensor.h"

bool check_continues(const Tensor_t &tensor);
bool check_dtype(const Tensor_t &tensor, InternalTensor::DataType dtype);
bool check_shape(const Tensor_t &tensor, std::vector<int64_t> shape);


#define TENSOR_CHECK(condition, msg)                                           \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::cerr << "[Error] " << __FILE__ <<  ":" << __LINE__            \
                      << "  `" #condition "` is false, " << (msg) << '\n';     \
            return MCFLASHATTN_STATUS_ILLEGAL_TENSOR;                          \
        }                                                                      \
    } while (0)


#define CHECK_CONTIGUOUS(x) TENSOR_CHECK(check_continues(x), #x " must be contiguous at last dimension")

#define CHECK_DTYPE(x, dtype) TENSOR_CHECK(check_dtype(x, dtype), #x " must be the same dtype with given")

#define CHECK_SHAPE(x, ...) TENSOR_CHECK(check_shape(x, {__VA_ARGS__}), #x " must be the same shape with given")



#define CHECK_THROW(condition, msg)                                        \
    do {                                                                       \
        if(!(condition)) {                                                     \
            std::cerr << "[Error] " << __FILE__ <<  ":" << __LINE__            \
                      << "  `" #condition "` is false, " << (msg) << '\n';     \
            throw std::invalid_argument(msg);                                  \
        }                                                                      \
    }while(0)
