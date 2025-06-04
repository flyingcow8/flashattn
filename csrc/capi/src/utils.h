#pragma once

#include "flash_attn.h"
#include "tensor.h"
#include <initializer_list>

#define CHECK_MSG(x, ...)                             \
    do {                                              \
        if ((x) == false) {                           \
            throw std::invalid_argument(__VA_ARGS__); \
        }                                             \
    } while (0)

bool check_tensor(const std::initializer_list<Tensor_t> &tensor_list);
bool check_tensor_type(const std::initializer_list<Tensor_t> &tensor_list, InternalTensor::DataType dtype);
bool check_tensor_shape(const std::initializer_list<Tensor_t> &tensor_list, std::vector<int64_t> shape);