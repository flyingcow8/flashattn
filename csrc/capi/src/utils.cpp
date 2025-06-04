#include "utils.h"

bool check_tensor(const std::initializer_list<Tensor_t> &tensor_list) {
    for (auto tensor : tensor_list) {
        if (tensor == nullptr) return false;
        if (tensor->data == nullptr) return false;

        auto t = (InternalTensor *)tensor->data;

        if (t->strides[t->strides.size() - 1] != 1) {
            std::cerr << "Input tensor must have contiguous last dimension" << std::endl;

            return false;
        }
    }

    return true;
}

bool check_tensor_type(const std::initializer_list<Tensor_t> &tensor_list, InternalTensor::DataType dtype) {
    for (auto tensor : tensor_list) {
        auto t = (InternalTensor *)tensor->data;

        if (t->dtype != dtype) {
            return false;
        }
    }

    return true;
}

bool check_tensor_shape(const std::initializer_list<Tensor_t> &tensor_list, std::vector<int64_t> shape) {
    for (auto tensor : tensor_list) {
        if (tensor == nullptr) return false;
        if (tensor->data == nullptr) return false;

        auto t = (InternalTensor *)tensor->data;

        if (t->sizes.size() != shape.size()) {
            std::cerr << "Input tensor must have same shape size" << std::endl;

            return false;
        } else if (t->sizes != shape) {
            std::cerr << "Input tensor must have same shape" << std::endl;

            return false;
        }
    }

    return true;
}