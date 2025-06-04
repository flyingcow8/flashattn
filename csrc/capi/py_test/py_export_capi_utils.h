#pragma once

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include "flash_attn.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) \
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

inline void get_philox_state(c10::optional<at::Generator> &gen_, at::Tensor &rng_state, int64_t counter_offset) {
    TORCH_CHECK(rng_state.dtype() == torch::kInt64, "rng_state tensor must have dtype Int64");
    TORCH_CHECK(rng_state.stride(-1) == 1, "rng_state tensor must have contiguous last dimension");
    CHECK_SHAPE(rng_state, 2);

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_, at::cuda::detail::getDefaultCUDAGenerator());

    std::lock_guard<std::mutex> lock(gen->mutex_);

    auto philox_args = gen->philox_cuda_state(counter_offset);
    auto seeds = at::cuda::philox::unpack(philox_args);

    uint64_t *rng_state_p;
    rng_state_p = reinterpret_cast<uint64_t *>(rng_state.data_ptr());
    rng_state_p[0] = std::get<0>(seeds);
    rng_state_p[1] = std::get<1>(seeds);
}

inline Tensor_t convert_mcfa_tensor(const at::Tensor &at_tensor) {
    auto sizes = at_tensor.sizes();
    auto dims = sizes.size();

    if (dims <= 0 || (dims == 1 && sizes[0] == 0)) {
        return nullptr;
    }

    auto dtype = at_tensor.dtype();
    mcflashattnDataType_t mcft_dtype = MCFLASHATTN_DATATYPE_NONE;

    if (dtype == torch::kFloat16)
        mcft_dtype = MCFLASHATTN_DATATYPE_FP16;
    else if (dtype == torch::kBFloat16)
        mcft_dtype = MCFLASHATTN_DATATYPE_BF16;
    else if (dtype == torch::kFloat)
        mcft_dtype = MCFLASHATTN_DATATYPE_FP32;
    else if (dtype == torch::kDouble)
        mcft_dtype = MCFLASHATTN_DATATYPE_FP64;
    else if (dtype == torch::kInt32)
        mcft_dtype = MCFLASHATTN_DATATYPE_INT32;
    else if (dtype == torch::kInt64)
        mcft_dtype = MCFLASHATTN_DATATYPE_INT64;
    else if (dtype == torch::kInt8)
        mcft_dtype = MCFLASHATTN_DATATYPE_INT8;
    else
        assert(0);

    auto strides = at_tensor.strides();
    if (dims == 1) {
        return make_tensor1d(at_tensor.data_ptr(), mcft_dtype, sizes[0], strides[0]);
    } else if (dims == 2) {
        return make_tensor2d(at_tensor.data_ptr(), mcft_dtype, sizes[0], sizes[1], strides[0], strides[1]);
    } else if (dims == 3) {
        return make_tensor3d(at_tensor.data_ptr(), mcft_dtype, sizes[0], sizes[1], sizes[2], strides[0], strides[1],
                             strides[2]);
    } else if (dims == 4) {
        return make_tensor4d(at_tensor.data_ptr(), mcft_dtype, sizes[0], sizes[1], sizes[2], sizes[3], strides[0],
                             strides[1], strides[2], strides[3]);
    } else if (dims == 5) {
        return make_tensor5d(at_tensor.data_ptr(), mcft_dtype, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],
                             strides[0], strides[1], strides[2], strides[3], strides[4]);
    } else {
        return nullptr;
    }
}

inline Tensor_t convert_mcfa_tensor(const c10::optional<at::Tensor> &at_tensor) {
    if (at_tensor.has_value()) {
        return convert_mcfa_tensor(at_tensor.value());
    }

    return nullptr;
}

inline void release_mcfa_tensor(const std::initializer_list<Tensor_t> &list) {
    for (Tensor_t tensor : list) {
        release_tensor(tensor);
    }
}

inline void print_torch_tensor(const std::string &tag, const at::Tensor &at_tensor) {
    auto sizes = at_tensor.sizes();
    auto dims = sizes.size();
    std::cout << ">>>>> torch " << tag << std::endl;
    std::cout << "size:[";
    for (int i = 0; i < dims; i++) {
        std::cout << sizes[i];
        if (i < dims - 1) {
            std::cout << ",";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "stride:[";
    for (int i = 0; i < dims; i++) {
        std::cout << at_tensor.stride(i);
        if (i < dims - 1) {
            std::cout << ",";
        }
    }
    std::cout << "]" << std::endl;
}

inline void print_torch_tensor(const std::string &tag, const c10::optional<at::Tensor> &at_tensor) {
    if (at_tensor.has_value()) {
        print_torch_tensor(tag, at_tensor.value());
    } else {
        std::cout << ">>>>>> torch " << tag << std::endl;
        std::cout << "null" << std::endl;
    }
}

inline void print_mcfa_tensor(const std::string &tag, Tensor_t tensor) {
    std::cout << ">>>>>> mcft " << tag << std::endl;
    if (tensor == nullptr) {
        std::cout << "null" << std::endl;
    } else {
        print_tensor_info(tensor);
    }
}