cmake_minimum_required(VERSION 3.10)

project(ft_attention LANGUAGES CXX C CUDA)

add_compile_options(-std=c++17)

add_definitions(-DUSE_MACA) # for PyTorch of maca
add_definitions(-DNV_ARCH_A100)
add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -isystem --expt-relaxed-constexpr --expt-extended-lambda")

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${CONDA_ENV_DIR}/include/python3.8
    ${CONDA_ENV_DIR}/lib/python3.8/site-packages/torch/include
    ${CONDA_ENV_DIR}/lib/python3.8/site-packages/torch/include/torch/csrc/api/include
    ${CONDA_ENV_DIR}/lib/python3.8/site-packages/torch/include/TH
    ${CONDA_ENV_DIR}/lib/python3.8/site-packages/torch/include/THC

    ${MACA_PATH}/tools/wcuda/include
    ${MACA_PATH}/include
    ${MACA_PATH}/include/mcr
    ${MACA_PATH}/include/mcrand
    ${MACA_PATH}/include/common
)

link_directories(
    ${CONDA_ENV_DIR}/lib/python3.8/site-packages/torch/lib
)

add_library(ft_attention SHARED
    "ft_attention.cpp"
    "decoder_masked_multihead_attention.cu"
    "gpu_timer.cpp"
    "ft_attention_fp16_32_32.cu"
    "ft_attention_fp16_48_64.cu"
    "ft_attention_fp16_64_64.cu"
    "ft_attention_fp16_80_128.cu"
    "ft_attention_fp16_96_128.cu"
    "ft_attention_fp16_128_128.cu"
    "ft_attention_fp16_160_256.cu"
    "ft_attention_fp16_192_256.cu"
    "ft_attention_fp16_224_256.cu"
    "ft_attention_fp16_256_256.cu"
    "ft_attention_float_32_32.cu"
    "ft_attention_float_48_64.cu"
    "ft_attention_float_64_64.cu"
    "ft_attention_float_80_128.cu"
    "ft_attention_float_96_128.cu"
    "ft_attention_float_128_128.cu"
    "ft_attention_float_160_256.cu"
    "ft_attention_float_192_256.cu"
    "ft_attention_float_224_256.cu"
    "ft_attention_float_256_256.cu"
)


target_link_libraries(ft_attention PUBLIC torch_python)

# /home/qili/miniconda3/envs/flash_attention2/lib/python3.8/site-packages/torch/lib
# /root/anaconda3/envs/qili/lib/python3.8/site-packages/torch/lib