cmake_minimum_required(VERSION 3.10)
project(flash_attn_inference_example)
set(CMAKE_BUILD_TYPE "Debug")

add_definitions(-w)

if(NOT DEFINED ENV{MACA_PATH})
    message(FATAL_ERROR "Please make sure the environment variable MACA_PATH is set.
    using export MACA_PATH=/your/maca/path")
endif()

if(NOT DEFINED MACA_CLANG_PATH)
    message(FATAL_ERROR "\n The MACA_CLANG_PATH is not specified.")
else()
    set(MXCC ${MACA_CLANG_PATH}/mxcc)
    set(CMAKE_C_COMPILER ${MXCC})
    set(CMAKE_CXX_COMPILER ${MXCC})
    add_compile_options(-std=c++17)
    add_compile_options(-x maca)
endif()

# add local dir first, and use it first
INCLUDE_DIRECTORIES(${FLASH_ATTN_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/mctlass/include) # mctlass

LINK_DIRECTORIES(${FLASH_ATTN_LINK_DIR})

INCLUDE_DIRECTORIES($ENV{MACA_PATH}/include/mcr)
INCLUDE_DIRECTORIES($ENV{MACA_PATH}/include)
INCLUDE_DIRECTORIES(${FLASH_ATTN_INCLUDE_DIR})
LINK_DIRECTORIES(${FLASH_ATTN_LINK_DIR})

find_library(MC_RUNTIME_LIB mcruntime HINTS $ENV{MACA_PATH}/lib)

add_executable(flash_attn_fwd_example flash_attn_fwd_cpp_example.cpp)
target_link_libraries(flash_attn_fwd_example mcFlashAttn mcblas ${MC_RUNTIME_LIB})

add_executable(flash_attn_varlen_fwd_example flash_attn_varlen_fwd_cpp_example.cpp)
target_link_libraries(flash_attn_varlen_fwd_example mcFlashAttn mcblas ${MC_RUNTIME_LIB})

add_executable(flash_attn_fwd_kvcache_example flash_attn_fwd_kvcache_cpp_example.cpp)
target_link_libraries(flash_attn_fwd_kvcache_example mcFlashAttn mcblas ${MC_RUNTIME_LIB})
