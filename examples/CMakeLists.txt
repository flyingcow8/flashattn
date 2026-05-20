cmake_minimum_required(VERSION 3.10)
project(flash_attn_capi_example)
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

# Get environment variable FLASHATTN_BUILD_PROJECTS
set(ENV_FLASHATTN_BUILD_PROJECT $ENV{FLASHATTN_BUILD_PROJECTS})

# Clear cached variables for fresh start
unset(PROJECT_LIST CACHE)
unset(TARGET_ARCH_LIST CACHE)

# Set ARCH_MAP
set(ARCH_MAP
      "c500:xcore1000"
      "c600:xcore1500"
)

if(DEFINED ENV{FLASHATTN_BUILD_PROJECTS})
    set(PROJECT_LIST ${ENV_FLASHATTN_BUILD_PROJECT} CACHE STRING "Value from environment variable FLASHATTN_BUILD_PROJECTS")
else()
    # Enable multi-architecture compilation by default
    set(PROJECT_LIST "C500:C600")
endif()

if(DEFINED PROJECT_LIST)
    string(REGEX REPLACE "([^:]+):" "\\1;" BUILT_PROJECT_LIST "${PROJECT_LIST}")
    string(REPLACE "\"" "" TARGET_ARCH_LIST "${BUILT_PROJECT_LIST}")
endif()

foreach(ITEM IN LISTS TARGET_ARCH_LIST)
    string(TOLOWER "${ITEM}" ARCH_LOWER)
    set(TARGET_ARCH "")
    foreach(PAIR ${ARCH_MAP})
        string(REPLACE ":" ";" PAIR_LIST ${PAIR})
        list(GET PAIR_LIST 0 CURRENT_KEY)
        if(${CURRENT_KEY} STREQUAL ${ARCH_LOWER})
            list(GET PAIR_LIST 1 TARGET_ARCH)
            break()
        endif()
    endforeach()
    if(TARGET_ARCH STREQUAL "")
        set(TARGET_ARCH "xcore1000")
    endif()
    message("${ARCH_LOWER} target arch is ${TARGET_ARCH}")
    add_compile_options(--offload-arch=${TARGET_ARCH})
endforeach()

# add local dir first, and use it first
INCLUDE_DIRECTORIES(${FLASH_ATTN_INCLUDE_DIR})
LINK_DIRECTORIES(${FLASH_ATTN_LINK_DIR})

# default path in maca, use when FLASH_ATTN_*_DIR not exit
INCLUDE_DIRECTORIES($ENV{MACA_PATH}/include/mcr)
INCLUDE_DIRECTORIES($ENV{MACA_PATH}/include)
LINK_DIRECTORIES($ENV{MACA_PATH}/lib)

find_library(MC_RUNTIME_LIB mcruntime HINTS $ENV{MACA_PATH}/lib)
message(STATUS "MC_RUNTIME_LIB: ${MC_RUNTIME_LIB}")

add_executable(flash_attn_fwd_example flash_attn_fwd_cpp_example.cpp)
target_link_libraries(flash_attn_fwd_example mcFlashAttn ${MC_RUNTIME_LIB})

add_executable(flash_attn_varlen_fwd_example flash_attn_varlen_fwd_cpp_example.cpp)
target_link_libraries(flash_attn_varlen_fwd_example mcFlashAttn ${MC_RUNTIME_LIB})

add_executable(flash_attn_fwd_kvcache_example flash_attn_fwd_kvcache_cpp_example.cpp)
target_link_libraries(flash_attn_fwd_kvcache_example mcFlashAttn ${MC_RUNTIME_LIB})


add_executable(flash_attn_bwd_example flash_attn_bwd_cpp_example.cpp)
target_link_libraries(flash_attn_bwd_example mcFlashAttn ${MC_RUNTIME_LIB})

message(STATUS "TEST INATSLL PATH: ${MACA_PATH}/restricted/Internal_Test/acl/bin/flash_attn/cpp_example" )
install( TARGETS
    flash_attn_fwd_example
    flash_attn_varlen_fwd_example
    flash_attn_fwd_kvcache_example
    flash_attn_bwd_example
    DESTINATION ${MACA_PATH}/restricted/Internal_Test/acl/bin/flash_attn/cpp_example )
