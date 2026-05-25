cmake_minimum_required( VERSION 3.16 )

project(mcflashAttn VERSION 1.0)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

if( NOT DEFINED MACA_PATH )
    message(FATAL_ERROR "MACA_PATH is not defined.")
endif()

# Get environment variable FLASHATTN_BUILD_PROJECTS
set(ENV_FLASHATTN_BUILD_PROJECT $ENV{FLASHATTN_BUILD_PROJECTS})

# Clear cached variables for fresh start
unset(PROJECT_LIST CACHE)
unset(TARGET_ARCH_LIST CACHE)

if(DEFINED ENV{FLASHATTN_BUILD_PROJECTS})
    set(PROJECT_LIST ${ENV_FLASHATTN_BUILD_PROJECT} CACHE STRING "Value from environment variable FLASHATTN_BUILD_PROJECTS")
else()
    # Enable multi-architecture compilation by default
    set(PROJECT_LIST "C500:C600")
endif()

if(DEFINED PROJECT_LIST)
    string(REGEX REPLACE "([^:]+):" "\\1;" BUILT_PROJECT_LIST "${PROJECT_LIST}")
    string(REPLACE "\"" "" TARGET_ARCH_LIST "${BUILT_PROJECT_LIST}")
    list(LENGTH TARGET_ARCH_LIST TARGET_ARCH_LIST_LEN)
endif()

if(TARGET_ARCH_LIST_LEN EQUAL 1)
    list(GET TARGET_ARCH_LIST 0 ITEM)
    string(TOLOWER "${ITEM}" ARCH_LOWER)
    if(ARCH_LOWER STREQUAL "c600")
        add_definitions(-DXCORE1500)
        set(MACA_ARCH "xcore1500")
        message(STATUS "Set arch=xcore1500")
    else()
        # Set c500 as default single-architecture compilation target
        add_definitions(-DXCORE1000)
        set(MACA_ARCH "xcore1000")
        message(STATUS "Set arch=xcore1000")
    endif()
else()
    set(MACA_ARCH "all")
    message (STATUS "Set arch=xcore1000,xcore1500")
endif()

option(BUILD_WITH_CPP    "Build cpp library" true)
option(BUILD_WITH_KERNEL "Build kernel"      true)
option(BUILD_WITH_HOST   "Build host static library" true)
option(BUILD_WITH_BWD_KERNEL "Build backward kernels" OFF)
option(FWD_ENABLE_LOCAL "Build forward local-attention kernel variants" OFF)
option(FWD_ENABLE_ALIBI "Build forward ALiBi kernel variants" OFF)
option(FWD_ENABLE_SOFTCAP "Build forward softcap kernel variants" OFF)
option(FWD_ENABLE_APPENDKV "Build forward append-KV kernel variants" OFF)
option(FWD_ENABLE_CAUSAL "Build forward causal kernel variants" OFF)
set(FWD_MN_LIST "DEFAULT" CACHE STRING "Comma-separated xcore1000 fwd tile list, or DEFAULT for the default dispatch tiles")
set(FWD_SPLIT_MN_LIST "DEFAULT" CACHE STRING "Comma-separated xcore1000 fwd_split tile list, or DEFAULT for the default dispatch tiles")

message(STATUS "BUILD_WITH_CPP:${BUILD_WITH_CPP}")
message(STATUS "BUILD_WITH_KERNEL:${BUILD_WITH_KERNEL}")
message(STATUS "BUILD_WITH_HOST:${BUILD_WITH_HOST}")
message(STATUS "BUILD_WITH_BWD_KERNEL:${BUILD_WITH_BWD_KERNEL}")
message(STATUS "FWD_MN_LIST:${FWD_MN_LIST}")
message(STATUS "FWD_SPLIT_MN_LIST:${FWD_SPLIT_MN_LIST}")
message(STATUS "FWD_ENABLE_LOCAL:${FWD_ENABLE_LOCAL}")
message(STATUS "FWD_ENABLE_ALIBI:${FWD_ENABLE_ALIBI}")
message(STATUS "FWD_ENABLE_SOFTCAP:${FWD_ENABLE_SOFTCAP}")
message(STATUS "FWD_ENABLE_APPENDKV:${FWD_ENABLE_APPENDKV}")
message(STATUS "FWD_ENABLE_CAUSAL:${FWD_ENABLE_CAUSAL}")

# set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
# set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "../compile_time.sh")

set(CMAKE_C_COMPILER ${MACA_PATH}/tools/cu-bridge/bin/cucc)
set(CMAKE_CXX_COMPILER ${MACA_PATH}/tools/cu-bridge/bin/cucc)
set(CMAKE_PREFIX_PATH ${PREBUILT})
message(STATUS "CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")

add_compile_options(-w)
add_compile_options(-Wno-format)
add_compile_options(-x maca)
add_compile_options(--compiler-options -fPIC)
add_compile_options(-O3)
add_compile_options(-std=c++17)
add_compile_options(--expt-relaxed-constexpr)
add_compile_options(--expt-extended-lambda)
add_compile_options(--use_fast_math)
add_compile_options(-fno-strict-aliasing)
add_compile_options("SHELL:-mllvm -metaxgpu-inlinescope=50")
add_compile_options("SHELL:-mllvm -metaxgpu-disable-bsm-offset=0")
add_compile_options(-Xclang -menable-no-nans)
add_compile_options("SHELL:-gencode arch=compute_80,code=sm_80")
add_compile_options("SHELL:-gencode arch=compute_90,code=sm_90")
add_compile_options("SHELL:-mllvm -metaxgpu-enable-promote-kernel-arguments")

add_compile_options("SHELL:-mllvm -metaxgpu-const-KArgs-LDU=1")
add_compile_options("SHELL:-mllvm -metaxgpu-enable-gvn-loadclobber=false")
add_compile_options("SHELL:-mllvm -metaxgpu-enable-unorder-dispatch")
add_compile_options("SHELL:-mllvm -metaxgpu-disable-early-vector-combine=true")
add_compile_options("SHELL:-mllvm -metaxgpu-llvm19-inline-max-bb=2000")
add_compile_options("SHELL:-mllvm -metaxgpu-preisel-simplifycfg=true")

add_compile_definitions(EXPORT_LIB)
add_compile_definitions(USE_MACA)
add_compile_definitions(NV_ARCH_A100)
add_compile_definitions(__FAST_HALF_CVT__)
add_compile_definitions(__MERGE_LDS_B64)
if(NOT BUILD_WITH_BWD_KERNEL)
    add_compile_definitions(FLASHATTENTION_DISABLE_BACKWARD)
    add_compile_definitions(FLASHATTENTION_DISABLE_DROPOUT)
    add_compile_definitions(DROPOUT_FALSE)
endif()
if(NOT FWD_ENABLE_LOCAL)
    add_compile_definitions(FLASHATTENTION_DISABLE_LOCAL)
    add_compile_definitions(LOCAL_FALSE)
endif()
if(NOT FWD_ENABLE_ALIBI)
    add_compile_definitions(FLASHATTENTION_DISABLE_ALIBI)
    add_compile_definitions(ALIBI_FALSE)
endif()
if(NOT FWD_ENABLE_SOFTCAP)
    add_compile_definitions(FLASHATTENTION_DISABLE_SOFTCAP)
    add_compile_definitions(SOFTCAP_FALSE)
endif()
if(NOT FWD_ENABLE_APPENDKV)
    add_compile_definitions(FLASHATTENTION_DISABLE_APPENDKV)
    add_compile_definitions(APPENDKV_FALSE)
endif()
if(NOT FWD_ENABLE_CAUSAL)
    add_compile_definitions(FLASHATTENTION_DISABLE_CAUSAL)
    add_compile_definitions(CAUSAL_FALSE)
endif()

if(NOT DEFINED FA_TYPE)
    set(FA_TYPE "ALL")
endif()

if(NOT DEFINED HDIM)
    if(DEFINED HDIM_CONFIG_LIST AND NOT "${HDIM_CONFIG_LIST}" STREQUAL "")
        # Internal multi-hdim sentinel for source selection. Keep the user-facing
        # configure log explicit so it does not look like hdim 0 is being built.
        set(HDIM 0)
        set(HDIM_STATUS "HDIM_CONFIG_LIST=${HDIM_CONFIG_LIST}")
    else()
        set(HDIM 0)
        set(HDIM_STATUS "HDIM=ALL")
    endif()
else()
    set(HDIM_STATUS "HDIM=${HDIM}")
endif()
message(STATUS "${HDIM_STATUS},FA_TYPE=${FA_TYPE}")

STRING(TOUPPER "${FA_TYPE}" FA_TYPE_UPPER)

if (${FA_TYPE_UPPER} STREQUAL "ALL")
    add_compile_definitions(FA_DTYPE_ALL)
elseif (${FA_TYPE_UPPER} STREQUAL "FP16")
    add_compile_definitions(FA_DTYPE_FP16)
elseif (${FA_TYPE_UPPER} STREQUAL "BF16")
    add_compile_definitions(FA_DTYPE_BF16)
else()
    MESSAGE(FATAL_ERROR "FA_TYPE is ${FA_TYPE}, which is not ALL, FP16, or BF16")
endif()

if(NOT HDIM STREQUAL "0")
    add_compile_definitions(HDIM_CONFIG=${HDIM})
    add_compile_definitions(HDIM_${HDIM})
else()
    if(DEFINED HDIM_CONFIG_LIST AND NOT "${HDIM_CONFIG_LIST}" STREQUAL "")
        add_compile_definitions("HDIM_CONFIG=${HDIM_CONFIG_LIST}")
        string(REPLACE "," ";" HDIM_CONFIG_ITEMS "${HDIM_CONFIG_LIST}")
        foreach(HDIM_CONFIG_ITEM IN LISTS HDIM_CONFIG_ITEMS)
            string(STRIP "${HDIM_CONFIG_ITEM}" HDIM_CONFIG_ITEM)
            add_compile_definitions(HDIM_${HDIM_CONFIG_ITEM})
        endforeach()
    else()
        add_compile_definitions(HDIM_ALL)
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/flash_api
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/flash_dispatch
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/flash_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/flash_global
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/flash_kernel/feature
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/flash_kernel/fwd_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/flash_kernel/bwd_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/flash_kernel/kernel_traits
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/flash_run
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/common
    ${MACA_PATH}/tools/cu-bridge/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mctlass/include
)

include(./cmake/build_flash_attn.cmake)

if (BUILD_WITH_KERNEL)

    STRING(TOLOWER "${FA_TYPE}" FA_TYPE_LOWER)

    if(BUILD_WITH_HOST)
        build_flash_attn_host(mcFlashAttnHostStatic)
    endif()
    if(${MACA_ARCH} STREQUAL "xcore1000")
        build_flash_attn_kernel(mcFlashAttnKernelXcore1000Static "xcore1000" ${HDIM} ${FA_TYPE})
        target_compile_options(mcFlashAttnKernelXcore1000Static PRIVATE --offload-arch=xcore1000)
    elseif (${MACA_ARCH} STREQUAL "xcore1500")
        build_flash_attn_kernel(mcFlashAttnKernelXcore1500Static "xcore1500" ${HDIM} ${FA_TYPE})
        target_compile_options(mcFlashAttnKernelXcore1500Static PRIVATE --offload-arch=xcore1500)
    elseif (${MACA_ARCH} STREQUAL "all")
        build_flash_attn_kernel(mcFlashAttnKernelXcore1000Static "xcore1000" ${HDIM} ${FA_TYPE})
        build_flash_attn_kernel(mcFlashAttnKernelXcore1500Static "xcore1500" ${HDIM} ${FA_TYPE})
        target_compile_options(mcFlashAttnKernelXcore1000Static PRIVATE --offload-arch=xcore1000)
        target_compile_options(mcFlashAttnKernelXcore1500Static PRIVATE --offload-arch=xcore1500)
    else()
        message(FATAL_ERROR "the arch ${MACA_ARCH} is not supported")
    endif()
endif()

# build cpp so , will be closed when start from setup
if(BUILD_WITH_CPP)

    file(GLOB CAPI_SRC "csrc/capi/src/*.cpp")

    add_library(mcFlashAttn SHARED  ${CAPI_SRC})

    target_link_directories(
        mcFlashAttn PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${MACA_PATH}/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/build_kernel
    )
    target_link_libraries(mcFlashAttn
                            -Wl,--whole-archive
                            mcFlashAttnHostStatic
                            -Wl,--no-whole-archive
                            mcblas
                            yaml-cpp)
    if(${MACA_ARCH} STREQUAL "xcore1000")
        target_link_libraries(mcFlashAttn
                            -Wl,--whole-archive
                            mcFlashAttnKernelXcore1000Static
                            -Wl,--no-whole-archive)
        message(STATUS "xcore1000 static lib is linked")
    elseif (${MACA_ARCH} STREQUAL "xcore1500")
        target_link_libraries(mcFlashAttn
                            -Wl,--whole-archive
                            mcFlashAttnKernelXcore1500Static
                            -Wl,--no-whole-archive)
        message(STATUS "xcore1500 static lib is linked")
    elseif (${MACA_ARCH} STREQUAL "all")
        target_link_libraries(mcFlashAttn
                            -Wl,--whole-archive
                            mcFlashAttnKernelXcore1000Static
                            mcFlashAttnKernelXcore1500Static
                            -Wl,--no-whole-archive)
        message(STATUS "xcore1500 and xcore 1500 static lib is linked")
    endif()



    set_target_properties(mcFlashAttn PROPERTIES
                        INSTALL_RPATH "$ORIGIN;$ORIGIN/../../lib;$ORIGIN/../../lib64;$ORIGIN/../lib64"
    )

    target_link_options(mcFlashAttn PRIVATE "-T${CMAKE_SOURCE_DIR}/a.lds")
    target_link_options(mcFlashAttn PRIVATE "-nodefaultrpath")

    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi/flash_attn.h DESTINATION include/flash_attn
    )

    install(TARGETS mcFlashAttn LIBRARY DESTINATION lib)
    # set cpack using param
    set( LIBRARY_NAME mcflashattn )
    include (cmake/CPack.cmake)

endif(BUILD_WITH_CPP)
