macro(build_flash_attn_kernel LIB_NAME  MACA_ARCH HDIM DTYPE)

    set(HDIM_FILTER "")
    if(NOT HDIM STREQUAL "0")
        set(HDIM_FILTER ${HDIM})
    endif()

    set(DTYPE_FILTER "*")

    STRING(TOLOWER "${DTYPE}" DTYPE_LOWER)
    if(NOT ${DTYPE_LOWER} STREQUAL "all")
        set(DTYPE_FILTER ${DTYPE_LOWER})
        message(STATUS "DTYPE:${DTYPE}")
    endif()

    set(SRC_FILTER "${HDIM_FILTER}*_${DTYPE_FILTER}_*")

    message(STATUS "SRC_FILTER:${SRC_FILTER}")

    set(SRC_PARENT ${CMAKE_CURRENT_SOURCE_DIR}/tools/generator)
    if(FAST_BUILD)
        set(SRC_PARENT "build_kernel")
    endif()

    file(GLOB FWD_TRAITS_SRC "${SRC_PARENT}/run_flash_template/${MACA_ARCH}/fwd/flash_fwd_hdimqk${SRC_FILTER}.cpp")
    file(GLOB FWD_SPLIT_TRAITS_SRC "${SRC_PARENT}/run_flash_template/${MACA_ARCH}/fwd_split/flash_fwd_splitkv_hdimqk${SRC_FILTER}.cpp")
    file(GLOB BWD_TRAITS_SRC "${SRC_PARENT}/run_flash_template/${MACA_ARCH}/bwd/flash_bwd_hdimqk${SRC_FILTER}.cpp")

    file(GLOB FWD_KERNEL_SRC "${SRC_PARENT}/full_kernels/${MACA_ARCH}/fwd/flash_fwd_hdimqk${SRC_FILTER}.cpp")
    file(GLOB FWD_SPLIT_KERNEL_SRC "${SRC_PARENT}/full_kernels/${MACA_ARCH}/fwd_split/flash_fwd_splitkv_hdimqk${SRC_FILTER}.cpp")
    file(GLOB BWD_KERNEL_SRC "${SRC_PARENT}/full_kernels/${MACA_ARCH}/bwd/flash_bwd_hdimqk${SRC_FILTER}.cpp")


    add_library(${LIB_NAME} STATIC
        ${FWD_KERNEL_SRC}
        ${FWD_TRAITS_SRC}
        ${BWD_KERNEL_SRC}
        ${BWD_TRAITS_SRC}
        ${FWD_SPLIT_KERNEL_SRC}
        ${FWD_SPLIT_TRAITS_SRC}
    )
    message (STATUS "lib_name ${LIB_NAME}")

endmacro()

macro(build_flash_attn_host LIB_NAME)
    set(FLASH_ATTN_SRC
        csrc/flash_attn/flash_run/flash_performance_mode.cpp
        csrc/flash_attn/flash_run/flash_launch_parameter.cpp
        csrc/flash_attn/flash_run/run_mha_fwd.cpp
        csrc/flash_attn/flash_run/run_mha_bwd.cpp
        csrc/flash_attn/utils/print_parameter.cpp
        csrc/common/process_str.cpp
        csrc/common/logger.cpp
    )
    add_library(${LIB_NAME} STATIC
        ${FLASH_ATTN_SRC}
    )

endmacro()
