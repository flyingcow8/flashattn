cmake_minimum_required( VERSION 3.16 )

project(mcflashAttn VERSION 1.0)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

set(CMAKE_C_COMPILER ${MACA_PATH}/tools/cu-bridge/bin/cucc)
set(CMAKE_CXX_COMPILER ${MACA_PATH}/tools/cu-bridge/bin/cucc)

include(./cmake/capi.cmake)


add_compile_options(-x maca)
add_compile_options(--compiler-options -fPIC)
add_compile_options(-O3)
add_compile_options(-std=c++17)
add_compile_options(--expt-relaxed-constexpr)
add_compile_options(--expt-extended-lambda)
add_compile_options(--use_fast_math)
add_compile_options("SHELL:-mllvm -metaxgpu-inlinescope=50")
add_compile_options("SHELL:-mllvm -metaxgpu-disable-bsm-offset=0")
add_compile_options(-Xclang -menable-no-nans)
add_compile_options("SHELL:-gencode arch=compute_80,code=sm_80")
add_compile_options("SHELL:-gencode arch=compute_90,code=sm_90")
add_compile_options("SHELL:-mllvm -metaxgpu-enable-promote-kernel-arguments")
add_compile_options("SHELL:-mllvm -metaxgpu-const-KArgs-LDU=1")
add_compile_options("SHELL:-mllvm -metaxgpu-enable-unorder-dispatch")

add_compile_definitions(EXPORT_LIB)
add_compile_definitions(USE_MACA)
add_compile_definitions(NV_ARCH_A100)
add_compile_definitions(__FAST_HALF_CVT__)
add_compile_definitions(__MERGE_LDS_B64)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src
    ${MACA_PATH}/tools/cu-bridge/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mctlass/include
)


# Set the folder of object file
set(PYTHON_SETUP_BUILD ${CMAKE_SOURCE_DIR}/build)
find_temp_folder(BUID_TEMP_DIR ${PYTHON_SETUP_BUILD})
set(OBJ_FILES_DIR ${BUID_TEMP_DIR}/csrc/flash_attn/src/)
message(STATUS ${OBJ_FILES_DIR})

# message(STATUS ${BWD_OBJECT_FILES})
get_fwd_compile_obj(${OBJ_FILES_DIR} FWD_COMPILE_OBJ FWD_PY_OBJ)
get_fwd_split_compile_obj(${OBJ_FILES_DIR} FWD_SPLIT_COMPILE_OBJ FWD_SPLIT_PY_OBJ)
get_bwd_compile_obj(${OBJ_FILES_DIR} BWD_COMPILE_OBJ BWD_PY_OBJ)

# message(STATUS "FWD_COMPILE_OBJ:${FWD_COMPILE_OBJ}")
# message(STATUS "FWD_PY_OBJ:${FWD_PY_OBJ}")
# message(STATUS "FWD_SPLIT_COMPILE_OBJ:${FWD_SPLIT_COMPILE_OBJ}")
# message(STATUS "FWD_SPLIT_PY_OBJ:${FWD_SPLIT_PY_OBJ}")
# message(STATUS "BWD_COMPILE_OBJ:${BWD_COMPILE_OBJ}")
# message(STATUS "BWD_PY_OBJ:${BWD_PY_OBJ}")

set(OBJECT_FILES ${FWD_PY_OBJ} ${FWD_SPLIT_PY_OBJ} ${BWD_PY_OBJ})
set(COMPILE_SRC_FILES ${FWD_COMPILE_OBJ} ${FWD_SPLIT_COMPILE_OBJ} ${BWD_COMPILE_OBJ})

set(LIBRARY_NAME flash_attn2_static)

add_library(${LIBRARY_NAME} STATIC csrc/capi/src/empty.cpp)
set_target_properties(${LIBRARY_NAME} PROPERTIES
                      OUTPUT_NAME ${LIBRARY_NAME})


set(STATIC_LIB lib${LIBRARY_NAME})

add_custom_target(build_static_lib ALL
    COMMAND ${CMAKE_AR} -rc ${CMAKE_CURRENT_BINARY_DIR}/${STATIC_LIB}.a ${OBJECT_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

include_directories(
    csrc/capi
)

set(capisrc
    csrc/capi/src/flash_capi.cpp
    csrc/capi/src/tensor.cpp
    csrc/capi/src/utils.cpp
    csrc/capi/src/helper.cpp
    csrc/flash_attn/src/run_mha.cpp
)

add_library(mcFlashAttn SHARED ${capisrc} ${COMPILE_SRC_FILES})

add_dependencies(mcFlashAttn build_static_lib)

link_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${MACA_PATH}/lib
)

target_link_libraries(mcFlashAttn
                      -Wl,--whole-archive
                      ${LIBRARY_NAME}
                      -Wl,--no-whole-archive
                      mcblas)

set_target_properties(mcFlashAttn PROPERTIES
                    INSTALL_RPATH "$ORIGIN;$ORIGIN/../../lib;$ORIGIN/../../lib64;$ORIGIN/../lib64"
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/capi/flash_attn.h DESTINATION include/flash_attn
)



install(TARGETS mcFlashAttn LIBRARY DESTINATION lib)
# set cpack using param
set( LIBRARY_NAME mcflashattn )
if("${PACKAGE_LOCATION}" STREQUAL "package")
    include (cmake/CPack_1.cmake)
else()
    include (cmake/CPack.cmake)
endif()
