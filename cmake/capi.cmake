

set(HEAD_SIZE 32 64 96 128 160 192 224 256)
set(DATA_TYPE bf16 fp16)

set(FWD_HEAD_SIZE 32 64 96 128 160 192 224 256)
set(FWD_DATA_TYPE bf16 fp16)

set(BWD_HEAD_SIZE 32 64 96 128 160 192 224 256)
set(BWD_DATA_TYPE bf16 fp16)

set(FWD_SPLIT_HEAD_SIZE 32 64 96 128 160 192 224 256)
set(FWD_SPLIT_DATA_TYPE bf16 fp16)

function(find_temp_folder OUTPUT_VAR DIRECTORY_PATH)
    set(FOUND_DIRS "")
    file(GLOB ALL_ITEMS RELATIVE ${DIRECTORY_PATH} ${DIRECTORY_PATH}/*)

    foreach(item IN LISTS ALL_ITEMS)
        set(full_path ${DIRECTORY_PATH}/${item})

        if(IS_DIRECTORY ${full_path} AND item MATCHES "temp.*")
            list(APPEND FOUND_DIRS ${full_path})
        endif()
    endforeach()

    set(${OUTPUT_VAR} ${FOUND_DIRS} PARENT_SCOPE)
endfunction()

function(cp_rename_file target)
    foreach(arg ${ARGN})
        get_filename_component(FILE_ABSOLUTE_NAME ${arg} ABSOLUTE)
        get_filename_component(FILE_NAME ${arg} NAME)
        string(REGEX REPLACE "sm80.cu" "sm80.cpp" NEW_FILE_NAME ${FILE_NAME})
        configure_file(${FILE_ABSOLUTE_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${NEW_FILE_NAME} COPYONLY)
    endforeach()
endfunction()


function(capi_compile FWD_FILES BWD_FILES SPLIT_FWD_FILES)

    set(FWD)
    set(BWD)
    SET(SPLIT_FWD)

    foreach(hz ${HEAD_SIZE})
        foreach(dt ${DATA_TYPE})

            list(APPEND FWD "flash_fwd_hdim${hz}_${dt}_sm80")
            list(APPEND BWD "flash_bwd_hdim${hz}_${dt}_sm80")
            list(APPEND SPLIT_FWD "flash_fwd_split_hdim${hz}_${dt}_sm80")

        endforeach()
    endforeach()

    set(${FWD_FILES} ${FWD} PARENT_SCOPE)

endfunction()



function(get_fwd_list FWD_LIST)
    set(TEMP)

    foreach(hz ${HEAD_SIZE})
        foreach(dt ${DATA_TYPE})
            list(APPEND TEMP "flash_fwd_hdim${hz}_${dt}_sm80")
            list(APPEND TEMP "flash_fwd_hdim${hz}_${dt}_causal_sm80")
        endforeach()
    endforeach()

    set(${FWD_LIST} ${TEMP} PARENT_SCOPE)

endfunction()

function(get_bwd_list BWD_LIST)
    set(TEMP)

    foreach(hz ${HEAD_SIZE})
        foreach(dt ${DATA_TYPE})
            list(APPEND TEMP "flash_bwd_hdim${hz}_${dt}_sm80")
            list(APPEND TEMP "flash_bwd_hdim${hz}_${dt}_causal_sm80")
        endforeach()
    endforeach()

    set(${BWD_LIST} ${TEMP} PARENT_SCOPE)

endfunction()


function(get_fwd_split_list FWD_SPLIT_LIST)
    set(TEMP)

    foreach(hz ${HEAD_SIZE})
        foreach(dt ${DATA_TYPE})
            list(APPEND TEMP "flash_fwd_split_hdim${hz}_${dt}_sm80")
            list(APPEND TEMP "flash_fwd_split_hdim${hz}_${dt}_causal_sm80")
        endforeach()
    endforeach()

    set(${FWD_SPLIT_LIST} ${TEMP} PARENT_SCOPE)

endfunction()


function(get_compile_obj FLASHATTN PYTHON_OBJ_DIR FWD_COMPILE_OBJ FWD_PY_OBJ)

    message(STATUS "PY_OBJE_DIR:${PYTHON_OBJ_DIR}")
    set(PY_OBJ_DIR "${PYTHON_OBJ_DIR}/${FLASHATTN}")
    set(SRC_DIR ${CMAKE_SOURCE_DIR}/csrc/flash_attn/src/${FLASHATTN})

    # Get the obj file of python setup
    file(GLOB ALL_ITEMS RELATIVE ${PY_OBJ_DIR} ${PY_OBJ_DIR}/*)

    set(PY_OBJ_LIST)
    foreach(item IN LISTS ALL_ITEMS)
        string(REGEX REPLACE ".o" "" tmp ${item})
        list(APPEND PY_OBJ_LIST "${tmp}")
    endforeach()


    # Get file name of capi
    if("${FLASHATTN}" STREQUAL "fwd")
        get_fwd_list(FWD_OBJS)
    elseif("${FLASHATTN}" STREQUAL "bwd")
        get_bwd_list(FWD_OBJS)
    else()
        get_fwd_split_list(FWD_OBJS)
    endif()


    set(PY_TEMP_LIST)
    set(COMPILE_TEMP_LIST)
    set(COMPILE_CPP_LIST)

    # Check if the corresponding .o file exists for the file
    foreach(obj ${FWD_OBJS})
        list(FIND PY_OBJ_LIST ${obj} index)

        if( NOT ${index} EQUAL -1) # exist of python setup
            set(obj_file "${PY_OBJ_DIR}/${obj}.o")
            list(APPEND PY_TEMP_LIST ${obj_file})
        else()
            set(src_file "${SRC_DIR}/${obj}.cu")
            list(APPEND COMPILE_TEMP_LIST ${src_file})
            list(APPEND COMPILE_CPP_LIST ${CMAKE_CURRENT_BINARY_DIR}/${obj}.cpp)
        endif()
    endforeach()

    # copy *.cu file to build_cpp/*.cpp
    cp_rename_file(mcFlashAttn_cpy ${COMPILE_TEMP_LIST})
    message(STATUS "-----" ${COMPILE_TEMP_LIST})


    set(${FWD_COMPILE_OBJ} ${COMPILE_CPP_LIST} PARENT_SCOPE)
    set(${FWD_PY_OBJ} ${PY_TEMP_LIST} PARENT_SCOPE)

endfunction()


function(get_fwd_compile_obj PYTHON_OBJ_DIR FWD_COMPILE_OBJ FWD_PY_OBJ)

    get_compile_obj("fwd" ${PYTHON_OBJ_DIR} FWD_COMPILE_TEMP_OBJ FWD_PY_TEMP_OBJ)

    set(${FWD_COMPILE_OBJ} ${FWD_COMPILE_TEMP_OBJ} PARENT_SCOPE)
    set(${FWD_PY_OBJ} ${FWD_PY_TEMP_OBJ} PARENT_SCOPE)

endfunction()


function(get_bwd_compile_obj PYTHON_OBJ_DIR FWD_COMPILE_OBJ FWD_PY_OBJ)

    get_compile_obj("bwd" ${PYTHON_OBJ_DIR} FWD_COMPILE_TEMP_OBJ FWD_PY_TEMP_OBJ)

    set(${FWD_COMPILE_OBJ} ${FWD_COMPILE_TEMP_OBJ} PARENT_SCOPE)
    set(${FWD_PY_OBJ} ${FWD_PY_TEMP_OBJ} PARENT_SCOPE)

endfunction()

function(get_fwd_split_compile_obj PYTHON_OBJ_DIR FWD_COMPILE_OBJ FWD_PY_OBJ)

    get_compile_obj("fwd_split" ${PYTHON_OBJ_DIR} FWD_COMPILE_TEMP_OBJ FWD_PY_TEMP_OBJ)

    set(${FWD_COMPILE_OBJ} ${FWD_COMPILE_TEMP_OBJ} PARENT_SCOPE)
    set(${FWD_PY_OBJ} ${FWD_PY_TEMP_OBJ} PARENT_SCOPE)

endfunction()