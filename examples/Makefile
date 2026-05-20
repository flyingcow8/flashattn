# Please set env refering to README.md
#export MACA_PATH=/your/maca/path
#FLASH_ATTN_INCLUDE_DIR=/path/to/your/flash_attn.h/location
#FLASH_ATTN_LINK_DIR=/path/to/your/libmcFlashAttn.so/location
MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
LOCAL_FLASH_ATTN_DIR=${PWD}/../build_cpp/install
FLASH_ATTN_INCLUDE_DIR?=${LOCAL_FLASH_ATTN_DIR}/include
FLASH_ATTN_LINK_DIR?=${LOCAL_FLASH_ATTN_DIR}/lib

BUILD_PROJECTS_SCRIPT_PATH := ../tools/build_scripts/build_projects_related.sh

run_build_projects_script_%:
	@$(BUILD_PROJECTS_SCRIPT_PATH) $* || (echo "Execution failed with code $$?")

all: run_build_projects_script_sdk
	mkdir -p build
	cd build; \
	cmake .. \
			-DFLASH_ATTN_INCLUDE_DIR=${FLASH_ATTN_INCLUDE_DIR} \
			-DFLASH_ATTN_LINK_DIR=${FLASH_ATTN_LINK_DIR} \
			-DMACA_CLANG_PATH=${MACA_CLANG_PATH} \
			-DCMAKE_EXPORT_COMPILE_COMMANDS=1 && \
			make -j8

clean:
	rm -rf build
