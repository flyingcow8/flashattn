CURRENT_CPU_NUM:=$(shell grep "cpu cores" /proc/cpuinfo | head -1 | awk '{print $$4}')
# If in maca build, restrict num_jobs to 140 to resolve x86 docker which using 384 cores leads to OOM
ifdef BUILDROOT
# In x86 docker maca building, "make -j112" is a stable parallel number, so we use 140 here to get 140*0.8=112
	CURRENT_CPU_NUM:=$(shell awk -v a=$(CURRENT_CPU_NUM) 'BEGIN {if(a>140) print 140; else print a}')
endif
MHA_NUM_JOBS:=$(shell awk -v n=$(CURRENT_CPU_NUM) 'BEGIN {print int(n*0.8)}')
# HDIM_LIST ?= 128
HDIM_LIST ?= 128 256
empty :=
space := $(empty) $(empty)
comma := ,
HDIM_CONFIG_LIST := $(subst $(space),$(comma),$(strip $(HDIM_LIST)))
DTYPE ?= BF16
BUILD_WITH_BWD_KERNEL ?= FALSE
FWD_MN_LIST ?= DEFAULT
FWD_SPLIT_MN_LIST ?= DEFAULT
FWD_ENABLE_LOCAL ?= FALSE
FWD_ENABLE_ALIBI ?= FALSE
FWD_ENABLE_SOFTCAP ?= FALSE
FWD_ENABLE_APPENDKV ?= FALSE
FWD_ENABLE_CAUSAL ?= FALSE
SUB_MODULE ?= fused_dense_lib
BUILD_PROJECTS_SCRIPT_PATH := ./tools/build_scripts/build_projects_related.sh
TORCH_EXTENSION_SCRIPT_PATH := ./tools/build_scripts/torch_extension_related.sh
run_build_projects_script_%:
	@chmod +x $(BUILD_PROJECTS_SCRIPT_PATH) && $(BUILD_PROJECTS_SCRIPT_PATH) $* || (echo "Execution failed with code $$?")
run_torch_extension_script_%:
	@chmod +x $(TORCH_EXTENSION_SCRIPT_PATH) && $(TORCH_EXTENSION_SCRIPT_PATH) $* || (echo "Execution failed with code $$?")

kernel:
	@mkdir -p build_kernel build_host
	@cd build_host && \
		cmake \
			-DMACA_PATH=${MACA_PATH} \
			-DBUILD_WITH_KERNEL=TRUE \
			-DBUILD_WITH_CPP=FALSE \
			-DBUILD_WITH_HOST=TRUE \
			-DBUILD_WITH_BWD_KERNEL=${BUILD_WITH_BWD_KERNEL} \
			-DFWD_MN_LIST=${FWD_MN_LIST} \
			-DFWD_SPLIT_MN_LIST=${FWD_SPLIT_MN_LIST} \
			-DFWD_ENABLE_LOCAL=${FWD_ENABLE_LOCAL} \
			-DFWD_ENABLE_ALIBI=${FWD_ENABLE_ALIBI} \
			-DFWD_ENABLE_SOFTCAP=${FWD_ENABLE_SOFTCAP} \
			-DFWD_ENABLE_APPENDKV=${FWD_ENABLE_APPENDKV} \
			-DFWD_ENABLE_CAUSAL=${FWD_ENABLE_CAUSAL} \
			-DHDIM_CONFIG_LIST=${HDIM_CONFIG_LIST} \
			-DFA_TYPE=${DTYPE} \
			.. && \
	make -j$(MHA_NUM_JOBS) mcFlashAttnHostStatic || exit 1; \
	mv libmcFlashAttnHostStatic.a ../build_kernel/; \
	cd .. && rm -rf build_host
	@for hd in $(HDIM_LIST); do \
		mkdir -p build_kernel_$$hd && \
		cd build_kernel_$$hd && \
		cmake \
			-DMACA_PATH=${MACA_PATH} \
			-DBUILD_WITH_KERNEL=TRUE \
			-DBUILD_WITH_CPP=FALSE \
			-DBUILD_WITH_HOST=FALSE \
			-DHDIM=$$hd \
			-DBUILD_WITH_BWD_KERNEL=${BUILD_WITH_BWD_KERNEL} \
			-DFWD_MN_LIST=${FWD_MN_LIST} \
			-DFWD_SPLIT_MN_LIST=${FWD_SPLIT_MN_LIST} \
			-DFWD_ENABLE_LOCAL=${FWD_ENABLE_LOCAL} \
			-DFWD_ENABLE_ALIBI=${FWD_ENABLE_ALIBI} \
			-DFWD_ENABLE_SOFTCAP=${FWD_ENABLE_SOFTCAP} \
			-DFWD_ENABLE_APPENDKV=${FWD_ENABLE_APPENDKV} \
			-DFWD_ENABLE_CAUSAL=${FWD_ENABLE_CAUSAL} \
			-DFA_TYPE=${DTYPE} \
			.. && \
		make -j$(MHA_NUM_JOBS) || exit 1; \
		cd .. || exit 1; \
	done
	@cd build_kernel && \
	for arch in Xcore1000 Xcore1500; do \
		if ! ls ../build_kernel_*/libmcFlashAttnKernel$${arch}Static.a 1> /dev/null 2>&1; then \
			echo "Skipping $$arch - not found"; \
			continue; \
		fi; \
		rm -rf tmp_$$arch && mkdir -p tmp_$$arch && cd tmp_$$arch && \
		for hd in $(HDIM_LIST); do \
			if [ -f ../../build_kernel_$$hd/libmcFlashAttnKernel$${arch}Static.a ]; then \
				ar x ../../build_kernel_$$hd/libmcFlashAttnKernel$${arch}Static.a && echo "Extracted from $$hd"; \
			else \
				echo "Warning: build_kernel_$$hd/libmcFlashAttnKernel$${arch}Static.a not found"; \
			fi; \
		done && \
		ls -la *.o 2>/dev/null | head -5 && \
		rm -f ../libmcFlashAttnKernel$${arch}Static.a && \
		ar -r ../libmcFlashAttnKernel$${arch}Static.a *.o && \
		cd .. && rm -rf tmp_$$arch; \
	done

cplus_api: run_build_projects_script_sdk kernel
	mkdir -p build_cpp
	cd build_cpp                      \
	&& cmake                          \
	-DMACA_PATH=${MACA_PATH}          \
	-DCPACK_GENERATOR=DEB             \
	-DBUILD_WITH_KERNEL=FALSE	      \
	-DBUILD_WITH_CPP=TRUE             \
	-DCMAKE_INSTALL_PREFIX=./install  \
	-DHDIM=${HDIM}                    \
	-DBUILD_WITH_BWD_KERNEL=${BUILD_WITH_BWD_KERNEL} \
	-DFWD_MN_LIST=${FWD_MN_LIST}      \
	-DFWD_SPLIT_MN_LIST=${FWD_SPLIT_MN_LIST} \
	-DFWD_ENABLE_LOCAL=${FWD_ENABLE_LOCAL} \
	-DFWD_ENABLE_ALIBI=${FWD_ENABLE_ALIBI} \
	-DFWD_ENABLE_SOFTCAP=${FWD_ENABLE_SOFTCAP} \
	-DFWD_ENABLE_APPENDKV=${FWD_ENABLE_APPENDKV} \
	-DFWD_ENABLE_CAUSAL=${FWD_ENABLE_CAUSAL} \
	-DFA_TYPE=${DTYPE}                \
	.. && make -j$(MHA_NUM_JOBS) && make install

python: run_build_projects_script_pytorch kernel
	BUILD_WITH_BWD_KERNEL=${BUILD_WITH_BWD_KERNEL} FWD_ENABLE_LOCAL=${FWD_ENABLE_LOCAL} FWD_ENABLE_ALIBI=${FWD_ENABLE_ALIBI} FWD_ENABLE_SOFTCAP=${FWD_ENABLE_SOFTCAP} FWD_ENABLE_APPENDKV=${FWD_ENABLE_APPENDKV} FWD_ENABLE_CAUSAL=${FWD_ENABLE_CAUSAL} python ./setup.py bdist_wheel ;                             \

mla: run_build_projects_script_pytorch
	mkdir -p dist
	cd ./csrc/flash_mla ;                                       \
	python ./setup.py bdist_wheel --dist-dir="../../dist"

other: run_build_projects_script_pytorch
	mkdir -p dist
	cd ./csrc/$(SUB_MODULE) ;                                       \
	python ./setup.py bdist_wheel --dist-dir="../../dist"

clean_mla:
	rm -rf ./csrc/flash_mla/build

clean_kernel:
	rm -rf ./build_kernel*

clean_capi:
	rm -rf ./build_cpp

clean:
	rm -rf build*
