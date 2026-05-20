CURRENT_CPU_NUM:=$(shell grep -c processor /proc/cpuinfo)
# If in maca build, restrict num_jobs to 140 to resolve x86 docker which using 384 cores leads to OOM
ifdef BUILDROOT
# In x86 docker maca building, "make -j112" is a stable parallel number, so we use 140 here to get 140*0.8=112
	CURRENT_CPU_NUM:=$(shell awk -v a=$(CURRENT_CPU_NUM) 'BEGIN {if(a>140) print 140; else print a}')
endif
MHA_NUM_JOBS:=$(shell awk -v n=$(CURRENT_CPU_NUM) 'BEGIN {print int(n * 0.8)}')
HDIM ?= 0
DTYPE ?= all
# FAST_BUILD is not currently supported.
FAST_BUILD ?= 0
GEN_KERNEL ?= 0
SUB_MODULE ?= fused_dense_lib
BUILD_PROJECTS_SCRIPT_PATH := ./tools/build_scripts/build_projects_related.sh
TORCH_EXTENSION_SCRIPT_PATH := ./tools/build_scripts/torch_extension_related.sh
run_build_projects_script_%:
	@$(BUILD_PROJECTS_SCRIPT_PATH) $* || (echo "Execution failed with code $$?")
run_torch_extension_script_%:
	@$(TORCH_EXTENSION_SCRIPT_PATH) $* || (echo "Execution failed with code $$?")

kernel:
	mkdir -p build_kernel
	cd build_kernel                      \
	&& cmake                             \
		-DMACA_PATH=${MACA_PATH}         \
		-DBUILD_WITH_KERNEL=TRUE         \
		-DBUILD_WITH_CPP=FALSE           \
		-DHDIM=${HDIM}                   \
		-DFAST_BUILD=${FAST_BUILD}       \
		-DGEN_KERNEL=$(GEN_KERNEL)       \
		-DFA_TYPE=${DTYPE}               \
		..                               \
	&& make -j$(MHA_NUM_JOBS)

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
	-DFAST_BUILD=${FAST_BUILD}        \
	-DGEN_KERNEL=$(GEN_KERNEL)        \
	-DFA_TYPE=${DTYPE}                \
	.. && make -j$(MHA_NUM_JOBS) && make install

python: run_build_projects_script_pytorch kernel
	python ./setup.py bdist_wheel ;                             \

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
	rm -rf ./build_kernel

clean_capi:
	rm -rf ./build_cpp

clean:
	rm -rf build*
