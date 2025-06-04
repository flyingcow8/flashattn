
CONDA_ENV_DIR=/home/qili/miniconda3/envs/flash_attention2

all:
	mkdir -p mx_build
	cd mx_build; \
	cmake_maca .. \
			-DMACA_PATH=${MACA_PATH} \
			-DCONDA_ENV_DIR=${CONDA_ENV_DIR} \
	&& make_maca -j64 VERBOSE=1


clean:
	rm -rf mx_build

install:
	pip install -e . --verbose

uninstall:
	rm -rf ./build
	rm -rf ./ft_attention.egg-info
	rm ./ft_attention.cpython-38-x86_64-linux-gnu.so
