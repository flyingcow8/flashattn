#!/bin/bash

# update the static kernel local
# e.g. ./update_static_kernel_local.sh xcore1000

# delete origin static kernel
rm ./full_kernels/$1/* -rf
rm ./run_flash_template/$1/* -rf

# generator new static kernel
./run_generate_local.sh $1

# copy new static kernel
cp -r ../../build_kernel/full_kernels/$1/* ./full_kernels/$1/
cp -r ../../build_kernel/run_flash_template/$1/* ./run_flash_template/$1/
