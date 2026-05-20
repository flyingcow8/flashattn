#!/bin/bash
# generate the static kernel local
# e.g. ./run_generate_local.sh xcore1000
python generate_kernel_traits.py --arch $1
python generate_kernels.py --arch $1 -u
python generate_fullkernels.py --arch $1
