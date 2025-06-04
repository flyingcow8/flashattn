# Copied from https://github.com/NVIDIA/apex/tree/master/csrc/megatron
# We add the case where seqlen = 4k and seqlen = 8k
import os
import subprocess
from packaging.version import parse, Version

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def get_cuda_bare_metal_version(cuda_dir):
    # raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    # output = raw_output.split()
    # release_idx = output.index("release") + 1
    # release = output[release_idx].split(".")
    # bare_metal_major = release[0]
    # bare_metal_minor = release[1][0]
    raw_output = "nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2023 NVIDIA Corporation Built on Mon_Apr__3_17:16:06_PDT_2023 Cuda compilation tools, release 12.1, V12.1.105 Build cuda_12.1.r12.1/compiler.32688072_0"
    bare_metal_major = "12"
    bare_metal_minor = "1"

    return raw_output, bare_metal_major, bare_metal_minor


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_70,code=sm_70")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")

setup(
    name='fused_softmax_lib',
    ext_modules=[
        CUDAExtension(
            name='fused_softmax_lib',
            sources=['fused_softmax.cpp', 'scaled_masked_softmax_cuda.cu', 'scaled_upper_triang_masked_softmax_cuda.cu'],
            extra_compile_args={
                               'cxx': ['-O3',],
                               'nvcc': append_nvcc_threads(['-O3', '--use_fast_math'] + cc_flag)
                               }
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
})
