import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

from typing import Optional, Union

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")


# lib_dir = Path(CUDA_HOME).parent.parent / "lib"
# libraries=["mcblas"]
lib_dir = "../../../build_cpp/install/lib"
libraries = ["mcFlashAttn"]


ext_modules = []

ext_modules.append(
    CUDAExtension(
        name="flash_attn_2_cuda",
        sources=[
            "py_export_capi.cpp",
            "py_export_capi_inference.cpp",
            "py_export_capi_training.cpp",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-w"] + generator_flag,
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-w"
                ]
                + generator_flag
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / ".."
        ],
        extra_objects = ['{}/lib{}.so'.format(lib_dir, l) for l in libraries]
    )
)

setup(
    name='test_cpp',  # 模块名称
    version="1.0.0",
    description="Flash Attention: capi-test interface",
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)