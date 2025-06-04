import os
import re
import ast
import subprocess
from pathlib import Path
from typing import Union
from packaging.version import parse, Version

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "fused_dense_lib"

def get_cuda_bare_metal_version(cuda_dir):
    # raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    # output = raw_output.split()
    # release_idx = output.index("release") + 1
    # bare_metal_version = parse(output[release_idx].split(",")[0])
    raw_output = "nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2023 NVIDIA Corporation Built on Mon_Apr__3_17:16:06_PDT_2023 Cuda compilation tools, release 12.1, V12.1.105 Build cuda_12.1.r12.1/compiler.32688072_0"
    bare_metal_version = Version("12.1")

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.2"):
            nvcc_extra_args.append("-gencode")
            nvcc_extra_args.append("arch=compute_75,code=sm_75")
            if bare_metal_version >= Version("11.6"):
                nvcc_extra_args.append("-gencode")
                nvcc_extra_args.append("arch=compute_80,code=sm_80")
            if bare_metal_version >= Version("11.8"):
                nvcc_extra_args.append("-gencode")
                nvcc_extra_args.append("arch=compute_90,code=sm_90")
            return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

def get_sha(fused_dense_lib_root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=fused_dense_lib_root)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "Unknown"

def get_package_version():
    fused_dense_lib_root = Path(__file__).parent.parent
    sha = get_sha(fused_dense_lib_root)
    with open(Path(this_dir) / ".." / ".." /  "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = str(ast.literal_eval(version_match.group(1)))
    maca_version = os.environ.get("CORE_MODULE_MACA_VERSION")
    if maca_version:
        return public_version + "+metax" + maca_version
    elif sha != "Unknown":
        return public_version + "+git" + sha[:7]
    else:
        return public_version

def get_torch_version():
    torch_version_raw = parse(torch.__version__)
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    return torch_version

setup(
    name=PACKAGE_NAME,
    version=get_package_version() + "torch" + get_torch_version() if get_torch_version() else get_package_version(),
    ext_modules=[
        CUDAExtension(
            name=PACKAGE_NAME,
            sources=['fused_dense.cpp', 'fused_dense_cuda.cu'],
            extra_compile_args={
                               'cxx': ['-O3',],
                               'nvcc': append_nvcc_threads(['-O3',])
                               }
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
})

