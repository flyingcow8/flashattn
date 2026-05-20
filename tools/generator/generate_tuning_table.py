import sys
import os
import argparse
from typing import List, Optional

# set project root to ../../flash_attn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..','flash_attn'))
sys.path.insert(0, project_root)

# import tuning.FalshTuningData package
import tuning.FlashTuningData.AttentionBenchTable as AttentionBenchTable
import tuning.FlashTuningData.AttentionProblem as AttentionProblem
import tuning.FlashTuningData.AttentionSolution as AttentionSolution
from tuning.FlashTuningData.Platform import Platform
from tuning.FlashTuningData.PyApiType import PyApiType
from tuning.FlashTuningData.CppApiType import CppApiType
from tuning.FlashTuningData.KernelType import KernelType
from tuning.FlashTuningData.DataType import DataType


import kernel_impl_template as temps

KERNEL_TYPE_MAP = {
    KernelType.Fwd: "Fwd",
    KernelType.FwdSplitkv: "FwdSplitkv",
    KernelType.Bwd: "Bwd",
}

DATA_TYPE_MAP = {
    DataType.float16: "float16",
    DataType.bfloat16: "bfloat16",
}

PLATFORM_MAP = {
    Platform.MXC500: "MXC500",
    Platform.MXC550: "MXC550",
}

PYAPI_TYPE_MAP = {
    PyApiType.FlashAttnFunc: "FlashAttnFunc",
    PyApiType.FlashAttnQKVPackedFunc: "FlashAttnQKVPackedFunc",
    PyApiType.FlashAttnKVPackedFunc: "FlashAttnKVPackedFunc",
    PyApiType.FlashAttnVarlenQKVPackedFunc: "FlashAttnVarlenQKVPackedFunc",
    PyApiType.FlashAttnVarlenKVPackedFunc: "FlashAttnVarlenKVPackedFunc",
    PyApiType.FlashAttnVarlenFunc: "FlashAttnVarlenFunc",
    PyApiType.FlashAttnWithKVCache: "FlashAttnWithKVCache",
}

CPPAPI_TYPE_MAP = {
    CppApiType.FlashFwd: "FlashFwd",
    CppApiType.FlashVarlenFwd: "FlashVarlenFwd",
    CppApiType.FlashBwd: "FlashBwd",
    CppApiType.FlashVarlenBwd: "FlashVarlenBwd",
    CppApiType.FlashFwdKvcache: "FlashFwdKvcache",
}


def dump_solution(solution: AttentionSolution.AttentionSolution) -> dict:
    return {
        "head_dim": solution.HeadDim(),
        "alg": solution.Alg(),
        "num_splits": solution.NumSplits(),
        "kernel_type": KERNEL_TYPE_MAP[solution.KernelType()],
        "kernel_id": solution.KernelId().decode('utf-8') if solution.KernelId() else None,
    }

def load_tuning_table(table: AttentionBenchTable.AttentionBenchTable) -> dict:

    tuning_table = {}
    for i in range(table.ProblemsLength()):
        problem = table.Problems(i)
        hash_code = problem.HashCode()
        solution = dump_solution(problem.Solution()) if problem.Solution() else None
        if solution is not None and hash_code is not None:
            tuning_table[hash_code] = solution

    return tuning_table

def parse_kernel_type(kernel_type):
    if kernel_type == "Fwd":
        return "mcFlashKernelType::KernelType_Fwd"
    if kernel_type == "FwdSplitkv":
        return "mcFlashKernelType::KernelType_FwdSplitkv"
    if kernel_type == "Bwd":
        return "mcFlashKernelType::KernelType_Bwd"

    print(f"Error, kernel_type={kernel_type} is not support")
    sys.exit(-1)

def generate_tuning_table_cpp(tuning_table, table_name, file_name):

    item_template = """{{{KEY}ULL,{VALUE}}},
    """

    solution_template = """{{"{KERNEL_ID}",{KERNEL_TYPE},{NUM_SPLITS},{HEAD_DIM},{ALG}}}"""

    content = ""
    for hash_code,solution in tuning_table.items():
        kernel_type = parse_kernel_type(solution["kernel_type"])
        solution_str = solution_template.format(
            KERNEL_ID = solution["kernel_id"],
            KERNEL_TYPE = kernel_type,
            NUM_SPLITS = solution["num_splits"],
            HEAD_DIM = solution["head_dim"],
            ALG = solution["alg"]
        )
        content += item_template.format(KEY = hash_code,VALUE=solution_str)

    tuning_table = temps.FLASH_TUNING_TABLE_TEMPLATE.format(
        TABLE_NAME = table_name,
        MAP_ITEM_LIST = content
    )

    cpp_map_path = f'out/{file_name}.cpp'
    with open(cpp_map_path, 'w') as cpp_file:
        cpp_file.write(tuning_table)

    return tuning_table

def generate_tuning_table(bin_path, platform):
    # Read the binary file
    with open(bin_path, 'rb') as f:
        buf = f.read()

    # Get the root table
    table = AttentionBenchTable.AttentionBenchTable.GetRootAsAttentionBenchTable(buf, 0)
    tuning_table = load_tuning_table(table)

    bin_platform = PLATFORM_MAP[table.Platform()]
    version = table.Version().decode('utf-8') if table.Version() else None
    print(f"{bin_platform},{version}")

    if bin_platform != platform:
        print(f"Warning, bin platfrom is {bin_platform}, not match the platform of {platform}")

    if platform == "MXC500":
        table_name = "default_c500_solution_table"
        file_name = "tuning_table_mxc500"
    elif platform == "MXC550":
        table_name = "default_c550_solution_table"
        file_name = "tuning_table_mxc550"
    else:
        print(f"Error,not support platform:{platform}")
        sys.exit(-1)


    generate_tuning_table_cpp(tuning_table, table_name, file_name)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate the solution file to cpp source file',
    )

    parser.add_argument("--mxc500", type=str, help="The solution file of mxc500")
    parser.add_argument("--mxc550", type=str, help="The solution file of mxc550")

    return parser.parse_args()

def main():

    arg = parse_args()

    config = {
        "MXC500": arg.mxc500,
        "MXC550": arg.mxc550
    }

    for platform,bin_path in config.items():
        generate_tuning_table(bin_path, platform)


if __name__ == "__main__":
    main()
