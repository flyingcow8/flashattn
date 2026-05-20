KERNEL_IMPL_TEMPLATE_BWD_XCORE1500 = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

// bool switch macros
{BOOL_SWITCH_MACROS}

#include "flash_parameter.h"
#include "flash_run_bwd_template_impl.h"
#include <mctlass/numeric_types.h>

template void Xcore1500::run_flash_bwd_template<
                {HEAD_DIM_QK},
                {BLOCK_M},
                {BLOCK_N},
                {K_NWARPS},
                {ATOM_LAYOUT_MSDP},
                {ATOM_LAYOUT_NDKV},
                {ATOM_LAYOUT_MDQ},
                {IS_V_IN_REGS},
                {IS_K_IN_REGS},
                {NO_DOUBLE_BUFFER},
                {DTYPE},
                {HEAD_DIM_V}
            >(Flash_bwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

"""

KERNEL_IMPL_TEMPLATE_BWD_XCORE1000 = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

// bool switch macros
{BOOL_SWITCH_MACROS}

#include "flash_parameter.h"
#include "flash_run_bwd_template_impl.h"
#include <mctlass/numeric_types.h>

template void Xcore1000::run_flash_bwd_template<
                {HEAD_DIM_QK},
                {BLOCK_M},
                {BLOCK_N},
                {K_NWARPS},
                {ATOM_LAYOUT_MSDP},
                {ATOM_LAYOUT_NDKV},
                {ATOM_LAYOUT_MDQ},
                {IS_V_IN_REGS},
                {IS_K_IN_REGS},
                {NO_DOUBLE_BUFFER},
                {DTYPE},
                {HEAD_DIM_V}
            >(Flash_bwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

"""

KERNEL_IMPL_TEMPLATE_FWD_XCORE1500 = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

// bool switch macros
{BOOL_SWITCH_MACROS}

#include "flash_parameter.h"
#include "flash_run_fwd_template_impl.h"
#include <mctlass/numeric_types.h>

template void Xcore1500::run_flash_fwd_template<
                {HEAD_DIM_QK},
                {BLOCK_M},
                {BLOCK_N},
                {K_NWARPS},
                {Is_Q_in_regs_},
                {Share_Q_K_smem_},
                {DTYPE},
                {HEAD_DIM_V}
            >(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

"""
KERNEL_IMPL_TEMPLATE_FWD_XCORE1000 = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

// bool switch macros
{BOOL_SWITCH_MACROS}

#include "flash_parameter.h"
#include "flash_run_fwd_template_impl.h"
#include <mctlass/numeric_types.h>

template void Xcore1000::run_flash_fwd_template<
                {HEAD_DIM_QK},
                {BLOCK_M},
                {BLOCK_N},
                {K_NWARPS},
                {Is_Q_in_regs_},
                {Share_Q_K_smem_},
                {DTYPE},
                {HEAD_DIM_V}
            >(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

"""
KERNEL_IMPL_TEMPLATE_FWD_SPLITKV_XCORE1000 = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

// bool switch macros
{BOOL_SWITCH_MACROS}

#include "flash_parameter.h"
#include "flash_run_fwd_split_template_impl.h"
#include <mctlass/numeric_types.h>

template void Xcore1000::run_flash_splitkv_fwd_template<
                {HEAD_DIM_QK},
                {BLOCK_M},
                {BLOCK_N},
                {K_NWARPS},
                {Is_Q_in_regs_},
                {Share_Q_K_smem_},
                {DTYPE},
                {HEAD_DIM_V}
            >(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params,cudaStream_t stream);

"""
KERNEL_IMPL_TEMPLATE_FWD_SPLITKV_XCORE1500 = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

// bool switch macros
{BOOL_SWITCH_MACROS}

#include "flash_parameter.h"
#include "flash_run_fwd_split_template_impl.h"
#include <mctlass/numeric_types.h>

template void Xcore1500::run_flash_splitkv_fwd_template<
                {HEAD_DIM_QK},
                {BLOCK_M},
                {BLOCK_N},
                {K_NWARPS},
                {Is_Q_in_regs_},
                {Share_Q_K_smem_},
                {DTYPE},
                {HEAD_DIM_V}
            >(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params,cudaStream_t stream);

"""
KERNEL_MAP_TEMPLATE_BWD = """
run_flash_bwd_template<{HEAD_DIM_QK},{BLOCK_M},{BLOCK_N},{K_NWARPS},{ATOM_LAYOUT_MSDP},{ATOM_LAYOUT_NDKV},{ATOM_LAYOUT_MDQ},{IS_V_IN_REGS},{IS_K_IN_REGS},{NO_DOUBLE_BUFFER},{DTYPE},{HEAD_DIM_V}>
"""
KERNEL_MAP_TEMPLATE_FWD = """
run_flash_fwd_template<{HEAD_DIM_QK},{BLOCK_M},{BLOCK_N},{K_NWARPS},{Is_Q_in_regs_},{Share_Q_K_smem_},{DTYPE},{HEAD_DIM_V}>
"""
KERNEL_MAP_TEMPLATE_FWD_SPLITKV = """
run_flash_splitkv_fwd_template<{HEAD_DIM_QK},{BLOCK_M},{BLOCK_N},{K_NWARPS},{Is_Q_in_regs_},{Share_Q_K_smem_},{DTYPE},{HEAD_DIM_V}>
"""
HDIM_HEADER_TEMPLATE="""
#pragma once
template<int Headdim_>
struct FlashHeaddim{
  constexpr static int Headdim = 0;
};

"""
HDIM_TEMPLATE="""
template<>
struct FlashHeaddim<{HDIM}>{{
  constexpr static int Headdim = {HDIM};
}};
"""
CPP_MAP_HEADER_FWD="""
#include <mctlass/numeric_types.h>
#include "kernel_scheduler.h"
#include "flash_parameter.h"

using namespace mcFlashAttn;
namespace Xcore1000 {{
template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_fwd_template(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);
}}\n
"""
CPP_MAP_HEADER_FWD_SPLITKV="""
#include <mctlass/numeric_types.h>
#include "kernel_scheduler.h"
#include "flash_parameter.h"

using namespace mcFlashAttn;
namespace Xcore1000 {{
template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_splitkv_fwd_template(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);
}}\n
"""
CPP_MAP_HEADER_BWD="""
#include <mctlass/numeric_types.h>
#include "kernel_scheduler.h"
#include "flash_parameter.h"

using namespace mcFlashAttn;
namespace Xcore1000 {{
template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    int AtomLayoutMSdP,
    int AtomLayoutNdKV,
    int AtomLayoutMdQ,
    bool Is_V_in_regs,
    bool Is_K_in_regs,
    bool No_double_buffer,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_bwd_template(Flash_bwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);
}}\n
"""

MAP_TEMPLATES={
    "bwd":CPP_MAP_HEADER_BWD + "#ifndef XCORE1500\n" + "std::unordered_map<std::string, BwdKernelFunctionType> mcKernelScheduler::bwd_kernel_map={LIST}\n" + "#else\n" + "std::unordered_map<std::string, BwdKernelFunctionType> mcKernelScheduler::bwd_kernel_map={{}};\n" + "#endif",
    "fwd":CPP_MAP_HEADER_FWD + "#ifndef XCORE1500\n" + "std::unordered_map<std::string, FwdKernelFunctionType> mcKernelScheduler::fwd_kernel_map={LIST}\n" + "#else\n" + "std::unordered_map<std::string, FwdKernelFunctionType> mcKernelScheduler::fwd_kernel_map={{}};\n" + "#endif",
    "fwd_split":CPP_MAP_HEADER_FWD_SPLITKV + "#ifndef XCORE1500\n" + "std::unordered_map<std::string, FwdSplitKernelFunctionType> mcKernelScheduler::fwd_split_kernel_map={LIST}\n" + "#else\n" + "std::unordered_map<std::string, FwdSplitKernelFunctionType> mcKernelScheduler::fwd_split_kernel_map={{}};\n" + "#endif",
}


FLASH_TUNING_TABLE_TEMPLATE = """
#include "flash_tuning_table.h"

std::unordered_map<std::size_t, mcFlashTuningSolution> mcFlashTuningTable::{TABLE_NAME} = {{
    {MAP_ITEM_LIST}
}};
"""
#WARNING 25-09-28: Check following when remove existing kernel id, otherwise may lead to unidentified symbol
OLD_FWD_MAP='''
{"fwd_hdimqk_32_hdimv_32_blockm_128_blockn_128_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<32,128,128,4,true,true,mctlass::bfloat16_t,32>},
{"fwd_hdimqk_32_hdimv_32_blockm_128_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<32,128,64,4,true,true,mctlass::bfloat16_t,32>},
{"fwd_hdimqk_32_hdimv_32_blockm_128_blockn_128_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<32,128,128,4,true,true,mctlass::half_t,32>},
{"fwd_hdimqk_32_hdimv_32_blockm_128_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<32,128,64,4,true,true,mctlass::half_t,32>},
{"fwd_hdimqk_64_hdimv_64_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<64,64,64,4,true,true,mctlass::bfloat16_t,64>},
{"fwd_hdimqk_64_hdimv_64_blockm_128_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<64,128,64,4,true,true,mctlass::bfloat16_t,64>},
{"fwd_hdimqk_64_hdimv_64_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<64,64,64,4,true,true,mctlass::half_t,64>},
{"fwd_hdimqk_64_hdimv_64_blockm_128_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<64,128,64,4,true,true,mctlass::half_t,64>},
{"fwd_hdimqk_96_hdimv_96_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<96,64,64,4,true,true,mctlass::bfloat16_t,96>},
{"fwd_hdimqk_96_hdimv_96_blockm_128_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<96,128,64,4,true,true,mctlass::bfloat16_t,96>},
{"fwd_hdimqk_96_hdimv_96_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<96,64,64,4,true,true,mctlass::half_t,96>},
{"fwd_hdimqk_96_hdimv_96_blockm_128_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<96,128,64,4,true,true,mctlass::half_t,96>},
{"fwd_hdimqk_128_hdimv_128_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<128,64,64,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_hdimqk_128_hdimv_128_blockm_128_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<128,128,64,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_hdimqk_128_hdimv_128_blockm_128_blockn_32_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<128,128,32,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_hdimqk_128_hdimv_128_blockm_64_blockn_32_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<128,64,32,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_hdimqk_128_hdimv_128_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<128,64,64,4,true,true,mctlass::half_t,128>},
{"fwd_hdimqk_128_hdimv_128_blockm_128_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<128,128,64,4,true,true,mctlass::half_t,128>},
{"fwd_hdimqk_128_hdimv_128_blockm_128_blockn_32_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<128,128,32,4,true,true,mctlass::half_t,128>},
{"fwd_hdimqk_128_hdimv_128_blockm_64_blockn_32_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<128,64,32,4,true,true,mctlass::half_t,128>},
{"fwd_hdimqk_160_hdimv_160_blockm_128_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<160,128,64,4,true,true,mctlass::bfloat16_t,160>},
{"fwd_hdimqk_160_hdimv_160_blockm_64_blockn_32_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<160,64,32,4,true,true,mctlass::bfloat16_t,160>},
{"fwd_hdimqk_160_hdimv_160_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<160,64,64,4,true,true,mctlass::bfloat16_t,160>},
{"fwd_hdimqk_160_hdimv_160_blockm_128_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<160,128,64,4,true,true,mctlass::half_t,160>},
{"fwd_hdimqk_160_hdimv_160_blockm_64_blockn_32_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<160,64,32,4,true,true,mctlass::half_t,160>},
{"fwd_hdimqk_160_hdimv_160_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<160,64,64,4,true,true,mctlass::half_t,160>},
{"fwd_hdimqk_192_hdimv_128_blockm_128_blockn_64_bfloat16_8_True_True_False", Xcore1000::run_flash_fwd_template<192,128,64,8,true,true,mctlass::bfloat16_t,128>},
{"fwd_hdimqk_192_hdimv_128_blockm_128_blockn_64_float16_8_True_True_False", Xcore1000::run_flash_fwd_template<192,128,64,8,true,true,mctlass::half_t,128>},
{"fwd_hdimqk_192_hdimv_192_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<192,64,64,4,true,true,mctlass::bfloat16_t,192>},
{"fwd_hdimqk_192_hdimv_192_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<192,64,64,4,true,true,mctlass::half_t,192>},
{"fwd_hdimqk_256_hdimv_256_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<256,64,64,4,true,true,mctlass::bfloat16_t,256>},
{"fwd_hdimqk_256_hdimv_256_blockm_64_blockn_32_bfloat16_4_True_True_False", Xcore1000::run_flash_fwd_template<256,64,32,4,true,true,mctlass::bfloat16_t,256>},
{"fwd_hdimqk_256_hdimv_256_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<256,64,64,4,true,true,mctlass::half_t,256>},
{"fwd_hdimqk_256_hdimv_256_blockm_64_blockn_32_float16_4_True_True_False", Xcore1000::run_flash_fwd_template<256,64,32,4,true,true,mctlass::half_t,256>},
'''

OLD_FWD_SPLIT_MAP='''
{"fwd_split_hdimqk_32_hdimv_32_blockm_64_blockn_64_bfloat16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<32,64,64,4,true,true,mctlass::bfloat16_t,32>},
{"fwd_split_hdimqk_32_hdimv_32_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<32,64,64,4,true,true,mctlass::bfloat16_t,32>},
{"fwd_split_hdimqk_32_hdimv_32_blockm_64_blockn_64_float16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<32,64,64,4,true,true,mctlass::half_t,32>},
{"fwd_split_hdimqk_32_hdimv_32_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<32,64,64,4,true,true,mctlass::half_t,32>},
{"fwd_split_hdimqk_64_hdimv_64_blockm_64_blockn_64_bfloat16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<64,64,64,4,true,true,mctlass::bfloat16_t,64>},
{"fwd_split_hdimqk_64_hdimv_64_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<64,64,64,4,true,true,mctlass::bfloat16_t,64>},
{"fwd_split_hdimqk_64_hdimv_64_blockm_64_blockn_64_float16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<64,64,64,4,true,true,mctlass::half_t,64>},
{"fwd_split_hdimqk_64_hdimv_64_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<64,64,64,4,true,true,mctlass::half_t,64>},
{"fwd_split_hdimqk_96_hdimv_96_blockm_64_blockn_64_bfloat16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<96,64,64,4,true,true,mctlass::bfloat16_t,96>},
{"fwd_split_hdimqk_96_hdimv_96_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<96,64,64,4,true,true,mctlass::bfloat16_t,96>},
{"fwd_split_hdimqk_96_hdimv_96_blockm_64_blockn_64_float16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<96,64,64,4,true,true,mctlass::half_t,96>},
{"fwd_split_hdimqk_96_hdimv_96_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<96,64,64,4,true,true,mctlass::half_t,96>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_32_bfloat16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,64,32,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_32_bfloat16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<128,64,32,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_64_bfloat16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,64,64,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<128,64,64,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_128_blockn_64_bfloat16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,128,64,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_128_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<128,128,64,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_32_blockn_32_bfloat16_2_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,32,32,2,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_32_blockn_32_bfloat16_2_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<128,32,32,2,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_16_blockn_16_bfloat16_1_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,16,16,1,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_16_blockn_16_bfloat16_1_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<128,16,16,1,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_32_float16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,64,32,4,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_32_float16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<128,64,32,4,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_64_float16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,64,64,4,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<128,64,64,4,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_128_blockn_64_float16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,128,64,4,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_128_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<128,128,64,4,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_32_blockn_32_float16_2_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,32,32,2,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_32_blockn_32_float16_2_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<128,32,32,2,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_16_blockn_16_float16_1_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,16,16,1,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_16_blockn_16_float16_1_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<128,16,16,1,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_160_hdimv_160_blockm_64_blockn_64_bfloat16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<160,64,64,4,true,true,mctlass::bfloat16_t,160>},
{"fwd_split_hdimqk_160_hdimv_160_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<160,64,64,4,true,true,mctlass::bfloat16_t,160>},
{"fwd_split_hdimqk_160_hdimv_160_blockm_64_blockn_64_float16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<160,64,64,4,true,true,mctlass::half_t,160>},
{"fwd_split_hdimqk_160_hdimv_160_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<160,64,64,4,true,true,mctlass::half_t,160>},
{"fwd_split_hdimqk_192_hdimv_192_blockm_64_blockn_64_bfloat16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<192,64,64,4,true,true,mctlass::bfloat16_t,192>},
{"fwd_split_hdimqk_192_hdimv_192_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<192,64,64,4,true,true,mctlass::bfloat16_t,192>},
{"fwd_split_hdimqk_192_hdimv_192_blockm_64_blockn_64_float16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<192,64,64,4,true,true,mctlass::half_t,192>},
{"fwd_split_hdimqk_192_hdimv_192_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<192,64,64,4,true,true,mctlass::half_t,192>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_32_bfloat16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<256,64,32,4,true,true,mctlass::bfloat16_t,256>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_32_bfloat16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<256,64,32,4,true,true,mctlass::bfloat16_t,256>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_64_bfloat16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<256,64,64,4,true,true,mctlass::bfloat16_t,256>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_64_bfloat16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<256,64,64,4,true,true,mctlass::bfloat16_t,256>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_32_float16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<256,64,32,4,true,true,mctlass::half_t,256>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_32_float16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<256,64,32,4,true,true,mctlass::half_t,256>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_64_float16_4_True_True_True", Xcore1000::run_flash_splitkv_fwd_template<256,64,64,4,true,true,mctlass::half_t,256>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_64_float16_4_True_True_False", Xcore1000::run_flash_splitkv_fwd_template<256,64,64,4,true,true,mctlass::half_t,256>},
'''
