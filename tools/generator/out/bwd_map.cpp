
#include <mctlass/numeric_types.h>
#include "kernel_scheduler.h"
#include "flash_parameter.h"

using namespace mcFlashAttn;
namespace Xcore1000 {
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
}

#ifndef XCORE1500
std::unordered_map<std::string, BwdKernelFunctionType> mcKernelScheduler::bwd_kernel_map= {

{"bwd_hdimqk_32_hdimv_32_blockm_32_blockn_128_bfloat16_4_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<32,32,128,4,2,2,2,true,true,true,mctlass::bfloat16_t,32>},
{"bwd_hdimqk_32_hdimv_32_blockm_32_blockn_128_float16_4_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<32,32,128,4,2,2,2,true,true,true,mctlass::half_t,32>},
{"bwd_hdimqk_64_hdimv_64_blockm_64_blockn_64_bfloat16_4_4_1_4_True_True_True", Xcore1000::run_flash_bwd_template<64,64,64,4,4,1,4,true,true,true,mctlass::bfloat16_t,64>},
{"bwd_hdimqk_64_hdimv_64_blockm_16_blockn_128_bfloat16_4_1_1_1_True_True_True", Xcore1000::run_flash_bwd_template<64,16,128,4,1,1,1,true,true,true,mctlass::bfloat16_t,64>},
{"bwd_hdimqk_64_hdimv_64_blockm_64_blockn_64_float16_4_4_1_4_True_True_True", Xcore1000::run_flash_bwd_template<64,64,64,4,4,1,4,true,true,true,mctlass::half_t,64>},
{"bwd_hdimqk_64_hdimv_64_blockm_16_blockn_128_float16_4_1_1_1_True_True_True", Xcore1000::run_flash_bwd_template<64,16,128,4,1,1,1,true,true,true,mctlass::half_t,64>},
{"bwd_hdimqk_96_hdimv_96_blockm_16_blockn_64_bfloat16_2_1_1_1_True_True_True", Xcore1000::run_flash_bwd_template<96,16,64,2,1,1,1,true,true,true,mctlass::bfloat16_t,96>},
{"bwd_hdimqk_96_hdimv_96_blockm_32_blockn_64_bfloat16_4_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<96,32,64,4,2,2,2,true,true,true,mctlass::bfloat16_t,96>},
{"bwd_hdimqk_96_hdimv_96_blockm_16_blockn_64_float16_2_1_1_1_True_True_True", Xcore1000::run_flash_bwd_template<96,16,64,2,1,1,1,true,true,true,mctlass::half_t,96>},
{"bwd_hdimqk_96_hdimv_96_blockm_32_blockn_64_float16_4_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<96,32,64,4,2,2,2,true,true,true,mctlass::half_t,96>},
{"bwd_hdimqk_128_hdimv_128_blockm_32_blockn_64_bfloat16_4_2_2_2_True_False_True", Xcore1000::run_flash_bwd_template<128,32,64,4,2,2,2,true,false,true,mctlass::bfloat16_t,128>},
{"bwd_hdimqk_128_hdimv_128_blockm_32_blockn_128_bfloat16_8_2_4_2_True_True_True", Xcore1000::run_flash_bwd_template<128,32,128,8,2,4,2,true,true,true,mctlass::bfloat16_t,128>},
{"bwd_hdimqk_128_hdimv_128_blockm_32_blockn_64_float16_4_2_2_2_True_False_True", Xcore1000::run_flash_bwd_template<128,32,64,4,2,2,2,true,false,true,mctlass::half_t,128>},
{"bwd_hdimqk_128_hdimv_128_blockm_32_blockn_128_float16_8_2_4_2_True_True_True", Xcore1000::run_flash_bwd_template<128,32,128,8,2,4,2,true,true,true,mctlass::half_t,128>},
{"bwd_hdimqk_160_hdimv_160_blockm_32_blockn_32_bfloat16_2_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<160,32,32,2,2,2,2,true,true,true,mctlass::bfloat16_t,160>},
{"bwd_hdimqk_160_hdimv_160_blockm_32_blockn_32_float16_2_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<160,32,32,2,2,2,2,true,true,true,mctlass::half_t,160>},
{"bwd_hdimqk_192_hdimv_128_blockm_32_blockn_32_bfloat16_4_2_1_2_True_True_True", Xcore1000::run_flash_bwd_template<192,32,32,4,2,1,2,true,true,true,mctlass::bfloat16_t,128>},
{"bwd_hdimqk_192_hdimv_128_blockm_32_blockn_64_bfloat16_8_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<192,32,64,8,2,2,2,true,true,true,mctlass::bfloat16_t,128>},
{"bwd_hdimqk_192_hdimv_128_blockm_32_blockn_32_float16_4_2_1_2_True_True_True", Xcore1000::run_flash_bwd_template<192,32,32,4,2,1,2,true,true,true,mctlass::half_t,128>},
{"bwd_hdimqk_192_hdimv_128_blockm_32_blockn_64_float16_8_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<192,32,64,8,2,2,2,true,true,true,mctlass::half_t,128>},
{"bwd_hdimqk_192_hdimv_192_blockm_32_blockn_32_bfloat16_4_2_1_2_True_True_True", Xcore1000::run_flash_bwd_template<192,32,32,4,2,1,2,true,true,true,mctlass::bfloat16_t,192>},
{"bwd_hdimqk_192_hdimv_192_blockm_32_blockn_64_bfloat16_8_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<192,32,64,8,2,2,2,true,true,true,mctlass::bfloat16_t,192>},
{"bwd_hdimqk_192_hdimv_192_blockm_32_blockn_32_float16_4_2_1_2_True_True_True", Xcore1000::run_flash_bwd_template<192,32,32,4,2,1,2,true,true,true,mctlass::half_t,192>},
{"bwd_hdimqk_192_hdimv_192_blockm_32_blockn_64_float16_8_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<192,32,64,8,2,2,2,true,true,true,mctlass::half_t,192>},
{"bwd_hdimqk_256_hdimv_256_blockm_32_blockn_32_bfloat16_8_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<256,32,32,8,2,2,2,true,true,true,mctlass::bfloat16_t,256>},
{"bwd_hdimqk_256_hdimv_256_blockm_32_blockn_32_float16_8_2_2_2_True_True_True", Xcore1000::run_flash_bwd_template<256,32,32,8,2,2,2,true,true,true,mctlass::half_t,256>},

};
#else
std::unordered_map<std::string, BwdKernelFunctionType> mcKernelScheduler::bwd_kernel_map={};
#endif
