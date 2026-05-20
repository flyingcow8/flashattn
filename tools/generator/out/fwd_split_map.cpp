
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
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_splitkv_fwd_template(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);
}

#ifndef XCORE1500
std::unordered_map<std::string, FwdSplitKernelFunctionType> mcKernelScheduler::fwd_split_kernel_map= {

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

{"fwd_split_hdimqk_32_hdimv_32_blockm_64_blockn_64_bfloat16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<32,64,64,4,true,true,mctlass::bfloat16_t,32>},
{"fwd_split_hdimqk_32_hdimv_32_blockm_64_blockn_64_float16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<32,64,64,4,true,true,mctlass::half_t,32>},
{"fwd_split_hdimqk_64_hdimv_64_blockm_64_blockn_64_bfloat16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<64,64,64,4,true,true,mctlass::bfloat16_t,64>},
{"fwd_split_hdimqk_64_hdimv_64_blockm_16_blockn_16_bfloat16_1_True_True", Xcore1000::run_flash_splitkv_fwd_template<64,16,16,1,true,true,mctlass::bfloat16_t,64>},
{"fwd_split_hdimqk_64_hdimv_64_blockm_64_blockn_64_float16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<64,64,64,4,true,true,mctlass::half_t,64>},
{"fwd_split_hdimqk_64_hdimv_64_blockm_16_blockn_16_float16_1_True_True", Xcore1000::run_flash_splitkv_fwd_template<64,16,16,1,true,true,mctlass::half_t,64>},
{"fwd_split_hdimqk_96_hdimv_96_blockm_64_blockn_64_bfloat16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<96,64,64,4,true,true,mctlass::bfloat16_t,96>},
{"fwd_split_hdimqk_96_hdimv_96_blockm_64_blockn_64_float16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<96,64,64,4,true,true,mctlass::half_t,96>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_32_bfloat16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,64,32,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_64_bfloat16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,64,64,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_128_blockn_64_bfloat16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,128,64,4,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_32_blockn_32_bfloat16_2_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,32,32,2,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_16_blockn_16_bfloat16_1_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,16,16,1,true,true,mctlass::bfloat16_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_32_float16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,64,32,4,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_64_float16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,64,64,4,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_128_blockn_64_float16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,128,64,4,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_32_blockn_32_float16_2_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,32,32,2,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_128_hdimv_128_blockm_16_blockn_16_float16_1_True_True", Xcore1000::run_flash_splitkv_fwd_template<128,16,16,1,true,true,mctlass::half_t,128>},
{"fwd_split_hdimqk_160_hdimv_160_blockm_64_blockn_64_bfloat16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<160,64,64,4,true,true,mctlass::bfloat16_t,160>},
{"fwd_split_hdimqk_160_hdimv_160_blockm_64_blockn_64_float16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<160,64,64,4,true,true,mctlass::half_t,160>},
{"fwd_split_hdimqk_192_hdimv_192_blockm_64_blockn_64_bfloat16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<192,64,64,4,true,true,mctlass::bfloat16_t,192>},
{"fwd_split_hdimqk_192_hdimv_192_blockm_64_blockn_64_float16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<192,64,64,4,true,true,mctlass::half_t,192>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_32_bfloat16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<256,64,32,4,true,true,mctlass::bfloat16_t,256>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_64_bfloat16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<256,64,64,4,true,true,mctlass::bfloat16_t,256>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_32_float16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<256,64,32,4,true,true,mctlass::half_t,256>},
{"fwd_split_hdimqk_256_hdimv_256_blockm_64_blockn_64_float16_4_True_True", Xcore1000::run_flash_splitkv_fwd_template<256,64,64,4,true,true,mctlass::half_t,256>},
{"fwd_split_hdimqk_512_hdimv_512_blockm_32_blockn_32_bfloat16_2_True_True", Xcore1000::run_flash_splitkv_fwd_template<512,32,32,2,true,true,mctlass::bfloat16_t,512>},
{"fwd_split_hdimqk_512_hdimv_512_blockm_32_blockn_32_float16_2_True_True", Xcore1000::run_flash_splitkv_fwd_template<512,32,32,2,true,true,mctlass::half_t,512>},

};
#else
std::unordered_map<std::string, FwdSplitKernelFunctionType> mcKernelScheduler::fwd_split_kernel_map={};
#endif
