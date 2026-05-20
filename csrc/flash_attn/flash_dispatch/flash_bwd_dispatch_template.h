#pragma once

#include <cuda.h>
#include <set>
#include <iostream>
#include "flash_parameter.h"
#include "static_switch.h"

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

namespace Xcore1500 {

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

enum class FlashBwdKernels {
    kBwdKernel_hdim128_32x64_256,
    kBwdKernel_hdim128_32x128_512,
    kBwdKernel_hdim96_32x64_256,
    kBwdKernel_hdim96_16x64_128,
};

__inline__ FlashBwdKernels flash_bwd_kernel_select_hdim128(const Flash_bwd_params &params) {
    if ((params.seqlen_k == 4096 && params.b == 2 && params.h_k == 56 && params.is_causal && !params.deterministic) ||          // deepseek
        ((params.seqlen_k == 8192 || params.seqlen_k == 4096) && params.b == 1 && params.is_causal && params.deterministic) ||  // gpt moe 567B or 70B
        (params.seqlen_k == 2623 && params.is_causal && params.deterministic) ||
        (params.seqlen_k == 8192 && params.seqlen_q == 8192) ||      // C500-28741
        (params.seqlen_k == 32768)) {
        return FlashBwdKernels::kBwdKernel_hdim128_32x128_512;
    } else {
        return FlashBwdKernels::kBwdKernel_hdim128_32x64_256;
    }
}

__inline__ FlashBwdKernels flash_bwd_kernel_select_hdim96(const Flash_bwd_params &params) {
    // when seqlen_q = 64 and seqlen_k in this set, use 16x64 can get better perf in scy case
    std::set<int> seqlen_k_set{235, 299, 363, 427, 555, 619, 683, 747, 811, 875, 939, 1195, 1323, 1643};
    if (params.seqlen_q == 64 && seqlen_k_set.count(params.seqlen_k) > 0) {
        return FlashBwdKernels::kBwdKernel_hdim96_16x64_128;
    } else {
        return FlashBwdKernels::kBwdKernel_hdim96_32x64_256;
    }
}


template<int Headdim, Arch arch>
inline void run_mha_bwd_dispatch(Flash_bwd_params &params, cudaStream_t stream);

template<>
inline void run_mha_bwd_dispatch<0, Arch::xcore1000>(Flash_bwd_params &params, cudaStream_t stream){
    std::cerr << "Error: HDIM of bwd is set to 0. Ensure that this configuration is only used in a development environment." << std::endl;
    return;
}

template<>
inline void run_mha_bwd_dispatch<0, Arch::xcore1500>(Flash_bwd_params &params, cudaStream_t stream){
    std::cerr << "Error: HDIM of bwd is set to 0. Ensure that this configuration is only used in a development environment." << std::endl;
    return;
}


#if CHECK_HDIM(32)
template<>
inline void run_mha_bwd_dispatch<32, Arch::xcore1000>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_bwd_template<Headdim, 32, 128, 4, 2, 2, 2, true, true, true, elem_type>(params, launch_params,stream);
    });
}

template<>
inline void run_mha_bwd_dispatch<32, Arch::xcore1500>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_bwd_template<Headdim, 32, 128, 4, 2, 2, 2, true, true, true, elem_type>(params, launch_params,stream);
    });
}

#endif

#if CHECK_HDIM(64)
template<>
inline void run_mha_bwd_dispatch<64, Arch::xcore1000>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_bwd_template<Headdim, 64, 64, 4, 4, 1, 4, true, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_bwd_dispatch<64, Arch::xcore1500>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_bwd_template<Headdim, 64, 64, 4, 4, 1, 4, true, true, true, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(96)
template<>
inline void run_mha_bwd_dispatch<96, Arch::xcore1000>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_bwd_template<Headdim, 32, 64, 4, 2, 2, 2, true, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_bwd_dispatch<96, Arch::xcore1500>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_bwd_template<Headdim, 32, 64, 4, 2, 2, 2, true, true, true, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(128)
template<>
inline void run_mha_bwd_dispatch<128, Arch::xcore1000>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
#ifdef __USE_128_32x32
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_bwd_template<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, elem_type>(params, launch_params, stream);
    });
#else
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_bwd_template<Headdim, 32, 64, 4, 2, 2, 2, true, false, true, elem_type>(params, launch_params, stream);
    });
#endif
}

template <>
inline void run_mha_bwd_dispatch<128, Arch::xcore1500>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        // Xcore1500::run_flash_bwd_template<Headdim, 64, 128, 8, 1, 2, 1,  true, true, false, elem_type>(
        //     params, launch_params, stream); // private mem
        Xcore1500::run_flash_bwd_template<Headdim, 32, 64, 4, 2, 2, 1, true, true, false, elem_type>(
            params, launch_params, stream);
    });
}
#endif



#if CHECK_HDIM(160)
template<>
inline void run_mha_bwd_dispatch<160, Arch::xcore1000>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    // [0619] in maca, only 32x32 tile support hdim%32 == 0 for now
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_bwd_template<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_bwd_dispatch<160, Arch::xcore1500>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_bwd_template<Headdim, 32, 32, 2, 2, 2, 2, true, true, true, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(192)
template<>
inline void run_mha_bwd_dispatch<192, Arch::xcore1000>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int HeaddimQ = 192;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        if (params.d_value_rounded == 128) { // only for hdim_q = 192 and hdim_v = 128
            constexpr static int HeaddimV = 128;
            Xcore1000::run_flash_bwd_template<HeaddimQ, 32, 64, 8, 2, 2, 2, true, true, true, elem_type, HeaddimV>(params, launch_params, stream);
        } else {
            Xcore1000::run_flash_bwd_template<HeaddimQ, 32, 32, 4, 2, 1, 2, true, true, true, elem_type>(params, launch_params, stream);
        }
    });
}

template<>
inline void run_mha_bwd_dispatch<192, Arch::xcore1500>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int HeaddimQ = 192;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        if (params.d_value_rounded == 128) { // only for hdim_q = 192 and hdim_v = 128
            constexpr static int HeaddimV = 128;
            // Xcore1500::run_flash_bwd_template<HeaddimQ, 32, 64, 8, 2, 2, 2, true, true, true, elem_type, HeaddimV>(params, launch_params, stream);//49
            Xcore1500::run_flash_bwd_template<HeaddimQ, 32, 64, 4, 1, 2, 1, true, true, false, elem_type, HeaddimV>(params, launch_params, stream);//66
            // Xcore1500::run_flash_bwd_template<HeaddimQ, 64, 64, 8, 2, 4, 2, true, true, false, elem_type, HeaddimV>(params, launch_params, stream);
        } else {
            // Xcore1500::run_flash_bwd_template<HeaddimQ, 32, 64, 8, 2, 2, 2, true, true, true, elem_type>(params, launch_params, stream);//43
            // Xcore1500::run_flash_bwd_template<HeaddimQ, 64, 64, 8, 2, 2, 2, true, true, false, elem_type>(params, launch_params, stream);//46
            Xcore1500::run_flash_bwd_template<HeaddimQ, 64, 64, 8, 2, 4, 2, true, true, false, elem_type>(params, launch_params, stream);//52
            // Xcore1500::run_flash_bwd_template<HeaddimQ, 32, 32, 4, 2, 2, 2, true, true, false, elem_type>(params, launch_params, stream);//47
        }
    });
}

#endif


#if CHECK_HDIM(256)
template<>
inline void run_mha_bwd_dispatch<256, Arch::xcore1000>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_bwd_template<Headdim, 32, 32, 8, 2, 2, 2, true, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_bwd_dispatch<256, Arch::xcore1500>(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    mcFlashAttn::Flash_launch_params launch_params;
    launch_params.performance_mode = false;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_bwd_template<Headdim, 32, 32, 4, 2, 2, 2, true, true, false, elem_type>(params, launch_params, stream);//30,pm12(C500 27)
        // Xcore1500::run_flash_bwd_template<Headdim, 32, 32, 8, 2, 2, 2, true, true, false, elem_type>(params, launch_params, stream);//
    });
}

#endif

#if CHECK_HDIM(512)
template<>
inline void run_mha_bwd_dispatch<512, Arch::xcore1000>(Flash_bwd_params &params, cudaStream_t stream) {
    std::cerr << "HDIM is set to 0. Not support bwd of headdim 512." << std::endl;
    return;
}

template<>
inline void run_mha_bwd_dispatch<512, Arch::xcore1500>(Flash_bwd_params &params, cudaStream_t stream) {
    std::cerr << "Xcore1500 currently does not support headdim 512." << std::endl;
    return;
}

#endif
