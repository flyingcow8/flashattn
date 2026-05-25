/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <iostream>
#include "flash_parameter.h"
#include "../utils/static_switch.h"
#define HDIM_ALL

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
void run_flash_fwd_template(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

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

namespace Xcore1500 {

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

namespace mcFlashAttn {

    template<int Headdim, Arch arch>
    inline void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, const cudaStream_t stream) {
        if constexpr (Headdim == 0) {
            std::cerr << "Error: HDIM of fwd is set to 0. Ensure that this configuration is only used in a development environment." << std::endl;
            return;
        }
        Flash_launch_params launch_params;

        constexpr static int kBlockM = 64;  // Fixed for all head dimensions
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            if constexpr (arch == Arch::xcore1000) {
                Xcore1000::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
            }
        });
    }

    #if CHECK_HDIM(32)
    template<>
    inline void run_mha_fwd_splitkv_dispatch<32, Arch::xcore1000>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 32;

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1000::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }

    template<>
    inline void run_mha_fwd_splitkv_dispatch<32, Arch::xcore1500>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 32;

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1500::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }
    #endif

    #if CHECK_HDIM(64)
    template<>
    inline void run_mha_fwd_splitkv_dispatch<64, Arch::xcore1000>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 64;

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1000::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }

    template<>
    inline void run_mha_fwd_splitkv_dispatch<64, Arch::xcore1500>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 64;

        constexpr static int kBlockM = 128;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1500::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, false, false, elem_type>(params, launch_params, stream);
        });
    }
    #endif

    #if CHECK_HDIM(96)
    template<>
    inline void run_mha_fwd_splitkv_dispatch<96, Arch::xcore1000>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 96;

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1000::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }

    template<>
    inline void run_mha_fwd_splitkv_dispatch<96, Arch::xcore1500>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 96;

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1500::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }
    #endif

    #if CHECK_HDIM(128)
    template<>
    inline void run_mha_fwd_splitkv_dispatch<128, Arch::xcore1000>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 128;

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1000::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }

    template<>
    inline void run_mha_fwd_splitkv_dispatch<128, Arch::xcore1500>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 128;

        if (params.seqlen_q <= 32) {
            FP16_SWITCH(!params.is_bf16, [&] {
                launch_params.block_type = 2;
                Xcore1500::run_flash_splitkv_fwd_template<Headdim, 16, 32, 1, true, true, elem_type>(params, launch_params, stream);
            });
        } else {
            FP16_SWITCH(!params.is_bf16, [&] {
                launch_params.block_type = 2;
                Xcore1500::run_flash_splitkv_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
            });
        }
    }
    #endif

    #if CHECK_HDIM(160)
    template<>
    inline void run_mha_fwd_splitkv_dispatch<160, Arch::xcore1000>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 160;

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1000::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }

    template<>
    inline void run_mha_fwd_splitkv_dispatch<160, Arch::xcore1500>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 160;

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1500::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }
    #endif

    #if CHECK_HDIM(192)
    template<>
    inline void run_mha_fwd_splitkv_dispatch<192, Arch::xcore1000>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 192;

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
                launch_params.block_type = 2;
                Xcore1000::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }

    template<>
    inline void run_mha_fwd_splitkv_dispatch<192, Arch::xcore1500>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 192;

        constexpr static int kBlockM = 128;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1500::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, false, false, elem_type>(params, launch_params, stream);
        });
    }
    #endif

    #if CHECK_HDIM(256)
    template<>
    inline void run_mha_fwd_splitkv_dispatch<256, Arch::xcore1000>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 256;

        constexpr static int kBlockM = 64;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1000::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }

    template<>
    inline void run_mha_fwd_splitkv_dispatch<256, Arch::xcore1500>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 256;

        constexpr static int kBlockM = 128;
        constexpr static int kBlockN = 64;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1500::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 4, false, false, elem_type>(params, launch_params, stream);
        });
    }
    #endif

    #if CHECK_HDIM(512)
    template<>
    inline void run_mha_fwd_splitkv_dispatch<512, Arch::xcore1000>(Flash_fwd_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int Headdim = 512;

        constexpr static int kBlockM = 32;
        constexpr static int kBlockN = 32;
        FP16_SWITCH(!params.is_bf16, [&] {
            launch_params.block_type = 2;
            Xcore1000::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, 2, true, true, elem_type>(params, launch_params, stream);
        });
    }

    template<>
    inline void run_mha_fwd_splitkv_dispatch<512, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream) {
        std::cerr << "Xcore1500 splitkv currently does not support headdim 512." << std::endl;
        return;
    }
    #endif

} // namespace mcFlashAttn end



template<int Headdim, Arch arch>
inline void run_mha_fwd_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template<>
inline void run_mha_fwd_dispatch<0, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream){
    std::cerr << "Error: HDIM of fwd is set to 0. Ensure that this configuration is only used in a development environment." << std::endl;
    return;
}

template<>
inline void run_mha_fwd_dispatch<0, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream){
    std::cerr << "Error: HDIM of fwd is set to 0. Ensure that this configuration is only used in a development environment." << std::endl;
    return;
}

#if CHECK_HDIM(32)
template<>
inline void run_mha_fwd_dispatch<32, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 32;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 128, 128, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<32, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 32;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    if (params.p_dropout < 1.0f) {
        FP16_SWITCH(!params.is_bf16, [&] {
            Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
        });
    } else {
        FP16_SWITCH(!params.is_bf16, [&] {
            Xcore1500::run_flash_fwd_template<Headdim, 128, 128, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }

}

#endif

#if CHECK_HDIM(64)
template<>
inline void run_mha_fwd_dispatch<64, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 64;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 64, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<64, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 64;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, false, false, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(96)
template<>
inline void run_mha_fwd_dispatch<96, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 96;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<96, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 96;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(128)
template<>
inline void run_mha_fwd_dispatch<128, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 64, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<128, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}
#endif

#if CHECK_HDIM(160)
template<>
inline void run_mha_fwd_dispatch<160, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 160;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 64, 32, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<160, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 160;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(192)
template<>
inline void run_mha_fwd_dispatch<192, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        if(params.d_value_rounded == 128) { // only for hdim_q = 192 and hdim_v = 128
            // available traits: [64,64,4], [64,32,4], [128,64,4], [128,64,8]
            Xcore1000::run_flash_fwd_template<Headdim, 128, 64, 8, true, true, elem_type, 128>(params, launch_params, stream);
        } else {
            Xcore1000::run_flash_fwd_template<Headdim, 64, 64, 4, true, true, elem_type>(params, launch_params, stream);
        }
    });
}

template<>
inline void run_mha_fwd_dispatch<192, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        if(params.d_value_rounded == 128) { // only for hdim_q = 192 and hdim_v = 128
            Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 8, true, true, elem_type, 128>(params, launch_params, stream);
        } else {
            Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, false, false, elem_type>(params, launch_params, stream);
        }
    });
}

#endif

#if CHECK_HDIM(256)
template<>
inline void run_mha_fwd_dispatch<256, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
#ifdef FLASHATTENTION_DISABLE_DROPOUT
        Xcore1000::run_flash_fwd_template<Headdim, 64, 32, 4, true, true, elem_type>(params, launch_params, stream);
#else
        bool is_dropout = params.p_dropout < 1.f;
        if (!is_dropout) {
            Xcore1000::run_flash_fwd_template<Headdim, 64, 32, 4, true, true, elem_type>(params, launch_params, stream);
        }
        else {
            Xcore1000::run_flash_fwd_template<Headdim, 64, 64, 4, true, true, elem_type>(params, launch_params, stream);
        }
#endif
    });
}

template<>
inline void run_mha_fwd_dispatch<256, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, false, false, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(512)
template<>
inline void run_mha_fwd_dispatch<512, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 512;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 64, 32, 8, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<512, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream) {
    std::cerr << "Xcore1500 currently does not support headdim 512." << std::endl;
    return;
}

#endif
