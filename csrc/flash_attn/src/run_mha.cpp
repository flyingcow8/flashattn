
#include <mctlass/numeric_types.h>
#include "run_mha.h"
#include "static_switch.h"
#include "flash.h"

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {
                    mcFlashAttn::run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    mcFlashAttn::run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal,
                        [&] { mcFlashAttn::run_mha_bwd_<elem_type, kHeadDim, Is_causal>(params, stream); });
        });
    });
}
