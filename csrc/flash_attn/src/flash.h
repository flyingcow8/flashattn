#pragma once

#include "flash_parameter.h"

namespace mcFlashAttn {

template <typename T, int Headdim, bool Is_causal>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template <typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template <typename T, int Headdim, bool Is_causal>
void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);

}  // namespace mcFlashAttn
