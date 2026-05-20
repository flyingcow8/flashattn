#pragma once

#include "flash_parameter.h"

void run_mha_bwd(mcFlashAttn::Flash_bwd_params &params, cudaStream_t stream);
void run_mha_fwd(mcFlashAttn::Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false);

