#pragma once

#include "flash_parameter.h"

using namespace mcFlashAttn;

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream);
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel = false);
