#pragma once

#include "flash_parameter.h"

void malloc_accum_by_numsplits(mcFlashAttn::Flash_fwd_params &params);
void compute_params_numsplits(mcFlashAttn::Flash_fwd_params &params, const int num_splits);
void update_params_numsplits(mcFlashAttn::Flash_fwd_params &params, const int block_nums_per_AP, const int block_n, const int block_m);
