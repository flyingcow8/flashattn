// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

// bool switch macros
#define EVENMN_FALSE

#include "flash_parameter.h"
#include "flash_run_fwd_split_template_impl.h"
#include <mctlass/numeric_types.h>

template void Xcore1000::run_flash_splitkv_fwd_template<
                256,
                64,
                64,
                4,
                true,
                true,
                mctlass::half_t,
                256
            >(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params,cudaStream_t stream);

