// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

// bool switch macros


#include "flash_parameter.h"
#include "flash_run_bwd_template_impl.h"
#include <mctlass/numeric_types.h>

template void Xcore1000::run_flash_bwd_template<
                160,
                32,
                32,
                2,
                2,
                2,
                2,
                true,
                true,
                true,
                mctlass::bfloat16_t,
                160
            >(Flash_bwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

