// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

// bool switch macros


#include "flash_parameter.h"
#include "flash_run_fwd_template_impl.h"
#include <mctlass/numeric_types.h>

template void Xcore1000::run_flash_fwd_template<
                512,
                64,
                32,
                8,
                true,
                true,
                mctlass::bfloat16_t,
                512
            >(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

