// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_parameter.h"
#include "flash.h"
#include "flash_fwd_launch_template.h"
#include <mctlass/numeric_types.h>

namespace mcFlashAttn {

template<>
void run_mha_fwd_<cutlass::half_t, 96, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim96<cutlass::half_t, false>(params, stream);
}

} // namespace mcFlashAttn
