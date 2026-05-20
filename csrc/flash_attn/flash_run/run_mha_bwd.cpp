#include <mctlass/numeric_types.h>
#include "run_mha.h"
#include "hdim_switch.h"
#include "static_switch.h"
#include "flash_bwd_dispatch_template.h"

void run_mha_bwd(mcFlashAttn::Flash_bwd_params &params, cudaStream_t stream) {
    HEADDIM_SWITCH(params.d, {
        ARCH_SWITCH(params.arch, kArch, [&] {
            run_mha_bwd_dispatch<kHeadDimension, kArch>(params,stream);
        });
    });
}
