#include <mctlass/numeric_types.h>
#include "run_mha.h"
#include "hdim_switch.h"
#include "static_switch.h"
#ifdef FLASHATTENTION_DISABLE_BACKWARD
#include <stdexcept>
#else
#include "flash_bwd_dispatch_template.h"
#endif

void run_mha_bwd(mcFlashAttn::Flash_bwd_params &params, cudaStream_t stream) {
#ifdef FLASHATTENTION_DISABLE_BACKWARD
    throw std::runtime_error("This flash attention build does not support backward.");
#else
    HEADDIM_SWITCH(params.d, {
        ARCH_SWITCH(params.arch, kArch, [&] {
            run_mha_bwd_dispatch<kHeadDimension, kArch>(params,stream);
        });
    });
#endif
}
