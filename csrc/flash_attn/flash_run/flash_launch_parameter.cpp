#include "flash_launch_parameter.h"

bool flash_bwd_work_balance(const mcFlashAttn::Flash_bwd_params &params,const int block_n) {
    if (!params.is_causal || params.deterministic || params.d != 128) return false;

    if (params.b >= 60) return false;

    const int num_n_block = (params.seqlen_k + block_n - 1) / block_n;
    return !(num_n_block % 2) ? true : false;
}

bool flash_bwd96_work_balance(const mcFlashAttn::Flash_bwd_params &params, const int headdim,const int block_n) {
    if (!params.is_causal || params.deterministic || headdim != 96) return false;

    if(params.seqlen_k < 2048) return false;

    const int num_n_block = (params.seqlen_k + block_n - 1) / block_n;
    return !(num_n_block % 2) ? true : false;
}


template<>
void flash_bwd_compute_launch_parameter<Arch::xcore1000>(
        const mcFlashAttn::Flash_bwd_params &params,
        const int headdim,
        const int block_m,
        const int block_n,
        mcFlashAttn::Flash_launch_params &launch_params){

    bool is_balance = flash_bwd96_work_balance(params, headdim, block_n);

    if (params.seqlen_q == 8192 && params.seqlen_k == 8192 && params.d == 128) {
        is_balance = true;
    }

    int gridtype_ = 0;
    if (params.deterministic) {
        // auto dprops = flash::mcGetCurrentDeviceProperties();
        // gridDimx = (dprops.multiProcessorCount + params.b * params.h - 1) / (params.b * params.h);
        gridtype_ = 0;
    }else if(headdim == 128 || headdim == 64){
        gridtype_ = 2;
    }else if(headdim == 32){
        gridtype_ = 5;
    }

    // Over-write grid type by env-variable
    int gridtype = get_grid_type("MHA_DEBUG_BWD_GTYPE", gridtype_);

    launch_params.is_balance = is_balance;
    launch_params.block_type = gridtype;
}

template<>
void flash_bwd_compute_launch_parameter<Arch::xcore1500>(
        const mcFlashAttn::Flash_bwd_params &params,
        const int headdim,
        const int block_m,
        const int block_n,
        mcFlashAttn::Flash_launch_params &launch_params){

    bool is_balance = flash_bwd96_work_balance(params, headdim, block_n);

    int gridtype_ = 0;
    if (params.deterministic) {
        // auto dprops = flash::mcGetCurrentDeviceProperties();
        // gridDimx = (dprops.multiProcessorCount + params.b * params.h - 1) / (params.b * params.h);
        gridtype_ = 0;
    }else if(headdim == 64){
        gridtype_ = 2;
    }else if(headdim == 32){
        gridtype_ = 5;
    }

    // Over-write grid type by env-variable
    int gridtype = get_grid_type("MHA_DEBUG_BWD_GTYPE", gridtype_);

    launch_params.is_balance = is_balance;
    launch_params.block_type = gridtype;
}


void flash_fwd_compute_launch_parameter(
        const mcFlashAttn::Flash_fwd_params &params,
        const int headdim,
        const int block_m,
        const int block_n,
        mcFlashAttn::Flash_launch_params &launch_params){

    // Over-write grid type by env-variable
    int block_type = get_grid_type("MHA_DEBUG_FWD_GTYPE", launch_params.block_type);
    launch_params.block_type = block_type;
}

void flash_fwd_splitkv_compute_launch_parameter(
        const mcFlashAttn::Flash_fwd_params &params,
        const int headdim,
        const int block_m,
        const int block_n,
        mcFlashAttn::Flash_launch_params &launch_params){

    // Over-write grid type by env-variable
    int block_type = get_grid_type("MHA_DEBUG_FWD_GTYPE", launch_params.block_type);
    launch_params.block_type = block_type;
}
