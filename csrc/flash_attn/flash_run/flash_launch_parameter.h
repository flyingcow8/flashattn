#pragma once

#include "flash_parameter.h"
#include "flash_performance_mode.h"
#include "arch.h"

bool flash_bwd_work_balance(const mcFlashAttn::Flash_bwd_params &params,const int block_n);
bool flash_bwd96_work_balance(const mcFlashAttn::Flash_bwd_params &params, const int headdim,const int block_n);


template<Arch arch>
void flash_bwd_compute_launch_parameter(
        const mcFlashAttn::Flash_bwd_params &params,
        const int headdim,
        const int block_m,
        const int block_n,
        mcFlashAttn::Flash_launch_params &launch_params);

template<>
void flash_bwd_compute_launch_parameter<Arch::xcore1000>(
        const mcFlashAttn::Flash_bwd_params &params,
        const int headdim,
        const int block_m,
        const int block_n,
        mcFlashAttn::Flash_launch_params &launch_params);

template<>
void flash_bwd_compute_launch_parameter<Arch::xcore1500>(
        const mcFlashAttn::Flash_bwd_params &params,
        const int headdim,
        const int block_m,
        const int block_n,
        mcFlashAttn::Flash_launch_params &launch_params);



void flash_fwd_compute_launch_parameter(
        const mcFlashAttn::Flash_fwd_params &params,
        const int headdim,
        const int block_m,
        const int block_n,
        mcFlashAttn::Flash_launch_params &launch_params);

void flash_fwd_splitkv_compute_launch_parameter(
        const mcFlashAttn::Flash_fwd_params &params,
        const int headdim,
        const int block_m,
        const int block_n,
        mcFlashAttn::Flash_launch_params &launch_params);
