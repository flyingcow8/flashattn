#pragma once

#include <string>
#include "cute/container/cuda_types.hpp"

#define UNPACK_GRID(bidx, bidh, bidb, type)                                             \
    if (type == 0)         {bidx =  blockIdx.x; bidh =  blockIdx.y; bidb = blockIdx.z;} \
    else if (type == 1)    {bidx =  blockIdx.x; bidh =  blockIdx.z; bidb = blockIdx.y;} \
    else if (type == 2)    {bidx =  blockIdx.y; bidh =  blockIdx.x; bidb = blockIdx.z;} \
    else if (type == 3)    {bidx =  blockIdx.y; bidh =  blockIdx.z; bidb = blockIdx.x;} \
    else if (type == 4)    {bidx =  blockIdx.z; bidh =  blockIdx.y; bidb = blockIdx.x;} \
    else if (type == 5)    {bidx =  blockIdx.z; bidh =  blockIdx.x; bidb = blockIdx.y;} \
    else                   {bidx =  blockIdx.x; bidh =  blockIdx.y; bidb = blockIdx.z;} \

int get_grid_type(std::string tag, int default_gtype);

cute::dim3 flash_bwd_compute_grid_dim(int x, int h, int b, int type);
cute::dim3 flash_fwd_compute_grid_dim(int num_m_block, int h, int b, int rowblock_parallel, int block_type);
cute::dim3 flash_fwd_splitkv_compute_grid_dim(const int& num_m_block,const int& num_splits, const int& h, const int &b, const int& block_type);
