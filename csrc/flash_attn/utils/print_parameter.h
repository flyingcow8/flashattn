#pragma once

#include <string>
#include <vector>
#include <sstream>
#include "flash_parameter.h"

std::stringstream process_params(mcFlashAttn::Flash_bwd_params params, const mcFlashAttn::Flash_launch_params launch_params,
                                            bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask ,const std::string& debug_flag);

void shape_print(mcFlashAttn::Flash_bwd_params params, const mcFlashAttn::Flash_launch_params launch_params,
                            bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask ,const std::string& debug_flag);

void debug_print(mcFlashAttn::Flash_bwd_params params, const mcFlashAttn::Flash_launch_params launch_params, const std::string& debug_flag, std::vector<int>* extra_flag=nullptr);
