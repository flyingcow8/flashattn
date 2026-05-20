#pragma once

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif
#include <ATen/cuda/CUDAGraphsUtils.cuh> // For at::cuda::philox::unpack

#include <mctlass/numeric_types.h>

#include "flash_parameter.h"


#define CHECK_HOST(x) TORCH_CHECK(!x.is_cuda(), #x " must be on HOST")
#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SHAPE_RET_BOOL(x, ...) (x.sizes() == torch::IntArrayRef({__VA_ARGS__}))

void set_params_fprop(mcFlashAttn::Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap=0.0f,
                      bool seqlenq_ngroups_swapped=false,
                      bool unpadded_lse=false,
                      int d_v=0,
                      int d_v_rounded=0);

void set_params_dgrad(mcFlashAttn::Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool deterministic,
                      const bool unpadded_lse=false,
                      const float softcap=0.0f,
                      int d_v=0,
                      int d_v_rounded=0);

void set_params_alibi(mcFlashAttn::Flash_fwd_params &params, c10::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads);
void set_params_rng_state(mcFlashAttn::Flash_fwd_params &params, at::Tensor &rng_state);
void get_philox_state(c10::optional<at::Generator> &gen_, at::Tensor &rng_state, int64_t counter_offset);

// score ->[bs,      head_num,      q,      k  ]
// mask -> [bs_mask, head_num_mask, q_mask, k_mask]
// Mask shape should satisfy these rules
// 1. bs % bs_mask == 0
// 2. head_num % head_num_mask == 0
// 3. q_mask == 1 or q_mask == q
// 4. k_mask == 1 or k_mask == k or k_mask == (k + 3) / 4 * 4 (align k to multiples of 4)
std::vector<int64_t>
get_attn_mask_stride(std::vector<int64_t> &mask_shape, std::vector<int64_t> &score_shape);

void set_params_attn_mask(mcFlashAttn::Flash_fwd_params &params, bool is_causal, at::Tensor &attn_mask, int batch_size, int num_heads, int seqlen_q, int seqlen_k);
