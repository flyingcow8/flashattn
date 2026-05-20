/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <mcr/mc_runtime_api.h>

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <mctlass/array.h>
#include <mctlass/mctlass.h>
#include <mctlass/numeric_conversion.h>
#include <mctlass/numeric_types.h>

#include "block_info.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;

__forceinline__ __device__ dim3 get_bidInfo(const int& blockType) {

    int m_block = blockIdx.y;
    int bidb = blockIdx.z;
    int bidh = blockIdx.x;

    if (blockType == 0) {
        int m_block = blockIdx.x;
        int bidb = blockIdx.z;
        int bidh = blockIdx.y;
        return dim3(m_block, bidb, bidh);
    }
    if (blockType == 1) {
        int m_block = blockIdx.x;
        int bidb = blockIdx.y;
        int bidh = blockIdx.z;
        return dim3(m_block, bidb, bidh);
    }

    if (blockType == 2) {
        int m_block = blockIdx.y;
        int bidb = blockIdx.z;
        int bidh = blockIdx.x;
        return dim3(m_block, bidb, bidh);
    }

    if (blockType == 3) {
        int m_block = blockIdx.y;
        int bidb = blockIdx.x;
        int bidh = blockIdx.z;
        return dim3(m_block, bidb, bidh);
    }

    if (blockType == 4) {
        int m_block = blockIdx.z;
        int bidb = blockIdx.x;
        int bidh = blockIdx.y;
        return dim3(m_block, bidb, bidh);
    }

    if (blockType == 5) {
        int m_block = blockIdx.z;
        int bidb = blockIdx.y;
        int bidh = blockIdx.x;
        return dim3(m_block, bidb, bidh);
    }

    return dim3(0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Split>
__forceinline__ __device__ dim3 get_bidInfo(const int& blockType, const int& h, int& n_split_idx) {

    if constexpr(Split) {
        int m_block = blockIdx.y;
        int bidb = blockIdx.x / h;
        int bidh = blockIdx.x - bidb * h;
        n_split_idx = blockIdx.z;
        if (blockType == 0) {
            m_block = blockIdx.z;
            n_split_idx = blockIdx.x;
            bidb = blockIdx.y / h;
            bidh = blockIdx.y - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        if (blockType == 1) {
            m_block = blockIdx.y;
            n_split_idx = blockIdx.x;
            bidb = blockIdx.z / h;
            bidh = blockIdx.z - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        if (blockType == 2) {
            m_block = blockIdx.z;
            n_split_idx = blockIdx.y;
            bidb = blockIdx.x / h;
            bidh = blockIdx.x - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        if (blockType == 3) {
            m_block = blockIdx.x;
            n_split_idx = blockIdx.y;
            bidb = blockIdx.z / h;
            bidh = blockIdx.z - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        if (blockType == 4) {
            m_block = blockIdx.y;
            n_split_idx = blockIdx.z;
            bidb = blockIdx.x / h;
            bidh = blockIdx.x - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        if (blockType == 5) {
            m_block = blockIdx.x;
            n_split_idx = blockIdx.z;
            bidb = blockIdx.y / h;
            bidh = blockIdx.y - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        return dim3(m_block, bidb, bidh);
    }
    n_split_idx = 0;
    return get_bidInfo(blockType);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template<>
__forceinline__ __device__ uint32_t relu2<mctlass::half_t>(const uint32_t x) {
    uint32_t res;
#if defined(__MACA_ARCH__)
    //asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    auto y = *reinterpret_cast<__half2 const *>(&x);
    __half zero = __half(0);
    y.x = y.x > zero ? y.x : zero;
    y.y = y.y > zero ? y.y : zero;
    res = *reinterpret_cast<uint32_t *>(&y);
#else
    const uint32_t zero = 0u;
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n"
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__MACA_ARCH__)
template<>
__forceinline__ __device__ uint32_t relu2<mctlass::bfloat16_t>(const uint32_t x) {
    //uint32_t res;
    //const uint32_t zero = 0u;
    // asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    //res = x > zero ? x : zero;
    auto y = *reinterpret_cast<__maca_bfloat162 const *>(&x);
    __maca_bfloat16 zero = __maca_bfloat16(0);
    y.x = y.x > zero ? y.x : zero;
    y.y = y.y > zero ? y.y : zero;
    uint32_t res = *reinterpret_cast<uint32_t *>(&y);
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

//#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#if 0

template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template<>
__forceinline__ __device__ uint32_t convert_relu2<mctlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
__forceinline__ __device__ uint32_t convert_relu2<mctlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 64 || THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint64_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// reduce val(tidx) val(tidx+16) val(tidx+32) val(tidx+48)
struct Partialreduce {
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        #if 0
        constexpr int OFFSET = 32;
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET));
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET / 2));
        return x;
        #endif

        /**********************************************
         ** Using one addtional __shfl_xor_sync can
         ** reduce the time of waiting arrive inst
        **********************************************/
        auto x1 = __shfl_xor_sync(uint64_t(-1), x, 48);
        auto x2 = __shfl_xor_sync(uint64_t(-1), x, 32);
        auto x3 = __shfl_xor_sync(uint64_t(-1), x, 16);
        return op(op(op(x, x1), x2), x3);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    if constexpr (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if constexpr (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if constexpr (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if constexpr (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// A and B in regs
template<typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, TiledMma tiled_mma) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////


template<bool A_in_regs=false, bool B_in_regs=false, int prefetch_lds_num=1, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_prefetch_lds(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    static_assert(decltype(size<2>(tCrA))::value >= prefetch_lds_num);

    #pragma unroll
    for (int i = 0; i < prefetch_lds_num + 1; ++i) {
        if constexpr (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i), tCrA_copy_view(_, _, i)); }
        if constexpr (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i), tCrB_copy_view(_, _, i)); }
    }
    // do first gemm outside the loop, for compiler obey the sequence
    cute::gemm(tiled_mma, tCrA(_, _, _0{}), tCrB(_, _, _0{}), acc);

    #pragma unroll
    for (int i = 1; i < size<2>(tCrA); ++i) {
        if (i + prefetch_lds_num < size<2>(tCrA)) {
            if constexpr (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + prefetch_lds_num), tCrA_copy_view(_, _, i + prefetch_lds_num)); }
            if constexpr (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + prefetch_lds_num), tCrB_copy_view(_, _, i + prefetch_lds_num)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int k = 0; k < size<2>(tCrB); ++k) {
        #pragma unroll
        for (int n = 0; n < size<1>(tCrB); n++) {
            #pragma unroll
            for (int m = 0; m < size<1>(tCrA); m++) {
                cute::gemm(tiled_mma, tCrA(_, m, k), tCrB(_, n, k), acc(_, m, n));
            }
            if (k < size<2>(tCrB) - 1) {
                cute::copy(smem_tiled_copy_B, tCsB(_, n, k + 1), tCrB_copy_view(_, n, k + 1));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename Tensor4, typename Tensor5, typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               Tensor4 &tCrB_tail, Tensor5 const& tCsB_tail, TiledMma tiled_mma,
                               TiledCopy smem_tiled_copy_B, ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                      // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) + size<1>(tCrB_tail) == size<2>(acc)); // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB_tail));                // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    Tensor tCrB_tail_copy_view = smem_thr_copy_B.retile_D(tCrB_tail);
    CUTE_STATIC_ASSERT_V(size<1>(tCrB_tail) == size<1>(tCrB_tail_copy_view));            // N

    cute::copy(smem_tiled_copy_B, tCsB_tail(_, _, _0{}), tCrB_tail_copy_view(_, _, _0{}));
    #pragma unroll
    for (int k = 0; k < size<2>(tCrB); ++k) {
        #pragma unroll
        for (int n = 0; n < size<1>(tCrB); n++) {
            #pragma unroll
            for (int m = 0; m < size<1>(tCrA); m++) {
                cute::gemm(tiled_mma, tCrA(_, m, k), tCrB(_, n, k), acc(_, m, n));
            }
            if (k < size<2>(tCrB) - 1) {
                cute::copy(smem_tiled_copy_B, tCsB(_, n, k + 1), tCrB_copy_view(_, n, k + 1));
            }
        }

        // tail
        #pragma unroll
        for (int n = 0; n < size<1>(tCrB_tail); n++) {
            #pragma unroll
            for (int m = 0; m < size<1>(tCrA); m++) {
                cute::gemm(tiled_mma, tCrA(_, m, k), tCrB_tail(_, n, k), acc(_, m, n + size<1>(tCrB)));
            }
            if (k < size<2>(tCrB_tail) - 1) {
                cute::copy(smem_tiled_copy_B, tCsB_tail(_, n, k + 1), tCrB_tail_copy_view(_, n, k + 1));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// cu: Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
// mc: Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(1, MMA_M), ncol=(4, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    //auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    //return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    return make_layout(make_layout(cute::Layout<_1>{}, get<1>(acc_layout)), make_layout(get<0>(acc_layout), get<2>(acc_layout)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    mctlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const mctlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

#define CONVERT_TENSOR_TYPE(type_s, type_d, tensor_s, tensor_d)                                                                         \
    constexpr int tensor_d##_numel = decltype(size(tensor_s))::value;                                                                   \
    mctlass::NumericArrayConverter<type_d, type_s, tensor_d##_numel > tensor_d##_convert_op;                                            \
    auto tensor_d##_frag = tensor_d##_convert_op(*reinterpret_cast<const mctlass::Array<type_s, tensor_d##_numel> *>(tensor_s.data())); \
    Tensor tensor_d = make_tensor(make_rmem_ptr<type_d>(&tensor_d##_frag), tensor_s.layout());

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_uint32 = recast<uint32_t>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, mctlass::half_t> || std::is_same_v<To_type, mctlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if 0
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
    #pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
    //Tensor out = flash::convert_type<To_type>(tensor);
    CONVERT_TENSOR_TYPE(From_type, To_type, tensor, out)
    flash::relu_(out);
#endif
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int M = 0>
__forceinline__ __device__ void cp_async_wait() {
    __builtin_mxc_arrive_gvmcnt(M);
}

// barrier_ex(2) == barrier_inst()
template <int N = 2>
__forceinline__ __device__ void sync_threads() {
    __builtin_mxc_arrive_bsmcnt(0);
    __builtin_mxc_barrier_ex(N);
}

template <int N = 2>
__forceinline__ __device__ void barrier() {
    __builtin_mxc_barrier_ex(N);
}

template <int M = 0, int N = 2>
__forceinline__ __device__ void barrier_gvm() {
    __builtin_mxc_arrive_gvmcnt(M);
    __builtin_mxc_barrier_ex(N);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
    // TD [2023-04-13]: Strange that the code below can cause race condition.
    // I think it's because the copies are under an if statement.
    // if (Is_even_K) {
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(S); ++m) {
    //         if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //             copy(tiled_copy, S(_, m, _), D(_, m, _));
    //         } else if (Clear_OOB_MN) {
    //             clear(D(_, m, _));
    //         }
    //     }
    // } else {  // It's slightly faster in this case if iterate over K first
    //     #pragma unroll
    //     for (int k = 0; k < size<2>(S); ++k) {
    //         if (predicate_K(k)) {
    //             #pragma unroll
    //             for (int m = 0; m < size<1>(S); ++m) {
    //                 if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //                     copy(tiled_copy, S(_, m, k), D(_, m, k));
    //                 } else if (Clear_OOB_MN) {
    //                     clear(D(_, m, k));
    //                 }
    //             }
    //         } else if (Clear_OOB_K) {  // There's no case where !Clear_OOB_K && Clear_OOB_MN
    //             if (Clear_OOB_MN || Is_even_MN) {
    //                 clear(D(_, _, k));
    //             } else {
    //                 #pragma unroll
    //                 for (int m = 0; m < size<1>(S); ++m) {
    //                     if (!(Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)) {
    //                         clear(D(_, m, k));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

// for tensor shape is (cols=4, m, k).
template <bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_b64(Tensor<Engine0, Layout0> const &S,
                                          Tensor<Engine1, Layout1> &D,
                                          Tensor<Engine2, Layout2> const &identity_MN,
                                          const int d,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    typedef __NATIVE_VECTOR__(2, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S(_, m, k).data().get());    // gmem
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                            row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            const int& d, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < d) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_reg_to_global(Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            const int &d, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto D_ptr = (VecType *)(reinterpret_cast<int32_t *>(D(_, m, k).data().ptr_));
            auto S_ptr = (VecType const *)(reinterpret_cast<int32_t const *>(S(_, m, k).data()));
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
            if constexpr (Is_even_K && Is_even_MN) {
                __builtin_mxc_stg_b128(D_ptr, 0, S_ptr[0], -1, true, false, false);
            } else {
                __builtin_mxc_stg_b128_predicator(D_ptr, 0, S_ptr[0], true, false, false, col_mask && row_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void copy_zero_to_global(Tensor<Engine0, Layout0> &D,
                            Tensor<Engine1, Layout1> const &identity_MN,
                            const int &d, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    typedef __NATIVE_VECTOR__(4, int) VecType;
    VecType val = {0, 0, 0, 0};
    #pragma unroll
    for (int m = 0; m < size<1>(D); ++m)
    {
        #pragma unroll
        for (int k = 0; k < size<2>(D); ++k) {
            auto D_ptr = (VecType *)(reinterpret_cast<int32_t *>(D(_, m, k).data().ptr_));
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
            __builtin_mxc_stg_b128_predicator(D_ptr, 0, val, true, false, false,
                                              col_mask && row_mask, 1, MACA_ICMP_EQ);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      Tensor<Engine3, Layout3> const &predicate_K,
                                      const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      const int &d, const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    Tensor reg = make_fragment_like(S);
    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            bool row_mask = get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN;
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            auto src_ptr = (VecType *)(S(_, m, k).data().ptr_);
            auto reg_ptr = (VecType *)(reg(_, m, k).data());
            reg_ptr[0] = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, false, true, false, false,
                                                            col_mask && row_mask, 1, MACA_ICMP_EQ);

            auto dst_ptr = (VecType *)(D(_, m, k).data().ptr_);
            __builtin_mxc_stg_b128_predicator(dst_ptr, 0, reg_ptr[0], true, false, false, col_mask && row_mask, 1, MACA_ICMP_EQ);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_global_to_reg(Tensor<Engine0, Layout0> const &S,
                            uint32_t *D_ptr, Tensor<Engine1, Layout1> const &identity_MN,
                            Tensor<Engine2, Layout2> const &predicate_K, const int max_MN=0) {

    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                const int idx = m * size<2>(S) * 4 + k * 4;
                if (Is_even_K || predicate_K(k)) {
                    cute::copy_global_to_reg(S(_, m, k), D_ptr + idx);
                } else if (Clear_OOB_K) {
                    D_ptr[idx] = 0;
                    D_ptr[idx + 1] = 0;
                    D_ptr[idx + 2] = 0;
                    D_ptr[idx + 3] = 0;
                }
            }
        } else if (Clear_OOB_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                const int idx = m * size<2>(S) * 4 + k * 4;
                D_ptr[idx] = 0;
                D_ptr[idx + 1] = 0;
                D_ptr[idx + 2] = 0;
                D_ptr[idx + 3] = 0;
            }
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void copy_global_to_reg(Tensor<Engine0, Layout0> const &S,
                            uint32_t *D_ptr, Tensor<Engine1, Layout1> const &identity_MN,
                            const int &d, const int &max_MN = 0) {

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            const int idx = m * size<2>(S) * 4 + k * 4;
            auto src_ptr = (VecType *)(S(_, m, k).data().ptr_);
            auto dst_ptr = (VecType *)(D_ptr + idx);
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
            if constexpr (Is_even_MN && Is_even_K) {
                dst_ptr[0] = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                dst_ptr[0] = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                                col_mask && row_mask, 1, MACA_ICMP_EQ);
            }
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void copy_global_to_share(Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, int* pred,
                            const int &d, const int &global_offset, const int &col_offset, const int &max_MN = 0) {

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            const int idx = m * size<2>(S) * 4 + k * 4;
            auto src_ptr = (VecType *)(S(_, m, k).data().ptr_ + global_offset);
            auto dst_ptr = (VecType *)(D(_, m, k).data().ptr_ + col_offset);
            bool col_mask = Is_even_K || (pred[k + int(size<1>(S))]) < d;
            bool row_mask = Is_even_MN || pred[m] < max_MN;
            if constexpr (Is_even_K && Is_even_MN) {
                __builtin_mxc_ldg_b128_bsm(dst_ptr, src_ptr, 0, -1, true, true, false, true);
            } else {
                __builtin_mxc_ldg_b128_bsm_predicator(dst_ptr, src_ptr, 0, true, true, false, true,
                                                      col_mask && row_mask, 1, MACA_ICMP_EQ);
            }
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine0, typename Layout0>
__forceinline__ __device__ void copy_reg_to_share(uint32_t *S_ptr, Tensor<Engine0, Layout0> &D) {

    #pragma unroll
    for (int m = 0; m < size<1>(D); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(D); ++k) {
            const int idx = m * size<2>(D) * 4 + k * 4;
            cute::copy_reg_to_share(S_ptr + idx, D(_, m, k));
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true, int ldg_type = 1,
            int N = 4, typename Engine0, typename Layout0>
__forceinline__ __device__ void copy_global_to_reg_V(Tensor<Engine0, Layout0> const &S,
                                            uint32_t *D_ptr, const int* pred_Row, const int* offset, const int &d, int max_MN = 0) {


    static_assert(N == 2 || N == 4 || N == 8);
    if constexpr(ldg_type == 0) { //Using ldg_b64
        typedef __NATIVE_VECTOR__(2, int) VecType;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            auto src_ptr = (VecType *)(reinterpret_cast<uint32_t *>(S(_, _0{}, _0{}).data().ptr_ + offset[i]));
            auto dst_ptr = (VecType *)(D_ptr + 2 * i);
            bool col_mask = Is_even_K || pred_Row[N + i] < d;
            bool row_mask = Is_even_MN || pred_Row[i] < max_MN;
            if constexpr (Is_even_MN && Is_even_K) {
                dst_ptr[0] = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
            } else {
                dst_ptr[0] = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                            col_mask && row_mask, 1, MACA_ICMP_EQ);
            }
        }
        return;
    }

    if constexpr(ldg_type == 2 && N == 8) { //ldg_b128 & ldg_b64 mixed
        typedef __NATIVE_VECTOR__(4, int) VecB128;
        #pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            auto src_ptr = (VecB128 *)(reinterpret_cast<uint32_t *>(S(_, _0{}, _0{}).data().ptr_ + offset[i]));
            auto dst_ptr = (VecB128 *)(D_ptr + 4 * i);
            bool col_mask = Is_even_K || pred_Row[N + i] < d;
            bool row_mask = Is_even_MN || pred_Row[i] < max_MN;
            if constexpr (Is_even_MN && Is_even_K) {
                dst_ptr[0] = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                dst_ptr[0] = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                            col_mask && row_mask, 1, MACA_ICMP_EQ);
            }
        }

        typedef __NATIVE_VECTOR__(2, int) VecB64;
        #pragma unroll 4
        for (int i = 4; i < N; ++i) {
            auto src_ptr = (VecB64 *)(reinterpret_cast<uint32_t *>(S(_, _0{}, _0{}).data().ptr_ + offset[i]));
            auto dst_ptr = (VecB64 *)(D_ptr + 16 + 2 * (i - 4));
            bool col_mask = Is_even_K || pred_Row[N + i] < d;
            bool row_mask = Is_even_MN || pred_Row[i] < max_MN;
            if constexpr (Is_even_MN && Is_even_K) {
                dst_ptr[0] = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
            } else {
                dst_ptr[0] = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                            col_mask && row_mask, 1, MACA_ICMP_EQ);
            }
        }
        return;
    }

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        auto src_ptr = (VecType *)(reinterpret_cast<uint32_t *>(S(_, _0{}, _0{}).data().ptr_ + offset[i]));
        auto dst_ptr = (VecType *)(D_ptr + 4 * i);
        bool col_mask = Is_even_K || pred_Row[N + i] < d;
        bool row_mask = Is_even_MN || pred_Row[i] < max_MN;
        if constexpr (Is_even_MN && Is_even_K) {
            dst_ptr[0] = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
        } else {
            dst_ptr[0] = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                        col_mask && row_mask, 1, MACA_ICMP_EQ);
        }
    }

}
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int sts_type = 1, int N = 4, bool Is_perm_4x4 = true>
__forceinline__ __device__ void copy_reg_to_share_V(uint32_t *S_ptr, uint32_t **D_ptr,const uint32_t* perm_mask) {


    static_assert(N == 2 || N == 4 || N == 8);
    if constexpr(sts_type == 0) {
        if constexpr (N == 2) {

            auto dst_ptr = S_ptr;
            auto a = __builtin_mxc_byte_perm(dst_ptr[2], dst_ptr[0], perm_mask[0]);
            dst_ptr[2] = __builtin_mxc_byte_perm(dst_ptr[2], dst_ptr[0], perm_mask[1]);
            dst_ptr[0] = a;
            a = __builtin_mxc_byte_perm(dst_ptr[3], dst_ptr[1], perm_mask[0]);
            dst_ptr[3] = __builtin_mxc_byte_perm(dst_ptr[3], dst_ptr[1], perm_mask[1]);
            dst_ptr[1] = a;

            #pragma unroll 2
            for (int i = 0; i < 2; ++i) {
                auto src_ptr = reinterpret_cast<uint64_t *>(S_ptr + 2 * i);
                auto sm_ptr = reinterpret_cast<uint64_t *>(D_ptr[i]);
                sm_ptr[0] = src_ptr[0];
            }
            return;
        }

        for (int i = 0; i < N / 4; ++i) {
            auto dst_ptr = S_ptr + i * 8;
            auto a = __builtin_mxc_byte_perm(dst_ptr[2], dst_ptr[0], perm_mask[0]);
            dst_ptr[2] = __builtin_mxc_byte_perm(dst_ptr[2], dst_ptr[0], perm_mask[1]);
            dst_ptr[0] = a;
            a = __builtin_mxc_byte_perm(dst_ptr[3], dst_ptr[1], perm_mask[0]);
            auto b = __builtin_mxc_byte_perm(dst_ptr[3], dst_ptr[1], perm_mask[1]);
            dst_ptr[1] = __builtin_mxc_byte_perm(dst_ptr[6], dst_ptr[4], perm_mask[0]);
            dst_ptr[3] = __builtin_mxc_byte_perm(dst_ptr[6], dst_ptr[4], perm_mask[1]);
            dst_ptr[4] = a;
            dst_ptr[6] = b;
            a = __builtin_mxc_byte_perm(dst_ptr[7], dst_ptr[5], perm_mask[0]);
            dst_ptr[7] = __builtin_mxc_byte_perm(dst_ptr[7], dst_ptr[5], perm_mask[1]);
            dst_ptr[5] = a;
        }

        #pragma unroll
        for (int i = 0; i < N; ++i) {
            auto src_ptr = reinterpret_cast<uint64_t *>(S_ptr + 2 * i);
            auto dst_ptr = reinterpret_cast<uint64_t *>(D_ptr[i]);
            dst_ptr[0] = src_ptr[0];
        }
        return;
    }

    if constexpr(sts_type == 2 && N == 8) { //Using sts_b128 first and then sts_b64
        auto ptr = S_ptr;
        auto a = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[0]);
        ptr[4] = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[1]);
        ptr[0] = a;
        a = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[0]);
        auto b = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[1]);
        ptr[1] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[0]);
        ptr[5] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[1]);
        ptr[8] = a;
        ptr[12] = b;
        a = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[0]);
        ptr[13] = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[1]);
        ptr[9] = a;

        a = __builtin_mxc_byte_perm(ptr[6], ptr[2], perm_mask[0]);
        ptr[6] = __builtin_mxc_byte_perm(ptr[6], ptr[2], perm_mask[1]);
        ptr[2] = a;
        a = __builtin_mxc_byte_perm(ptr[7], ptr[3], perm_mask[0]);
        b = __builtin_mxc_byte_perm(ptr[7], ptr[3], perm_mask[1]);
        ptr[3] = __builtin_mxc_byte_perm(ptr[14], ptr[10], perm_mask[0]);
        ptr[7] = __builtin_mxc_byte_perm(ptr[14], ptr[10], perm_mask[1]);
        ptr[10] = a;
        ptr[14] = b;
        a = __builtin_mxc_byte_perm(ptr[15], ptr[11], perm_mask[0]);
        ptr[15] = __builtin_mxc_byte_perm(ptr[15], ptr[11], perm_mask[1]);
        ptr[11] = a;

        typedef __NATIVE_VECTOR__(4, int) VecType;
        #pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            auto src_ptr = (VecType *)(S_ptr + 4 * i);
            auto dst_ptr = (VecType *)(D_ptr[i]);
            dst_ptr[0] = src_ptr[0];
        }

        ptr = S_ptr + 16;
        a = __builtin_mxc_byte_perm(ptr[2], ptr[0], perm_mask[0]);
        ptr[2] = __builtin_mxc_byte_perm(ptr[2], ptr[0], perm_mask[1]);
        ptr[0] = a;
        a = __builtin_mxc_byte_perm(ptr[3], ptr[1], perm_mask[0]);
        b = __builtin_mxc_byte_perm(ptr[3], ptr[1], perm_mask[1]);
        ptr[1] = __builtin_mxc_byte_perm(ptr[6], ptr[4], perm_mask[0]);
        ptr[3] = __builtin_mxc_byte_perm(ptr[6], ptr[4], perm_mask[1]);
        ptr[4] = a;
        ptr[6] = b;
        a = __builtin_mxc_byte_perm(ptr[7], ptr[5], perm_mask[0]);
        ptr[7] = __builtin_mxc_byte_perm(ptr[7], ptr[5], perm_mask[1]);
        ptr[5] = a;

        #pragma unroll
        for (int i = 4; i < N; ++i) {
            auto src_ptr = reinterpret_cast<uint64_t *>(S_ptr + 16 + 2 * (i - 4));
            auto dst_ptr = reinterpret_cast<uint64_t *>(D_ptr[i]);
            dst_ptr[0] = src_ptr[0];
        }

        return;

    }

    typedef __NATIVE_VECTOR__(4, int) VecType;
    if constexpr (N == 2) {
        auto dst_ptr = S_ptr;
        auto a = __builtin_mxc_byte_perm(dst_ptr[4], dst_ptr[0], perm_mask[0]);
        dst_ptr[4] = __builtin_mxc_byte_perm(dst_ptr[4], dst_ptr[0], perm_mask[1]);
        dst_ptr[0] = a;
        a = __builtin_mxc_byte_perm(dst_ptr[5], dst_ptr[1], perm_mask[0]);
        dst_ptr[5] = __builtin_mxc_byte_perm(dst_ptr[5], dst_ptr[1], perm_mask[1]);
        dst_ptr[1] = a;
        a = __builtin_mxc_byte_perm(dst_ptr[6], dst_ptr[2], perm_mask[0]);
        dst_ptr[6] = __builtin_mxc_byte_perm(dst_ptr[6], dst_ptr[2], perm_mask[1]);
        dst_ptr[2] = a;
        a = __builtin_mxc_byte_perm(dst_ptr[7], dst_ptr[3], perm_mask[0]);
        dst_ptr[7] = __builtin_mxc_byte_perm(dst_ptr[7], dst_ptr[3], perm_mask[1]);
        dst_ptr[3] = a;

        #pragma unroll 2
        for (int i = 0; i < 2; ++i) {
            auto src_ptr = (VecType *)(S_ptr + 4 * i);
            auto dst_ptr = (VecType *)(D_ptr[i]);
            dst_ptr[0] = src_ptr[0];
        }
        return;
    }

    if (N == 4 && Is_perm_4x4 == false) {
        auto ptr = S_ptr;
        auto a = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[0]);
        ptr[4] = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[1]);
        ptr[0] = a;
        a = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[0]);
        ptr[5] = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[1]);
        ptr[1] = a;
        a = __builtin_mxc_byte_perm(ptr[6], ptr[2], perm_mask[0]);
        ptr[6] = __builtin_mxc_byte_perm(ptr[6], ptr[2], perm_mask[1]);
        ptr[2] = a;
        a = __builtin_mxc_byte_perm(ptr[7], ptr[3], perm_mask[0]);
        ptr[7] = __builtin_mxc_byte_perm(ptr[7], ptr[3], perm_mask[1]);
        ptr[3] = a;

        ptr = S_ptr + 8;
        auto b = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[0]);
        ptr[4] = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[1]);
        ptr[0] = b;
        b = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[0]);
        ptr[5] = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[1]);
        ptr[1] = b;
        b = __builtin_mxc_byte_perm(ptr[6], ptr[2], perm_mask[0]);
        ptr[6] = __builtin_mxc_byte_perm(ptr[6], ptr[2], perm_mask[1]);
        ptr[2] = b;
        b = __builtin_mxc_byte_perm(ptr[7], ptr[3], perm_mask[0]);
        ptr[7] = __builtin_mxc_byte_perm(ptr[7], ptr[3], perm_mask[1]);
        ptr[3] = b;


        #pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            auto src_ptr = (VecType *)(S_ptr + 4 * i);
            auto dst_ptr = (VecType *)(D_ptr[i]);
            dst_ptr[0] = src_ptr[0];
        }

        return;
    }

    #pragma unroll
    for (int i = 0; i < N / 4; ++i) {
        auto ptr = S_ptr + i * 16;
        auto a = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[0]);
        ptr[4] = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[1]);
        ptr[0] = a;
        a = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[0]);
        auto b = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[1]);
        ptr[1] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[0]);
        ptr[5] = __builtin_mxc_byte_perm(ptr[12], ptr[8], perm_mask[1]);
        ptr[8] = a;
        ptr[12] = b;
        a = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[0]);
        ptr[13] = __builtin_mxc_byte_perm(ptr[13], ptr[9], perm_mask[1]);
        ptr[9] = a;

        a = __builtin_mxc_byte_perm(ptr[6], ptr[2], perm_mask[0]);
        ptr[6] = __builtin_mxc_byte_perm(ptr[6], ptr[2], perm_mask[1]);
        ptr[2] = a;
        a = __builtin_mxc_byte_perm(ptr[7], ptr[3], perm_mask[0]);
        b = __builtin_mxc_byte_perm(ptr[7], ptr[3], perm_mask[1]);
        ptr[3] = __builtin_mxc_byte_perm(ptr[14], ptr[10], perm_mask[0]);
        ptr[7] = __builtin_mxc_byte_perm(ptr[14], ptr[10], perm_mask[1]);
        ptr[10] = a;
        ptr[14] = b;
        a = __builtin_mxc_byte_perm(ptr[15], ptr[11], perm_mask[0]);
        ptr[15] = __builtin_mxc_byte_perm(ptr[15], ptr[11], perm_mask[1]);
        ptr[11] = a;

        #pragma unroll 4
        for (int j = 0; j < 4; ++j) {
            auto src_ptr = (VecType *)(S_ptr + i * 16 + 4 * j);
            auto dst_ptr = (VecType *)(D_ptr[j + 4 * i]);
            dst_ptr[0] = src_ptr[0];
        }
    }

    return;

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int lds_Tuple=1>
__forceinline__ __device__ void tensor_trans(void *dst, const uint32_t &stride) {

    auto dst_ptr = reinterpret_cast<uint32_t *>(dst);
    auto a = __builtin_mxc_byte_perm(dst_ptr[2], dst_ptr[0], 0x05040100);
    dst_ptr[2] = __builtin_mxc_byte_perm(dst_ptr[2], dst_ptr[0], 0x07060302);
    dst_ptr[0] = a;
    a = __builtin_mxc_byte_perm(dst_ptr[3], dst_ptr[1], 0x05040100);
    auto b = __builtin_mxc_byte_perm(dst_ptr[3], dst_ptr[1], 0x07060302);
    dst_ptr[1] = __builtin_mxc_byte_perm(dst_ptr[6], dst_ptr[4], 0x05040100);
    dst_ptr[3] = __builtin_mxc_byte_perm(dst_ptr[6], dst_ptr[4], 0x07060302);
    dst_ptr[4] = a;
    dst_ptr[6] = b;
    a = __builtin_mxc_byte_perm(dst_ptr[7], dst_ptr[5], 0x05040100);
    dst_ptr[7] = __builtin_mxc_byte_perm(dst_ptr[7], dst_ptr[5], 0x07060302);
    dst_ptr[5] = a;

    #pragma unroll
    for(int i = 1; i < lds_Tuple; ++i) {
        dst_ptr = dst_ptr + stride;
        a = __builtin_mxc_byte_perm(dst_ptr[2], dst_ptr[0], 0x05040100);
        dst_ptr[2] = __builtin_mxc_byte_perm(dst_ptr[2], dst_ptr[0], 0x07060302);
        dst_ptr[0] = a;
        a = __builtin_mxc_byte_perm(dst_ptr[3], dst_ptr[1], 0x05040100);
        b = __builtin_mxc_byte_perm(dst_ptr[3], dst_ptr[1], 0x07060302);
        dst_ptr[1] = __builtin_mxc_byte_perm(dst_ptr[6], dst_ptr[4], 0x05040100);
        dst_ptr[3] = __builtin_mxc_byte_perm(dst_ptr[6], dst_ptr[4], 0x07060302);
        dst_ptr[4] = a;
        dst_ptr[6] = b;
        a = __builtin_mxc_byte_perm(dst_ptr[7], dst_ptr[5], 0x05040100);
        dst_ptr[7] = __builtin_mxc_byte_perm(dst_ptr[7], dst_ptr[5], 0x07060302);
        dst_ptr[5] = a;
    }

}

//////////////////////////////////////// Mha bwd hdim128 opt ////////////////////////////////////////////
template <typename Engine, typename Layout>
__forceinline__ __device__ void swap_fragment(Tensor<Engine, Layout> &S) {
    using data_type = typename Engine::value_type;
    static_assert(decltype(size<0>(S))::value == 8);
    static_assert(std::is_same_v<data_type, mctlass::half_t> || std::is_same_v<data_type, mctlass::bfloat16_t>);

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(S); ++n) {
            uint64_t *first = reinterpret_cast<uint64_t *>(S(_, m, n).data());
            uint64_t *second = first + 1;
            uint64_t tmp = *first;
            *first = *second;
            *second = tmp;
        }
    }
}

template <typename T>
__forceinline__ __device__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<typename Engine, typename Layout>
__forceinline__ __device__ void stmatrix_trans(Tensor<Engine, Layout> &tCrC, void *sC) {
    using Dtype = typename Engine::value_type;
    static_assert(std::is_same_v<Dtype, mctlass::half_t> || std::is_same_v<Dtype, mctlass::bfloat16_t>);
    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == _4{});
    CUTE_STATIC_ASSERT_V(size<2>(tCrC) == _2{});

    #pragma unroll
    for (int i = 0; i < size<2>(tCrC); ++i) {
        auto r = reinterpret_cast<uint32_t *>(tCrC(_, 0, i).data().get());
        cute::reg_trans(r[0], r[1]);
    }

    constexpr int ThreadsPerGroup = 4;
    constexpr int SmemSizePerRow = 64;
    constexpr int SmemElemsPerLoad = sizeof(cute::uint64_t) / sizeof(Dtype);
    constexpr int SmemThreadsPerRow = SmemSizePerRow / SmemElemsPerLoad;

    auto row = __lane_id() / SmemThreadsPerRow;
    auto col = __lane_id() % SmemThreadsPerRow;
    col ^= (row * ThreadsPerGroup);

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sC) + threadIdx.x / 64 * (SmemElemsPerLoad * 64) + row * SmemSizePerRow + SmemElemsPerLoad * col;
    Tensor tCsC = make_tensor(make_smem_ptr(smem_ptr), make_layout(Shape<_4, _1, _2>{}, Stride<_1, _0, Int<16*SmemSizePerRow>>{}));
    cute::copy(tCrC, tCsC);
}

// for tiled mma 2x2, tile size 64x32 of A
template<typename Engine, typename TLayout>
__forceinline__ __device__ void ldmatrix_trans(Tensor<Engine, TLayout> &tArA, void *sA) {
    using Dtype = typename Engine::value_type;
    static_assert(std::is_same_v<Dtype, mctlass::half_t> || std::is_same_v<Dtype, mctlass::bfloat16_t>);
    CUTE_STATIC_ASSERT_V(size<0>(tArA) == _4{});

    constexpr int ThreadsPerGroup = 4;
    constexpr int SmemSizePerRow = 64;
    constexpr int SmemElemsPerLoad = sizeof(cute::uint64_t) / sizeof(Dtype);

    using LdsLayoutAtom = decltype(tile_to_shape(Layout<Shape<_4, Int<ThreadsPerGroup>>, Stride<Int<ThreadsPerGroup>, _1>>{}, Shape<_4, _16>{}));
    LdsLayoutAtom layout_atom;
    auto coord = layout_atom.get_hier_coord(__lane_id());
    auto row = cute::get<0>(coord);
    auto col_coord = cute::get<1>(coord);
    auto col = cute::get<0>(col_coord) + cute::get<1>(col_coord) * ThreadsPerGroup;
    col ^= (row * ThreadsPerGroup);

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sA) + threadIdx.x / 64 % 2 * (8 * SmemSizePerRow) + row * SmemSizePerRow + SmemElemsPerLoad * col;
    Tensor tAsA = make_tensor(make_smem_ptr(smem_ptr), make_layout(Shape<_4, _2, _2>{}, Stride<_1, Int<16*SmemSizePerRow>, Int<4*SmemSizePerRow>>{}));
    cute::copy(tAsA, tArA);
}

// tile 32x64,  wave 2x2
template<typename Engine, typename Layout>
__forceinline__ __device__ auto make_thr_tensor_stmatrix_trans(Tensor<Engine, Layout> &sC) {
    using Dtype = typename Engine::value_type;
    static_assert(std::is_same_v<Dtype, mctlass::half_t> || std::is_same_v<Dtype, mctlass::bfloat16_t>);

    constexpr int ThreadsPerGroup = 4;
    constexpr int SmemSizePerRow = 64;
    constexpr int SmemElemsPerLoad = sizeof(cute::uint64_t) / sizeof(Dtype);
    constexpr int SmemThreadsPerRow = SmemSizePerRow / SmemElemsPerLoad;

    auto row = __lane_id() / SmemThreadsPerRow;
    auto col = __lane_id() % SmemThreadsPerRow;
    col ^= (row * ThreadsPerGroup);

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sC.data().get()) + threadIdx.x / 64 * (SmemElemsPerLoad * 64) + row * SmemSizePerRow + SmemElemsPerLoad * col;
    return make_tensor(make_smem_ptr(smem_ptr), make_layout(Shape<_4, _1, _2>{}, Stride<_1, _0, Int<16*SmemSizePerRow>>{}));
}

template<typename Engine, typename Layout>
__forceinline__ __device__ auto make_thr_tensor_stmatrix_trans_64x64(Tensor<Engine, Layout> &sC) {
    using Dtype = typename Engine::value_type;
    static_assert(std::is_same_v<Dtype, mctlass::half_t> || std::is_same_v<Dtype, mctlass::bfloat16_t>);

    constexpr int ThreadsPerGroup = 4;
    constexpr int SmemSizePerRow = 64;
    constexpr int SmemElemsPerLoad = sizeof(cute::uint64_t) / sizeof(Dtype);//4
    constexpr int SmemThreadsPerRow = SmemSizePerRow / SmemElemsPerLoad;//16

    auto row = __lane_id() / SmemThreadsPerRow;
    auto col = __lane_id() % SmemThreadsPerRow;
    col ^= (row * ThreadsPerGroup);

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sC.data().get()) + threadIdx.x / 64 * (SmemElemsPerLoad * 64) + row * SmemSizePerRow + SmemElemsPerLoad * col;
    // return make_tensor(make_smem_ptr(smem_ptr), make_layout(Shape<_4, _2, _2>{}, Stride<_1, Int<16*SmemSizePerRow>, Int<2*16*SmemSizePerRow>>{}));
    return make_tensor(make_smem_ptr(smem_ptr), make_layout(Shape<_4, _1, _4>{}, Stride<_1, _0, Int<16*SmemSizePerRow>>{}));
}


template<typename Tensor0>
__forceinline__ __device__ void shuffle_4x4(Tensor0 &t) {
    #pragma unroll
    for (int i = 0; i < size<2>(t); ++i) {
        auto r = reinterpret_cast<uint32_t *>(t(_, 0, i).data().get());
        cute::reg_trans(r[0], r[1]);
    }
}

// for tiled mma 2x2, tile size 64x32 of A
template<typename Engine0, typename Layout0>
__forceinline__ __device__ auto make_thr_tensor_ldmatrix_trans(Tensor<Engine0, Layout0> &sA) {
    using Dtype = typename Engine0::value_type;
    static_assert(std::is_same_v<Dtype, mctlass::half_t> || std::is_same_v<Dtype, mctlass::bfloat16_t>);

    constexpr int ThreadsPerGroup = 4;
    constexpr int SmemSizePerRow = 64;
    constexpr int SmemElemsPerLoad = sizeof(cute::uint64_t) / sizeof(Dtype);

    using LdsLayoutAtom = decltype(tile_to_shape(Layout<Shape<_4, Int<ThreadsPerGroup>>, Stride<Int<ThreadsPerGroup>, _1>>{}, Shape<_4, _16>{}));
    LdsLayoutAtom layout_atom;
    auto coord = layout_atom.get_hier_coord(__lane_id());
    auto row = cute::get<0>(coord);
    auto col_coord = cute::get<1>(coord);
    auto col = cute::get<0>(col_coord) + cute::get<1>(col_coord) * ThreadsPerGroup;
    col ^= (row * ThreadsPerGroup);

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sA.data().get()) + threadIdx.x / 64 % 2 * (8 * SmemSizePerRow) + row * SmemSizePerRow + SmemElemsPerLoad * col;
    return make_tensor(make_smem_ptr(smem_ptr), make_layout(Shape<_4, _2, _2>{}, Stride<_1, Int<16*SmemSizePerRow>, Int<4*SmemSizePerRow>>{}));
}

// Tensor D must be reset in advance
template <bool Is_even_MN=true, bool Is_even_K=true, bool Is_async=false,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_4x4(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                         Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                         Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    static_assert(decltype(size<0, 0>(S))::value == 4);
    static_assert(decltype(size<0, 1>(S))::value == 4);
    static_assert(decltype(size<1>(S))::value == 1);
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    Layout s_l = S.layout();
    Tensor S_reshape = make_tensor(S.data(), make_layout(get<0, 0>(s_l), get<0, 1>(s_l), get<2>(s_l)));
    Layout d_l = D.layout();
    Tensor D_reshape = make_tensor(D.data(), make_layout(get<0, 0>(d_l), get<0, 1>(d_l), get<2>(d_l)));

    #pragma unroll
    for (int m = 0; m < size<1>(S_reshape); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, 0, 0)) + m < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S_reshape); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    if (Is_async) {
                        *(reinterpret_cast<uint64_t *>(D_reshape(_, m, k).data())) =
                            __builtin_mxc_load_global_async64(reinterpret_cast<uint64_t *>(S_reshape(_, m, k).data().get()));
                    } else {
                        cute::copy(S_reshape(_, m, k), D_reshape(_, m, k));
                    }
                }
            }
        }
    }
}

// Tensor D0 and D1 must be reset in advance.
template <bool Is_even_MN=true, bool Is_even_K=true, typename TiledCopy,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3,
          typename Engine4, typename Layout4, typename Engine5, typename Layout5>
__forceinline__ __device__ void copy2_4x4(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S0,Tensor<Engine1, Layout1> &D0,
                                          Tensor<Engine2, Layout2> const &S1, Tensor<Engine3, Layout3> &D1, Tensor<Engine4, Layout4> const &identity_MN,
                                          Tensor<Engine5, Layout5> const &predicate_K, const int max_MN=0) {
    Layout s_l = S0.layout();
    Tensor S0_reshape = make_tensor(S0.data(), make_layout(get<0, 0>(s_l), get<0, 1>(s_l), get<2>(s_l)));
    Tensor S1_reshape = make_tensor(S1.data(), make_layout(get<0, 0>(s_l), get<0, 1>(s_l), get<2>(s_l)));
    Layout d_l = D0.layout();
    Tensor D0_reshape = make_tensor(D0.data(), make_layout(get<0, 0>(d_l), get<0, 1>(d_l), get<2>(d_l)));
    Tensor D1_reshape = make_tensor(D1.data(), make_layout(get<0, 0>(d_l), get<0, 1>(d_l), get<2>(d_l)));

    #pragma unroll
    for (int m = 0; m < size<1>(S0_reshape); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, 0, 0)) + m < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S0_reshape); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S0_reshape(_, m, k), D0_reshape(_, m, k));
                    cute::copy(S1_reshape(_, m, k), D1_reshape(_, m, k));
                }
            }
        }
    }
}

// for tensor shape is (cols=8, m, k).
template <bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_b128(Tensor<Engine0, Layout0> const &S,
                                          Tensor<Engine1, Layout1> &D,
                                          Tensor<Engine2, Layout2> const &identity_MN,
                                          const int d,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (uint32_t m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (uint32_t k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S(_, m, k).data().get());    // gmem
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}


// for tensor shape is (cols=8, m, k).
template <bool Is_even_MN=true, bool Is_even_K=true, typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void copy_b128_bsm_async(Tensor0 const &S,
                                                    Tensor1 &&D,
                                                    Tensor2 const &identity_MN,
                                                    const int d,
                                                    const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S(_, m, k).data().get());    // gmem pointer
            auto dst_ptr = (VecType *)(D(_, m, k).data().get());    // smem pointer
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
            if constexpr (Is_even_K && Is_even_MN) {
                __builtin_mxc_ldg_b128_bsm(dst_ptr, src_ptr, 0, -1, true, true, false, true);
            } else {
                __builtin_mxc_ldg_b128_bsm_predicator(
                    dst_ptr, // shared memory pointer
                    src_ptr, // global memory pointer
                    0,       // Immediate value,use the default value 0.
                    true,    // bool
                    true,    // bool
                    false,   // bool
                    true,    // bool,If it is true, the compiler will not insert arrive.
                    col_mask && row_mask,
                    1,
                    MACA_ICMP_EQ
                );
            }
        }
    }
}

// for tensor shape is ((cols=2, rows), m, k).
template <bool Is_even_MN=true, bool Is_even_K=true, bool is_async = false, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_multirow_b32(Tensor<Engine0, Layout0> const &S,
                                                  Tensor<Engine1, Layout1> &D,
                                                  Tensor<Engine2, Layout2> const &identity_MN,
                                                  const int d,
                                                  const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank<0>(S) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank<0>(D) == Int<2>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    static_assert(decltype(size<0, 0>(S))::value == 2);

    typedef __NATIVE_VECTOR__(2, _Float16) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            #pragma unroll
            for (int r = 0; r < size<0, 1>(S); ++r) {
                // using 1-D way to workaround(2*r, 0, 0).
                auto src_ptr = reinterpret_cast<VecType *>(&S(r*2, m, k));   // gmem
                auto dst_ptr = reinterpret_cast<VecType *>(&D(r*2, m, k));   // rf
                bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
                bool row_mask = Is_even_MN || get<0>(identity_MN(r*2, m, 0)) < max_MN;
                if constexpr (Is_even_MN && Is_even_K) {
                    *dst_ptr = __builtin_mxc_ldg_b32(src_ptr, 0, -1, true, true, false, is_async);
                } else {
                    *dst_ptr = __builtin_mxc_ldg_b32_predicator(src_ptr, 0, true, true, false, is_async,
                                                                col_mask && row_mask, 1, MACA_ICMP_EQ);
                }
            }
        }
    }
}

// for tensor shape is ((cols=4, rows), m, k).
template <bool Is_even_MN=true, bool Is_even_K=true,  bool is_async = false, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_multirow_b64(Tensor<Engine0, Layout0> const &S,
                                                  Tensor<Engine1, Layout1> &D,
                                                  Tensor<Engine2, Layout2> const &identity_MN,
                                                  const int d,
                                                  const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank<0>(S) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank<0>(D) == Int<2>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    Layout s_l = S.layout();
    Tensor S_reshape = make_tensor(S.data(), make_layout(get<0, 0>(s_l), get<0, 1>(s_l), get<2>(s_l)));
    Layout d_l = D.layout();
    Tensor D_reshape = make_tensor(D.data(), make_layout(get<0, 0>(d_l), get<0, 1>(d_l), get<2>(d_l)));


    typedef __NATIVE_VECTOR__(2, int) VecType;
    #pragma unroll
    for (uint32_t m_idx = 0; m_idx < size<1>(S); ++m_idx) {
        #pragma unroll
        for (uint32_t r = 0; r < size<1>(S_reshape); ++r) {
            #pragma unroll
            for (uint32_t k = 0; k < size<2>(S_reshape); ++k) {
                auto src_ptr = (VecType *)(S_reshape(_, r, k).data().get() +
                             m_idx * get<1>(S.stride()));    // gmem
                auto dst_ptr = (VecType *)(D_reshape(_, r, k).data() +
                             m_idx * get<1>(D.stride()));         // rf
                bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
                bool row_mask = Is_even_MN || get<0>(identity_MN(r*4, m_idx, 0)) < max_MN;  // Note: identity_MN((0, r), m_idx, 0) is wrong, maybe cute's bug
                if constexpr (Is_even_MN && Is_even_K) {
                    *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, is_async);
                } else {
                    *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, is_async,
                                                                col_mask && row_mask, 1, MACA_ICMP_EQ);
                }
            }
        }
    }
}

// for tensor shape is ((cols=4, rows), m, k).
template <bool Is_even_MN=true, bool Is_even_K=true,  bool is_async = false, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_multirow_b64_part1(Tensor<Engine0, Layout0> const &S,
                                                  Tensor<Engine1, Layout1> &D,
                                                  Tensor<Engine2, Layout2> const &identity_MN,
                                                  const int d,
                                                  const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank<0>(S) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank<0>(D) == Int<2>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    Layout s_l = S.layout();
    Tensor S_reshape = make_tensor(S.data(), make_layout(get<0, 0>(s_l), get<0, 1>(s_l), get<2>(s_l)));
    Layout d_l = D.layout();
    Tensor D_reshape = make_tensor(D.data(), make_layout(get<0, 0>(d_l), get<0, 1>(d_l), get<2>(d_l)));


    typedef __NATIVE_VECTOR__(2, int) VecType;
    #pragma unroll
    for (uint32_t m_idx = 0; m_idx < 1; ++m_idx) {
        #pragma unroll
        for (uint32_t r = 0; r < size<1>(S_reshape); ++r) {
            #pragma unroll
            for (uint32_t k = 0; k < size<2>(S_reshape); ++k) {
                auto src_ptr = (VecType *)(S_reshape(_, r, k).data().get() +
                             m_idx * get<1>(S.stride()));    // gmem
                auto dst_ptr = (VecType *)(D_reshape(_, r, k).data() +
                             m_idx * get<1>(D.stride()));         // rf
                bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
                bool row_mask = Is_even_MN || get<0>(identity_MN(r*4, m_idx, 0)) < max_MN;  // Note: identity_MN((0, r), m_idx, 0) is wrong, maybe cute's bug
                if constexpr (Is_even_MN && Is_even_K) {
                    *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, is_async);
                } else {
                    *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, is_async,
                                                                col_mask && row_mask, 1, MACA_ICMP_EQ);
                }
            }
        }
    }
}

// for tensor shape is ((cols=4, rows), m, k).
template <bool Is_even_MN=true, bool Is_even_K=true,  bool is_async = false, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_multirow_b64_part2(Tensor<Engine0, Layout0> const &S,
                                                  Tensor<Engine1, Layout1> &D,
                                                  Tensor<Engine2, Layout2> const &identity_MN,
                                                  const int d,
                                                  const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank<0>(S) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank<0>(D) == Int<2>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    Layout s_l = S.layout();
    Tensor S_reshape = make_tensor(S.data(), make_layout(get<0, 0>(s_l), get<0, 1>(s_l), get<2>(s_l)));
    Layout d_l = D.layout();
    Tensor D_reshape = make_tensor(D.data(), make_layout(get<0, 0>(d_l), get<0, 1>(d_l), get<2>(d_l)));


    typedef __NATIVE_VECTOR__(2, int) VecType;
    #pragma unroll
    for (uint32_t m_idx = 1; m_idx < size<1>(S); ++m_idx) {
        #pragma unroll
        for (uint32_t r = 0; r < size<1>(S_reshape); ++r) {
            #pragma unroll
            for (uint32_t k = 0; k < size<2>(S_reshape); ++k) {
                auto src_ptr = (VecType *)(S_reshape(_, r, k).data().get() +
                             m_idx * get<1>(S.stride()));    // gmem
                auto dst_ptr = (VecType *)(D_reshape(_, r, k).data() +
                             m_idx * get<1>(D.stride()));         // rf
                bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
                bool row_mask = Is_even_MN || get<0>(identity_MN(r*4, m_idx, 0)) < max_MN;  // Note: identity_MN((0, r), m_idx, 0) is wrong, maybe cute's bug
                if constexpr (Is_even_MN && Is_even_K) {
                    *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, is_async);
                } else {
                    *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, is_async,
                                                                col_mask && row_mask, 1, MACA_ICMP_EQ);
                }
            }
        }
    }
}

template <typename Engine0, typename Layout0>
__forceinline__ __device__ auto make_thr_tensor_st_QdOt(Tensor<Engine0, Layout0> const &smem_base) {
    using Element = typename Engine0::value_type;
    Element *smem_ptr = reinterpret_cast<Element *>(smem_base.data().get()) + threadIdx.x * 16;
    return make_tensor(make_smem_ptr(smem_ptr), make_layout(Shape<Shape<_4, _4>, _1, _1>{},
                                                            Stride<Stride<_1, _4>, _0, _0>{}));
}

template <typename Engine0, typename Layout0>
__forceinline__ __device__ auto make_thr_tensor_ld_QdOt(Tensor<Engine0, Layout0> const &smem_base) {
    using Element = typename Engine0::value_type;
    Element *smem_ptr = reinterpret_cast<Element *>(smem_base.data().get()) + threadIdx.x / 128 * 64 + __lane_id() / 16 * 256 + __lane_id() % 16 * 4;
    return make_tensor(make_smem_ptr(smem_ptr), make_layout(Shape<_4, Shape<_2, _2>, _2>{},
                                                            Stride<_1, Stride<_128, _2048>, _1024>{}));
}


#define SWIZZLE_STORE_QDO(smem_s, reg, smem_d)      \
    cute::copy(smem_s, reg);                        \
    if (tidx / 8 % 2 == 1) {                        \
        flash::swap_fragment(reg);                  \
    }                                               \
    cute::copy(reg, smem_d);

////////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ void tensor_trans(void *dst) {

    auto dst_ptr = reinterpret_cast<uint32_t *>(dst);
    auto a = __builtin_mxc_byte_perm(dst_ptr[2], dst_ptr[0], 0x05040100);
    dst_ptr[2] = __builtin_mxc_byte_perm(dst_ptr[2], dst_ptr[0], 0x07060302);
    dst_ptr[0] = a;
    a = __builtin_mxc_byte_perm(dst_ptr[3], dst_ptr[1], 0x05040100);
    auto b = __builtin_mxc_byte_perm(dst_ptr[3], dst_ptr[1], 0x07060302);
    dst_ptr[1] = __builtin_mxc_byte_perm(dst_ptr[6], dst_ptr[4], 0x05040100);
    dst_ptr[3] = __builtin_mxc_byte_perm(dst_ptr[6], dst_ptr[4], 0x07060302);
    dst_ptr[4] = a;
    dst_ptr[6] = b;
    a = __builtin_mxc_byte_perm(dst_ptr[7], dst_ptr[5], 0x05040100);
    dst_ptr[7] = __builtin_mxc_byte_perm(dst_ptr[7], dst_ptr[5], 0x07060302);
    dst_ptr[5] = a;

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int lds_Tuple = 1>
__forceinline__ __device__ void tensor_swap(void *dst, const int &dst_stride) {

    auto dst_ptr = reinterpret_cast<uint32_t *>(dst);
    swap(dst_ptr[1], dst_ptr[4]);
    swap(dst_ptr[3], dst_ptr[6]);

    #pragma unroll
    for(int j = 1; j < lds_Tuple; ++j) {
        dst_ptr = dst_ptr + dst_stride;
        swap(dst_ptr[1], dst_ptr[4]);
        swap(dst_ptr[3], dst_ptr[6]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int lds_Tuple = 1, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void copy_trans(
    Tensor0        const&& src, Tensor1             && dst,
    const int& src_stride, const int& dst_stride,
    const int& cpy_offset) {

    auto sm_ptr = src.data().ptr_ + cpy_offset;
    #pragma unroll
    for(int j = 0; j < lds_Tuple; ++j) {
        auto dst_ptr = reinterpret_cast<uint64_t *>(dst.data() + dst_stride * j);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            auto src_ptr = reinterpret_cast<uint64_t *>(sm_ptr + (i << 6) + src_stride * j);
            dst_ptr[i] = src_ptr[0];
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_perm_4x4 = true, int lds_Tuple = 1, typename Tensor0, typename Tensor1,
          typename Tensor2, typename Tensor3, typename TiledMma>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB,
                                        Tensor3 const &tCsB,
                                        TiledMma tiled_mma,
                                        const int &cpy_offset,
                                        const uint32_t &tCsB_stride = 0,
                                        const uint32_t &tCrB_stride = 0) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K

    flash::copy_trans<lds_Tuple>(tCsB(_, _, _0{}), tCrB(_, _, _0{}), tCsB_stride, tCrB_stride, cpy_offset);

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            flash::copy_trans<lds_Tuple>(tCsB(_, _, i + 1), tCrB(_, _, i + 1), tCsB_stride, tCrB_stride, cpy_offset);
        }
        if (Is_perm_4x4 == false) {
            tensor_swap<lds_Tuple>(tCrB(_, _, i).data(), tCrB_stride >> 1);
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int lds_Tuple = 1, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm_reg(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB,
                                        TiledMma tiled_mma,
                                        const uint32_t &tCrB_stride) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K

    flash::tensor_trans<lds_Tuple>(tCrB(_, _, _0{}).data(), tCrB_stride >> 1);

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            flash::tensor_trans<lds_Tuple>(tCrB(_, _, _0{}).data(), tCrB_stride >> 1);
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    // #pragma unroll
    // for (int i = 0; i < size(tensor); ++i) {
    //     tensor(i) = mctlass::fast_tanh(tensor(i) * softcap);
    // }
    static_assert(decltype(size(tensor))::value % 2 == 0);
    typedef __NATIVE_VECTOR__(2, float) Float2;
    Float2 scale_vec = {softcap, softcap};
    Float2 beta_vec = {0.0f, 0.0f};
    #pragma unroll
    for (int i = 0; i < size(tensor); i += 2) {
        // tensor(i) = mctlass::fast_tanh(tensor(i) * softcap);
        Float2 x_vec = {tensor(i), tensor(i + 1)};
        x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, beta_vec);
        tensor(i) = mctlass::fast_tanh(x_vec[0]);
        tensor(i + 1) = mctlass::fast_tanh(x_vec[1]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename QuantElementType, typename ElementType, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void kvcache_dequant_1_to_8(Tensor<Engine0, Layout0> &QuantedCache,
                                                       Tensor<Engine1, Layout1> &ScaledCache,
                                                       Tensor<Engine2, Layout2> &Scale) {
    using Shape_Cache = decltype(QuantedCache.shape());
    using Shape_Scale = decltype(ScaledCache.shape());
    static_assert(cute::is_same_v<Shape_Cache, Shape_Scale>, "QuantedCache and ScaledCache dimensions must match");

    constexpr int dim0 = Int<16>{};
    constexpr int dim1 = Int<1>{};
    constexpr int dimN = Int<2>{};
    constexpr int dimK = Int<1>{};

    CONVERT_TENSOR_TYPE(QuantElementType, ElementType, QuantedCache, ElementTypeCache)

#pragma unroll
    // int8_t cache * fp16 / bf16 scale
    for (int n = 0; n < dimN; ++n) {
#pragma unroll
        for (int k = 0; k < dimK; ++k) {
#pragma unroll
            for (int idim0 = 0; idim0 < dim0; ++idim0) {
#pragma unroll
                for (int idim1 = 0; idim1 < dim1; ++idim1) {
                    auto coord0 = make_coord(make_coord(idim0, idim1), n, k);
                    auto coord1 = make_coord(make_coord(idim0 / 8, idim1), n, k);
                    ScaledCache(coord0) = ElementTypeCache(coord0) * Scale(coord1);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void calculate_dtanh(Tensor<Engine0, Layout0> &src_tensor,
                                                Tensor<Engine1, Layout1> &dst_tensor, const float softcap) {
#pragma unroll
    for (int i = 0; i < size(src_tensor); ++i) {
        dst_tensor(i) = (1.f - (src_tensor(i) * src_tensor(i))) * softcap;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int lds_Tuple = 1, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void copy_trans_hdim96(
    Tensor0        const&& src, Tensor1             && dst,
    const uint32_t& src_stride, const uint32_t& dst_stride,
    const int& cpy_offset) {

    auto sm_ptr = src.data().ptr_ + cpy_offset;

    #pragma unroll
    for (uint32_t i = 0; i < lds_Tuple; ++i) {
        auto sptr = sm_ptr + i * src_stride;
        auto rptr = dst.data() + i * dst_stride;
        // #pragma unroll 4
        // for (uint32_t j = 0; j < 4; ++j) {
        //     auto reg_v = reinterpret_cast<uint32_t *>(rptr + (j << 1));
        //     auto share_v = reinterpret_cast<uint32_t *>(sptr + (j << 5));
        //     reg_v[0] = share_v[0];
        // }
        auto reg_v0 = reinterpret_cast<uint32_t *>(rptr);
        auto share_v0 = reinterpret_cast<uint32_t *>(sptr);
        auto reg_v1 = reinterpret_cast<uint32_t *>(rptr + 2);
        auto share_v1 = reinterpret_cast<uint32_t *>(sptr + 32);
        auto reg_v2 = reinterpret_cast<uint32_t *>(rptr + 4);
        auto share_v2 = reinterpret_cast<uint32_t *>(sptr + 64);
        auto reg_v3 = reinterpret_cast<uint32_t *>(rptr + 6);
        auto share_v3 = reinterpret_cast<uint32_t *>(sptr + 96);
        reg_v0[0] = share_v0[0];
        reg_v1[0] = share_v2[0];
        reg_v2[0] = share_v1[0];
        reg_v3[0] = share_v3[0];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int lds_Tuple = 1,typename Tensor0>
__forceinline__ __device__ void tensor_trans_hdim96(Tensor0 const&& src, const uint32_t &stride) {

    #pragma unroll
    for (uint32_t i = 0; i < lds_Tuple; ++i) {
        // auto ptr = reinterpret_cast<uint32_t *>(src.data() + i * stride);
        // auto a = __builtin_mxc_byte_perm(ptr[1], ptr[0], 0x05040100);
        // ptr[1] = __builtin_mxc_byte_perm(ptr[1], ptr[0], 0x07060302);
        // ptr[0] = a;
        // auto b = __builtin_mxc_byte_perm(ptr[3], ptr[2], 0x05040100);
        // ptr[3] = __builtin_mxc_byte_perm(ptr[3], ptr[2], 0x07060302);
        // ptr[2] = b;
        // flash::swap(ptr[1], ptr[2]);
        auto ptr = reinterpret_cast<uint32_t *>(src.data() + i * stride);
        auto a = __builtin_mxc_byte_perm(ptr[2], ptr[0], 0x05040100);
        ptr[2] = __builtin_mxc_byte_perm(ptr[2], ptr[0], 0x07060302);
        ptr[0] = a;
        auto b = __builtin_mxc_byte_perm(ptr[3], ptr[1], 0x05040100);
        ptr[3] = __builtin_mxc_byte_perm(ptr[3], ptr[1], 0x07060302);
        ptr[1] = b;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int lds_Tuple = 3, typename Tensor0, typename Tensor1,
          typename Tensor2, typename Tensor3, typename TiledMma>
__forceinline__ __device__ void gemm_rs_hdim96(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB,
                                        Tensor3 const &tCsB,
                                        TiledMma tiled_mma,
                                        const int &cpy_offset,
                                        const uint32_t &tCsB_stride = 0,
                                        const uint32_t &tCrB_stride = 0) {

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K

    flash::copy_trans_hdim96<lds_Tuple>(tCsB(_, _, _0{}), tCrB(_, _, _0{}), tCsB_stride, tCrB_stride, cpy_offset);

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            flash::copy_trans_hdim96<lds_Tuple>(tCsB(_, _, i + 1), tCrB(_, _, i + 1), tCsB_stride, tCrB_stride, cpy_offset);
        }
        flash::tensor_trans_hdim96<lds_Tuple>(tCrB(_, _, i), tCrB_stride);
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// resolves offset of a slice of a paged kv copy from gmem.
// assumes that the tensor has already been positioned at the correct head.
__forceinline__ __device__
int64_t resolve_thread_kv_page_slice_offset(const int page_block_size, const int* block_table, const int page_stride, const int row_stride, const int row_offset, const int col_offset) {
    const int virtual_page_idx = row_offset >> __builtin_ctz(page_block_size);
    const int page_offset = row_offset - virtual_page_idx * page_block_size;

    return block_table[virtual_page_idx] * ((int64_t) page_stride)
        + page_offset * (row_stride)
        + col_offset;
}

// when prefetch ldg page_idx, use the follow function
__forceinline__ __device__
int64_t resolve_thread_kv_page_slice_offset(const int page_block_size, const int page_idx, const int page_offset, const int page_stride, const int row_stride, const int col_offset) {

    return page_idx * ((int64_t) page_stride)
        + page_offset * row_stride
        + col_offset;
}

// for tensor shape is ((cols=4, rows), m, k).
template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_multirow_b64_page_one(Tensor<Engine0, Layout0> const &S_base,
                                                  Tensor<Engine1, Layout1> &S,
                                                  Tensor<Engine2, Layout2> &D,
                                                  Tensor<Engine3, Layout3> const &identity_MN,
                                                  const int d,
                                                  const int n_block,
                                                  const int *block_table,
                                                  const int page_stride,
                                                  const int row_stride,
                                                  const int page_block_size,
                                                  const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank<0>(S) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank<0>(D) == Int<2>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    Layout s_l = S.layout();
    Tensor S_reshape = make_tensor(S.data(), make_layout(get<0, 0>(s_l), get<0, 1>(s_l), get<2>(s_l)));
    Layout d_l = D.layout();
    Tensor D_reshape = make_tensor(D.data(), make_layout(get<0, 0>(d_l), get<0, 1>(d_l), get<2>(d_l)));
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / 4;
    constexpr int kGmemRowsPerThread = 4;
    constexpr bool UseWarpsNx1 = Kernel_traits::UseWarpsNx1;
    // load 4x4 per thread

    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(2, int) VecType;
    #pragma unroll
    for (int m_idx = 0; m_idx < size<1>(S); ++m_idx) {
        #pragma unroll
        for (int r = 0; r < size<1>(S_reshape); ++r) {
            const int row_offset = m_idx * kNThreads / kGmemThreadsPerRow * kGmemRowsPerThread + tidx / kGmemThreadsPerRow * kGmemRowsPerThread
                                + r + n_block * kBlockN - (UseWarpsNx1 ? 0 : tidx / 128 * kBlockN);
            const int col_offset = tidx % kGmemThreadsPerRow * 4 + (UseWarpsNx1 ? 0 : tidx / 128 * Kernel_traits::kBlockKSmem);
            const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);
            #pragma unroll
            for (int k = 0; k < size<2>(S_reshape); ++k) {
                auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S_reshape.stride()) * k);
                auto dst_ptr = (VecType *)(D_reshape(_, r, k).data() + m_idx * get<1>(D.stride()));          // rf
                bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
                bool row_mask = Is_even_MN || get<0>(identity_MN(r*4, m_idx, 0)) < max_MN;  // identity_MN((0, r), 0, 0) is
                if constexpr (Is_even_MN && Is_even_K) {
                    *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
                } else {
                    *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                                col_mask && row_mask, 1, MACA_ICMP_EQ);
                }
            }
        }
    }
}


// for tensor shape is ((cols=4, rows), m, k).
template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_multirow_b32_page_one(Tensor<Engine0, Layout0> const &S_base,
                                                  Tensor<Engine1, Layout1> &S,
                                                  Tensor<Engine2, Layout2> &D,
                                                  Tensor<Engine3, Layout3> const &identity_MN,
                                                  const int d,
                                                  const int n_block,
                                                  const int *block_table,
                                                  const int page_stride,
                                                  const int row_stride,
                                                  const int page_block_size,
                                                  const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank<0>(S) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank<0>(D) == Int<2>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    static_assert(decltype(size<1>(S))::value == 1);
    Layout s_l = S.layout();
    Tensor S_reshape = make_tensor(S.data(), make_layout(get<0, 0>(s_l), get<0, 1>(s_l), get<2>(s_l))); // (4, 4, 2)
    Layout d_l = D.layout();
    Tensor D_reshape = make_tensor(D.data(), make_layout(get<0, 0>(d_l), get<0, 1>(d_l), get<2>(d_l))); // (4, 4, 2)
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / 4;  // 16
    constexpr int kGmemRowsPerThread = 4;
    constexpr bool UseWarpsNx1 = Kernel_traits::UseWarpsNx1;
    // load 4x4 per thread

    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(2, _Float16) VecType;
    #pragma unroll
    for (int r = 0; r < size<1>(S_reshape); ++r) {
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + r + n_block * kBlockN - (UseWarpsNx1 ? 0 : tidx / 128 * kBlockN);
        const int col_offset = tidx % kGmemThreadsPerRow * 4 + (UseWarpsNx1 ? 0 : tidx / 128 * Kernel_traits::kBlockKSmem);
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S_reshape); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S_reshape.stride()) * k);
            auto dst_ptr = (VecType *)(D_reshape(_, r, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(r*4, 0, 0)) < max_MN;  // identity_MN((0, r), 0, 0) is
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b32(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b32_predicator(src_ptr, 0, true, true, false, false,
                                                            col_mask && row_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}


template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b128_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / 8;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x8 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * 8;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

// when prefetch ldg page_idx, use the follow function
template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b128_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const uint32_t *page_idx,
                                          const uint32_t *page_offset,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / 8;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x8 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * 8;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, page_idx[m], page_offset[m], page_stride, row_stride, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

// when prefetch ldg page_idx and ldgbsm, use the follow function
template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b128_page_bsm_async(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const uint32_t *page_idx,
                                          const uint32_t *page_offset,
                                          const int swz_offset,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / 8;
    constexpr int kGmemRowsPerThread = 1;

    // load 1x8 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * 8;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, page_idx[m], page_offset[m], page_stride, row_stride, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + swz_offset + global_kv_page_offset + get<2>(S.stride()) * k);
            // auto src_ptr = (VecType *)(S(_, m, k).data().get());
            auto dst_ptr = (VecType *)(D(_, m, k).data().get());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_K && Is_even_MN) {
                __builtin_mxc_ldg_b128_bsm(dst_ptr, src_ptr, 0, -1, true, true, false, true);
            } else {
                __builtin_mxc_ldg_b128_bsm_predicator(
                    dst_ptr, // shared memory pointer
                    src_ptr, // global memory pointer
                    0,       // Immediate value,use the default value 0.
                    true,    // bool
                    true,    // bool
                    false,   // bool
                    true,    // bool,If it is true, the compiler will not insert arrive.
                    col_mask && row_mask,
                    1,
                    MACA_ICMP_EQ
                );
            }
        }
    }
}

template <typename Kernel_traits, typename Engine0, typename Layout0>
__forceinline__ __device__ void copy_page(Tensor<Engine0, Layout0> &S,
                                          uint32_t *page_idx,
                                          uint32_t *page_offset,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / 8;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x8 per thread
    int tidx = threadIdx.x;
    const int log2_page_size = __builtin_ctz(page_block_size);

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        int virtual_page_idx = row_offset >> log2_page_size;
        page_offset[m] = row_offset - virtual_page_idx * page_block_size;
        page_idx[m] = block_table[virtual_page_idx];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b128_quanted_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;     // 64
    constexpr int kNThreads = Kernel_traits::kNThreads;   // 256
    constexpr int kGmemThreadsPerRow = 8;   // 8
    constexpr int kGmemRowsPerThread = 1;
    // load 1x8 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * 16;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);

#pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b32_scale_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int dequant_group=8,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kGmemThreadsPerRow = 8;
    constexpr int kGmemRowsPerThread = 1;
    using Element = typename Kernel_traits::Element;

    // load 2 fp16/bf16 scale per thread use ldg_b32
    int tidx = threadIdx.x;
    typedef __NATIVE_VECTOR__(2, _Float16) VecType;

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * 2;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);

#pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            *dst_ptr = __builtin_mxc_ldg_b32(src_ptr, 0, -1, true, true, false, false);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////


// resolves offset of a slice of a paged kv copy from gmem.
// assumes that the tensor has already been positioned at the correct head.
template <typename Kernel_traits>
__forceinline__ __device__
int64_t resolve_thread_kv_page_slice_offset(const int tidx, const int n_block_max, const int page_block_size,
                            const int* block_table, const int page_stride, const int row_stride, const int row_idx = 0) {
    constexpr int kGmemThreadsPerRow = Kernel_traits::kGmemThreadsPerRow;
    constexpr int kGmemRowsPerThread = Kernel_traits::kGmemRowsPerThread;
    constexpr int kGmemElemsPerLoad = Kernel_traits::kGmemElemsPerLoad;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    const int col_offset = tidx % kGmemThreadsPerRow * kGmemElemsPerLoad;
    const int block_row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread;
    const int global_row_offset = block_row_offset + (n_block_max - 1) * kBlockN;
    const int page_offset = global_row_offset % page_block_size;
    const int virtual_page_idx = (global_row_offset >> __builtin_ctz(page_block_size)) + row_idx;

    return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
        + page_offset * ((int64_t) row_stride)
        + col_offset;
}

template <typename Kernel_traits>
__forceinline__ __device__
int64_t resolve_thread_kv_page_slice_offset_b64(const int tidx, const int n_block_max, const int page_block_size,
                            const int* block_table, const int page_stride, const int row_stride, const int row_idx = 0) {
    constexpr int kGmemThreadsPerRow = Kernel_traits::kGmemThreadsPerRowB64;
    constexpr int kGmemRowsPerThread = Kernel_traits::kGmemRowsPerThreadB64;
    constexpr int kGmemElemsPerLoad = Kernel_traits::kGmemElemsPerLoadB64;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    const int col_offset = tidx % kGmemThreadsPerRow * kGmemElemsPerLoad;
    const int block_row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread;
    const int global_row_offset = block_row_offset + (n_block_max - 1) * kBlockN;
    const int page_offset = global_row_offset % page_block_size;
    const int virtual_page_idx = (global_row_offset >> __builtin_ctz(page_block_size)) + row_idx;

    return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
        + page_offset * ((int64_t) row_stride)
        + col_offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// for page block size = 1
template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3, typename Engine4, typename Layout4>
__forceinline__ __device__ void copy_page_one(TiledCopy tiled_copy,
                                              Tensor<Engine0, Layout0> const &S,
                                              Tensor<Engine1, Layout1> &D,
                                              Tensor<Engine2, Layout2> const &identity_MN,
                                              Tensor<Engine3, Layout3> const &predicate_K,
                                              Tensor<Engine4, Layout4> const &base,
                                              const int n_block_max,
                                              const int *block_table,
                                              const int page_stride,
                                              const int row_stride,
                                              const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    auto src_ptr = base.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(threadIdx.x, n_block_max, 1,
                                   block_table, page_stride, row_stride, m) + get<2>(S.stride()) * k;
                    *(reinterpret_cast<uint128_t *>(D(_, m, k).data().get())) = *(reinterpret_cast<uint128_t *>(src_ptr.get()));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Layout reshape function. Given a layout with modes ((v1, v2), m, k), returns (v1, v2, k),
// where v2 may be a tuple itself, in the case of swizzled smem-backed thread tiles. This ensures
// that paged and non-paged copies result in equivalently shaped, if not necessarily strided, tensors.
template <class Shape, class Stride>
__forceinline__ __device__
auto reshape_thread_tile(Layout<Shape, Stride> l) {
    return make_layout(append(get<0>(l.shape()), get<2>(l.shape())),
                        append(get<0>(l.stride()), get<2>(l.stride())));
}

// reshapes and flattens the thread tile layout. A separate function is needed for the case where
// one of the modes of l is a layout itself and must be flattened, as opposed to keeping it intact
// for the case of swizzled layouts
template <class Shape, class Stride>
__forceinline__ __device__
auto reshape_flatten_thread_tile(Layout<Shape, Stride> l) {
    auto mode_0 = filter(flatten(get<0>(l)));
    return make_layout(append(mode_0.shape(), get<2>(l.shape())),
                        append(mode_0.stride(), get<2>(l.stride())));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ decltype(auto) permute_4x4_b16(Tensor<Engine, Layout> &t) {
    using data_type = typename Engine::value_type;
    Tensor tPerm = make_tensor<data_type>(Shape<_4, _4>{});
    uint32_t v1, v2;
    uint32_t *dest;

    #pragma unroll
    for (int i = 0; i < size<2>(t); ++i) {
        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()) + 1);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()) + 1);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 2).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 3).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()) + 1);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()) + 1);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 2).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 3).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        cute::copy(tPerm, t(_, _, i));
    }
    return t;
}

template <typename Engine, typename Layout>
__forceinline__ __device__ decltype(auto) permute_2x4_b16(Tensor<Engine, Layout> &t) {
    // transpose 2x4 to 4x2
    using data_type = typename Engine::value_type;
    Tensor tPerm = make_tensor<data_type>(Shape<_4, _2>{});
    Tensor res = make_tensor(t.data(), make_layout(make_shape(_4{}, _2{}, size<2>(t)), make_stride(_1{}, _4{}, _8{})));
    uint32_t v1, v2;
    uint32_t *dest;

    #pragma unroll
    for (int i = 0; i < size<2>(t); ++i) {
        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        cute::copy(tPerm, res(_, _, i));
    }
    return res;
}

template <typename Engine0, typename Layout0>
__forceinline__ __device__ void permute_4x2_b16(Tensor<Engine0, Layout0> &t) {
    static_assert(decltype(size<0, 0>(t))::value == 2);
    static_assert(decltype(size<0, 1>(t))::value == 4);

    // transpose 2x4 to 4x2
    using data_type = typename Engine0::value_type;
    Tensor tPerm = make_tensor<data_type>(Shape<_4, _2>{});
    uint32_t v1, v2;
    uint32_t *dest;

    #pragma unroll
    for (int i = 0; i < size<1>(t); ++i) {
        #pragma unroll
        for (int j = 0; j < size<2>(t); ++j) {
            v1 = *(reinterpret_cast<uint32_t *>(t(_, i, j).data()));
            v2 = *(reinterpret_cast<uint32_t *>(t(_, i, j).data()) + 1);
            dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data());
            *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
            dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data());
            *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

            v1 = *(reinterpret_cast<uint32_t *>(t(_, i, j).data()) + 2);
            v2 = *(reinterpret_cast<uint32_t *>(t(_, i, j).data()) + 3);
            dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data()) + 1;
            *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
            dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data()) + 1;
            *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

            cute::copy(tPerm, t(_, i, j));
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ decltype(auto) permute_8x4_b16(Tensor<Engine, Layout> &t) {
    // transpose 2x4 to 4x2
    using data_type = typename Engine::value_type;
    Tensor tPerm = make_tensor<data_type>(Shape<_4, _8>{});
    Tensor res = make_tensor(t.data(), make_layout(make_shape(_4{}, _8{}, size<2>(t)), make_stride(_1{}, _4{}, _32{})));
    uint32_t v1, v2;
    uint32_t *dest;

    #pragma unroll
    for (int i = 0; i < size<2>(t); ++i) {
        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

                v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()) + 1);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()) + 1);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 2).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 3).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()) + 1);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()) + 1);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 2).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 3).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()) + 2);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()) + 2);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 4).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 5).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()) + 2);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()) + 2);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 4).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 5).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()) + 3);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()) + 3);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 6).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 7).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()) + 3);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()) + 3);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 6).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 7).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        cute::copy(tPerm, res(_, _, i));
    }
    return res;
}


// do transpose for a tensor dim 0 and dim 1
template <int sz0, int sz1, typename Engine, typename Layout>
__forceinline__ __device__ decltype(auto) permute_b16(Tensor<Engine, Layout> &t) {
    if constexpr (sz0 == 2 && sz1 == 4) {
        return permute_2x4_b16(t);
    } else if constexpr (sz0 == 4 && sz1 == 4) {
        return permute_4x4_b16(t);
    } else if constexpr (sz0 == 8 && sz1 == 4) {
        return permute_8x4_b16(t);
    }
}

// solving the bank conflict of swizzle<2,3,3>, match with offset_swz233
template <typename Engine0, typename Layout0>
__forceinline__ __device__ void swap_swz233(Tensor<Engine0, Layout0> &t) {
    if (__lane_id() >= 32) {
        flash::swap_fragment(t);
    }
}

template <typename Engine0, typename Layout0>
__forceinline__ __device__ void swap_swz334(Tensor<Engine0, Layout0> &t) {
    if (__lane_id() / 8 % 2 == 1) {
        flash::swap_fragment(t);
    }
}

// bwd, use for scaled_ds
template <bool Is_dropout>
__forceinline__ __device__ float pointwise_mult(float p, float dp, float d) {
    if constexpr(!Is_dropout) {
        return p * (dp - d);
    }
    return p * (p >= 0 ? dp - d : d);
}

template <typename Tensor0>
__forceinline__ __device__ void reorder_C(Tensor0& acc_C) {
    CUTE_STATIC_ASSERT_V(rank(acc_C) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(acc_C) == Int<16>{});
    // 4x4 transpose
    auto acc_C_reshape = make_tensor(acc_C.data(), make_layout(make_shape(Shape<_4, _4>{}, size<1>(acc_C), size<2>(acc_C))));
    auto acc_C_reorder = make_fragment_like(acc_C_reshape);
    #pragma unroll
    for (int m = 0; m < size<1>(acc_C_reshape); m++) {
        #pragma unroll
        for (int n = 0; n < size<2>(acc_C_reshape); n++) {
            #pragma unroll
            for (int i = 0; i < size<0, 0>(acc_C_reshape); i++) {
                #pragma unroll
                for (int j = 0; j < size<0, 1>(acc_C_reshape); j++) {
                    acc_C_reorder(make_coord(j, i), m, n) = acc_C_reshape(make_coord(i, j), m, n);
                }
            }
        }
    }
    cute::copy(acc_C_reorder, acc_C_reshape);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_N,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_B_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA           const& tiled_mma) {
    using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
    constexpr int TileShape_N = decltype(size<1>(TileShape_MNK{}))::value;
    constexpr int TileShape_K = decltype(size<2>(TileShape_MNK{}))::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // Divide by 2 because right now we always use 2 for the ValLayout
    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;

    auto t = make_tile(Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,   // (8, 2, 2) or (8, 4, 2)
                              Stride<_1, Int<MMAStride_N>, _8> >{},       // (1, 64, 8) or (1, 32, 8)
                       make_layout(Int<TileShape_K>{}));
    // if (cute::thread0()) {printf("make_tiled_copy_B_warpcontiguousN "); print(t); printf("\n");  }
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutB_TV(), t);
}

template <int MMA_N,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA           const& tiled_mma) {
    using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
    constexpr int TileShape_M = decltype(size<0>(TileShape_MNK{}))::value;
    constexpr int TileShape_N = decltype(size<1>(TileShape_MNK{}))::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // Divide by 2 because right now we always use 2 for the ValLayout
    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    auto t = make_tile(make_layout(Int<TileShape_M>{}),
                       Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,   // (8, 2, 2) or (8, 4, 2)
                              Stride<_1, Int<MMAStride_N>, _8> >{});       // (1, 64, 8) or (1, 32, 8)
    // if (cute::thread0()) {printf("make_tiled_copy_C_warpcontiguousN "); print(t); printf("\n");  }
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
}


#define UNPACK_GRID(bidx, bidh, bidb, type)                                             \
    if (type == 0)         {bidx =  blockIdx.x; bidh =  blockIdx.y; bidb = blockIdx.z;} \
    else if (type == 1)    {bidx =  blockIdx.x; bidh =  blockIdx.z; bidb = blockIdx.y;} \
    else if (type == 2)    {bidx =  blockIdx.y; bidh =  blockIdx.x; bidb = blockIdx.z;} \
    else if (type == 3)    {bidx =  blockIdx.y; bidh =  blockIdx.z; bidb = blockIdx.x;} \
    else if (type == 4)    {bidx =  blockIdx.z; bidh =  blockIdx.y; bidb = blockIdx.x;} \
    else if (type == 5)    {bidx =  blockIdx.z; bidh =  blockIdx.x; bidb = blockIdx.y;} \
    else                   {bidx =  blockIdx.x; bidh =  blockIdx.y; bidb = blockIdx.z;} \

#define CHECK_MSG(x, ...) do { if((x) == false) {throw std::invalid_argument(__VA_ARGS__);} }while(0)
#define CUDA_CHECK(expr) {auto x = (expr); CHECK_MSG(x == cudaSuccess, #expr + std::string(" check failed!"));}
#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

}  // namespace flash
