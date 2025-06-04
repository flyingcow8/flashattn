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

namespace flash {

template <typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template <>
__forceinline__ __device__ uint32_t relu2<mctlass::half_t>(const uint32_t x) {
    uint32_t res;
#if defined(__MACA_ARCH__)

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
template <>
__forceinline__ __device__ uint32_t relu2<mctlass::bfloat16_t>(const uint32_t x) {
    auto y = *reinterpret_cast<__maca_bfloat162 const *>(&x);
    __maca_bfloat16 zero = __maca_bfloat16(0);
    y.x = y.x > zero ? y.x : zero;
    y.y = y.y > zero ? y.y : zero;
    uint32_t res = *reinterpret_cast<uint32_t *>(&y);
    return res;
}
#endif

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

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T const &x, T const &y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
    __device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T const &x, T const &y) { return x + y; }
};

template <int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template <>
struct Allreduce<64> {
    template <typename T, typename Operator>
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

template <>
struct Allreduce<2> {
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, 1));
        return x;
    }
};

template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsA,
                                     Tensor4 const &tCsB, TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A,
                                     TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    if (!A_in_regs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    if (!B_in_regs) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    }
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) {
                cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            }
            if (!B_in_regs) {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, TiledMma tiled_mma) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_opt(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsA,
                                         Tensor4 const &tCsB, TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A,
                                         TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A,
                                         ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    if (!A_in_regs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    if (!B_in_regs) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    }
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) {
                cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            }
            if (!B_in_regs) {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
        }

        __builtin_mxc_schedbound_begin();
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        __builtin_mxc_schedbound_end();
    }
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename TiledMma, typename TiledCopy,
          typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                                        TiledMma tiled_mma, TiledCopy smem_tiled_copy_B, ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);

    return make_layout(make_layout(cute::Layout<_1>{}, get<1>(acc_layout)),
                       make_layout(get<0>(acc_layout), get<2>(acc_layout)));
};

template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    mctlass::NumericArrayConverter<To_type, From_type, numel> convert_op;

    auto frag = convert_op(*reinterpret_cast<const mctlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

#define CONVERT_TENSOR_TYPE(type_s, type_d, tensor_s, tensor_d)                                                      \
    constexpr int tensor_d##_numel = decltype(size(tensor_s))::value;                                                \
    mctlass::NumericArrayConverter<type_d, type_s, tensor_d##_numel> tensor_d##_convert_op;                          \
    auto tensor_d##_frag =                                                                                           \
        tensor_d##_convert_op(*reinterpret_cast<const mctlass::Array<type_s, tensor_d##_numel> *>(tensor_s.data())); \
    Tensor tensor_d = make_tensor(make_rmem_ptr<type_d>(&tensor_d##_frag), tensor_s.layout());

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;

    Tensor tensor_uint32 = recast<uint32_t>(tensor);
#pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, mctlass::half_t> || std::is_same_v<To_type, mctlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if 0

    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
#pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else

    CONVERT_TENSOR_TYPE(From_type, To_type, tensor, out)
    flash::relu_(out);
#endif
    return out;
}

template <int N>
__forceinline__ __device__ void cp_async_wait() {
    __builtin_mxc_arrive_gvmcnt(N);
}

__forceinline__ __device__ void sync_threads() {
    __builtin_mxc_arrive_bsmcnt(0);
    __builtin_mxc_barrier_inst();
}

__forceinline__ __device__ void barrier() {
    __builtin_mxc_barrier_inst();
}

template <int N>
__forceinline__ __device__ void barrier_gvm() {
    __builtin_mxc_arrive_gvmcnt(N);
    __builtin_mxc_barrier_inst();
}

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2,
          typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                     Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                     Tensor<Engine3, Layout3> const &predicate_K, const int max_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

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
}

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2,
          typename Layout2>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                     Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                     const int &d, const int max_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

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

template <bool Is_even_MN = true, bool Is_even_K = true, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_reg_to_global(Tensor<Engine0, Layout0> const &S, Tensor<Engine1, Layout1> &D,
                                                   Tensor<Engine2, Layout2> const &identity_MN, const int &d,
                                                   const int max_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
    typedef __NATIVE_VECTOR__(4, int) VecType;
#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
#pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto D_ptr = (VecType *)(reinterpret_cast<int32_t *>(D(_, m, k).data().ptr_));
            auto S_ptr = (VecType const *)(reinterpret_cast<int32_t const *>(S(_, m, k).data()));
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
            __builtin_mxc_stg_b128_predicator(D_ptr, 0, S_ptr[0], true, false, false, col_mask && row_mask, 1,
                                              MACA_ICMP_EQ);
        }
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
__forceinline__ __device__ void copy_zero_to_global(Tensor<Engine0, Layout0> &D,
                                                    Tensor<Engine1, Layout1> const &identity_MN, const int &d,
                                                    const int max_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    typedef __NATIVE_VECTOR__(4, int) VecType;
    VecType val = {0, 0, 0, 0};
#pragma unroll
    for (int m = 0; m < size<1>(D); ++m) {
#pragma unroll
        for (int k = 0; k < size<2>(D); ++k) {
            auto D_ptr = (VecType *)(reinterpret_cast<int32_t *>(D(_, m, k).data().ptr_));
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
            __builtin_mxc_stg_b128_predicator(D_ptr, 0, val, true, false, false, col_mask && row_mask, 1, MACA_ICMP_EQ);
        }
    }
}

template <bool Is_even_K = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S, Tensor<Engine1, Layout1> &D,
                                               Tensor<Engine2, Layout2> const &identity_MN,
                                               Tensor<Engine3, Layout3> const &predicate_K, const int max_MN = 0,
                                               const int min_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

template <bool Is_even_K = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S, Tensor<Engine1, Layout1> &D,
                                               Tensor<Engine2, Layout2> const &identity_MN, const int &d,
                                               const int max_MN = 0, const int min_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
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
            reg_ptr[0] = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, false, true, false, false, col_mask && row_mask,
                                                           1, MACA_ICMP_EQ);

            auto dst_ptr = (VecType *)(D(_, m, k).data().ptr_);
            __builtin_mxc_stg_b128_predicator(dst_ptr, 0, reg_ptr[0], true, false, false, col_mask && row_mask, 1,
                                              MACA_ICMP_EQ);
        }
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_global_to_reg(Tensor<Engine0, Layout0> const &S, uint32_t *D_ptr,
                                                   Tensor<Engine1, Layout1> const &identity_MN,
                                                   Tensor<Engine2, Layout2> const &predicate_K, const int max_MN = 0) {
    /* *************************************************
    ** TODO: Refine this function to more generalization
    * *************************************************/

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

template <bool Is_even_MN = true, bool Is_even_K = true, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
__forceinline__ __device__ void copy_global_to_reg(Tensor<Engine0, Layout0> const &S, uint32_t *D_ptr,
                                                   Tensor<Engine1, Layout1> const &identity_MN, const int &d,
                                                   const int &max_MN = 0) {
    /* *************************************************
    ** TODO: Refine this function to more generalization
    * *************************************************/
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
            dst_ptr[0] = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false, col_mask && row_mask,
                                                           1, MACA_ICMP_EQ);
        }
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
__forceinline__ __device__ void copy_global_to_share(Tensor<Engine0, Layout0> const &S, Tensor<Engine1, Layout1> &D,
                                                     int *pred, const int &d, const int &global_offset,
                                                     const int &col_offset, const int &max_MN = 0) {
    /* *************************************************
    ** TODO: Refine this function to more generalization
    * *************************************************/
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
            __builtin_mxc_ldg_b128_bsm_predicator(dst_ptr, src_ptr, 0, true, true, false, true, col_mask && row_mask, 1,
                                                  MACA_ICMP_EQ);
        }
    }
}

template <typename Engine0, typename Layout0>
__forceinline__ __device__ void copy_reg_to_share(uint32_t *S_ptr, Tensor<Engine0, Layout0> &D) {
/* *************************************************
** TODO: Refine this function to more generalization
* *************************************************/
#pragma unroll
    for (int m = 0; m < size<1>(D); ++m) {
#pragma unroll
        for (int k = 0; k < size<2>(D); ++k) {
            const int idx = m * size<2>(D) * 4 + k * 4;
            cute::copy_reg_to_share(S_ptr + idx, D(_, m, k));
        }
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, bool Is_ldg_B128 = true, int N = 4, typename Engine0,
          typename Layout0>
__forceinline__ __device__ void copy_global_to_reg_V(Tensor<Engine0, Layout0> const &S, uint32_t *D_ptr,
                                                     const int *pred_Row, const int *offset, const int &d,
                                                     int max_MN = 0) {
    /* *************************************************
    ** TODO: Refine this function to more generalization
    * *************************************************/

    static_assert(N == 2 || N == 4 || N == 8);
    if constexpr (Is_ldg_B128 == false) {
        typedef __NATIVE_VECTOR__(2, int) VecType;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            auto src_ptr = (VecType *)(reinterpret_cast<uint32_t *>(S(_, _0{}, _0{}).data().ptr_ + offset[i]));
            auto dst_ptr = (VecType *)(D_ptr + 2 * i);
            bool col_mask = Is_even_K || pred_Row[N] < d;
            bool row_mask = Is_even_MN || pred_Row[i] < max_MN;
            dst_ptr[0] = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false, col_mask && row_mask, 1,
                                                          MACA_ICMP_EQ);
        }
        return;
    }

    typedef __NATIVE_VECTOR__(4, int) VecType;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        auto src_ptr = (VecType *)(reinterpret_cast<uint32_t *>(S(_, _0{}, _0{}).data().ptr_ + offset[i]));
        auto dst_ptr = (VecType *)(D_ptr + 4 * i);
        bool col_mask = Is_even_K || pred_Row[N] < d;
        bool row_mask = Is_even_MN || pred_Row[i] < max_MN;
        dst_ptr[0] = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false, col_mask && row_mask, 1,
                                                       MACA_ICMP_EQ);
    }
}

template <bool Is_sts_B128 = true, int N = 4>
__forceinline__ __device__ void copy_reg_to_share_V(uint32_t *S_ptr, uint32_t **D_ptr, const uint32_t *perm_mask) {
    /* *************************************************
    ** TODO: Refine this function to more generalization
    * *************************************************/

    static_assert(N == 2 || N == 4 || N == 8);
    if constexpr (Is_sts_B128 == false) {
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

template <int lds_Tuple = 1>
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
    for (int i = 1; i < lds_Tuple; ++i) {
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

template <typename Engine, typename Layout>
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

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sC) + threadIdx.x / 64 * (SmemElemsPerLoad * 64) +
                      row * SmemSizePerRow + SmemElemsPerLoad * col;
    Tensor tCsC = make_tensor(make_smem_ptr(smem_ptr),
                              make_layout(Shape<_4, _1, _2>{}, Stride<_1, _0, Int<16 * SmemSizePerRow>>{}));
    cute::copy(tCrC, tCsC);
}

template <typename Engine, typename TLayout>
__forceinline__ __device__ void ldmatrix_trans(Tensor<Engine, TLayout> &tArA, void *sA) {
    using Dtype = typename Engine::value_type;
    static_assert(std::is_same_v<Dtype, mctlass::half_t> || std::is_same_v<Dtype, mctlass::bfloat16_t>);
    CUTE_STATIC_ASSERT_V(size<0>(tArA) == _4{});

    constexpr int ThreadsPerGroup = 4;
    constexpr int SmemSizePerRow = 64;
    constexpr int SmemElemsPerLoad = sizeof(cute::uint64_t) / sizeof(Dtype);

    using LdsLayoutAtom = decltype(
        tile_to_shape(Layout<Shape<_4, Int<ThreadsPerGroup>>, Stride<Int<ThreadsPerGroup>, _1>>{}, Shape<_4, _16>{}));
    LdsLayoutAtom layout_atom;
    auto coord = layout_atom.get_hier_coord(__lane_id());
    auto row = cute::get<0>(coord);
    auto col_coord = cute::get<1>(coord);
    auto col = cute::get<0>(col_coord) + cute::get<1>(col_coord) * ThreadsPerGroup;
    col ^= (row * ThreadsPerGroup);

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sA) + threadIdx.x / 64 % 2 * (8 * SmemSizePerRow) +
                      row * SmemSizePerRow + SmemElemsPerLoad * col;
    Tensor tAsA =
        make_tensor(make_smem_ptr(smem_ptr),
                    make_layout(Shape<_4, _2, _2>{}, Stride<_1, Int<16 * SmemSizePerRow>, Int<4 * SmemSizePerRow>>{}));
    cute::copy(tAsA, tArA);
}

template <typename Engine, typename Layout>
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

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sC.data().get()) + threadIdx.x / 64 * (SmemElemsPerLoad * 64) +
                      row * SmemSizePerRow + SmemElemsPerLoad * col;
    return make_tensor(make_smem_ptr(smem_ptr),
                       make_layout(Shape<_4, _1, _2>{}, Stride<_1, _0, Int<16 * SmemSizePerRow>>{}));
}

template <typename Tensor0>
__forceinline__ __device__ void shuffle_4x4(Tensor0 &t) {
#pragma unroll
    for (int i = 0; i < size<2>(t); ++i) {
        auto r = reinterpret_cast<uint32_t *>(t(_, 0, i).data().get());
        cute::reg_trans(r[0], r[1]);
    }
}

template <typename Engine0, typename Layout0>
__forceinline__ __device__ auto make_thr_tensor_ldmatrix_trans(Tensor<Engine0, Layout0> &sA) {
    using Dtype = typename Engine0::value_type;
    static_assert(std::is_same_v<Dtype, mctlass::half_t> || std::is_same_v<Dtype, mctlass::bfloat16_t>);

    constexpr int ThreadsPerGroup = 4;
    constexpr int SmemSizePerRow = 64;
    constexpr int SmemElemsPerLoad = sizeof(cute::uint64_t) / sizeof(Dtype);

    using LdsLayoutAtom = decltype(
        tile_to_shape(Layout<Shape<_4, Int<ThreadsPerGroup>>, Stride<Int<ThreadsPerGroup>, _1>>{}, Shape<_4, _16>{}));
    LdsLayoutAtom layout_atom;
    auto coord = layout_atom.get_hier_coord(__lane_id());
    auto row = cute::get<0>(coord);
    auto col_coord = cute::get<1>(coord);
    auto col = cute::get<0>(col_coord) + cute::get<1>(col_coord) * ThreadsPerGroup;
    col ^= (row * ThreadsPerGroup);

    Dtype *smem_ptr = reinterpret_cast<Dtype *>(sA.data().get()) + threadIdx.x / 64 % 2 * (8 * SmemSizePerRow) +
                      row * SmemSizePerRow + SmemElemsPerLoad * col;
    return make_tensor(
        make_smem_ptr(smem_ptr),
        make_layout(Shape<_4, _2, _2>{}, Stride<_1, Int<16 * SmemSizePerRow>, Int<4 * SmemSizePerRow>>{}));
}

template <bool Is_even_MN = true, bool Is_even_K = true, bool Is_async = false, typename TiledCopy, typename Engine0,
          typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3,
          typename Layout3>
__forceinline__ __device__ void copy_4x4(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                         Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                         Tensor<Engine3, Layout3> const &predicate_K, const int max_MN = 0) {
    static_assert(decltype(size<0, 0>(S))::value == 4);
    static_assert(decltype(size<0, 1>(S))::value == 4);
    static_assert(decltype(size<1>(S))::value == 1);
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

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
                        *(reinterpret_cast<uint64_t *>(D_reshape(_, m, k).data())) = __builtin_mxc_load_global_async64(
                            reinterpret_cast<uint64_t *>(S_reshape(_, m, k).data().get()));
                    } else {
                        cute::copy(S_reshape(_, m, k), D_reshape(_, m, k));
                    }
                }
            }
        }
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, typename TiledCopy, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3,
          typename Engine4, typename Layout4, typename Engine5, typename Layout5>
__forceinline__ __device__ void copy2_4x4(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S0,
                                          Tensor<Engine1, Layout1> &D0, Tensor<Engine2, Layout2> const &S1,
                                          Tensor<Engine3, Layout3> &D1, Tensor<Engine4, Layout4> const &identity_MN,
                                          Tensor<Engine5, Layout5> const &predicate_K, const int max_MN = 0) {
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

template <typename Engine0, typename Layout0>
__forceinline__ __device__ auto make_thr_tensor_st_QdOt(Tensor<Engine0, Layout0> const &smem_base) {
    using Element = typename Engine0::value_type;
    Element *smem_ptr = reinterpret_cast<Element *>(smem_base.data().get()) + threadIdx.x * 16;
    return make_tensor(make_smem_ptr(smem_ptr),
                       make_layout(Shape<Shape<_4, _4>, _1, _1>{}, Stride<Stride<_1, _4>, _0, _0>{}));
}

template <typename Engine0, typename Layout0>
__forceinline__ __device__ auto make_thr_tensor_ld_QdOt(Tensor<Engine0, Layout0> const &smem_base) {
    using Element = typename Engine0::value_type;
    Element *smem_ptr = reinterpret_cast<Element *>(smem_base.data().get()) + threadIdx.x / 128 * 64 +
                        __lane_id() / 16 * 256 + __lane_id() % 16 * 4;
    return make_tensor(make_smem_ptr(smem_ptr),
                       make_layout(Shape<_4, Shape<_2, _2>, _2>{}, Stride<_1, Stride<_128, _2048>, _1024>{}));
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void clear_b128(Tensor<Engine, Layout> &tensor) {
    CUTE_STATIC_ASSERT_V(rank(tensor) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(tensor) == Int<8>{});

#pragma unroll
    for (int m = 0; m < size<1>(tensor); ++m) {
#pragma unroll
        for (int k = 0; k < size<2>(tensor); ++k) {
            *reinterpret_cast<uint128_t *>(tensor(_, m, k).data().get()) = 0;
        }
    }
}

#define SWIZZLE_STORE_QDO(smem_s, reg, smem_d) \
    cute::copy(smem_s, reg);                   \
    if (tidx / 8 % 2 == 1) {                   \
        flash::swap_fragment(reg);             \
    }                                          \
    cute::copy(reg, smem_d);

__forceinline__ __device__ void copy_share_reg_trans(uint64_t smem_ptr, uint32_t *rmem_ptr, uint32_t *cpy_offset,
                                                     const int &reg_size) {
    smem_ptr = smem_ptr - cpy_offset[4];

    /* ************************************************
    ** The address attribute of src_addr has benn destoried
    ** So we need  to use __attribute__((address_space (3)))
    * *************************************************/
    uint32_t __attribute__((address_space(3))) * src_ptr[4];
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        src_ptr[i] = (uint32_t __attribute__((address_space(3))) *)(smem_ptr) + cpy_offset[i];
    }
    CUTE_UNROLL
    for (int i = 0; i < reg_size; ++i) {
        rmem_ptr[2 * i] = src_ptr[i][0];
        rmem_ptr[2 * i + 1] = src_ptr[i][1];
    }
}

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

__forceinline__ __device__ void tensor_swap(void *dst) {
    auto dst_ptr = reinterpret_cast<uint32_t *>(dst);
    swap(dst_ptr[1], dst_ptr[4]);
    swap(dst_ptr[3], dst_ptr[6]);
}

template <int ldg_Num_B = 4, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename TiledMma>
__forceinline__ __device__ void gemm_rs_hdim64(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                                               TiledMma tiled_mma, uint32_t *cpy_offset) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    auto rmem_ptr = reinterpret_cast<uint32_t *>(tCrB(_, _, _0{}).data());
    const int reg_size = size(tCrB(_, _, _0{})) / 4;
    copy_share_reg_trans(reinterpret_cast<uint64_t const>(tCsB(_, _, _0{}).data().ptr_), rmem_ptr, cpy_offset,
                         reg_size);

#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            auto rmem_ptr = reinterpret_cast<uint32_t *>(tCrB(_, _, i + 1).data());
            copy_share_reg_trans(reinterpret_cast<uint64_t const>(tCsB(_, _, i + 1).data().ptr_), rmem_ptr, cpy_offset,
                                 reg_size);
        }
        if (ldg_Num_B == 2) {
            tensor_swap(tCrB(_, _, i).data());
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <int lds_Tuple = 1>
__forceinline__ __device__ void tensor_swap(void *dst, const int &dst_stride) {
    auto dst_ptr = reinterpret_cast<uint32_t *>(dst);
    swap(dst_ptr[1], dst_ptr[4]);
    swap(dst_ptr[3], dst_ptr[6]);

#pragma unroll
    for (int j = 1; j < lds_Tuple; ++j) {
        dst_ptr = dst_ptr + dst_stride;
        swap(dst_ptr[1], dst_ptr[4]);
        swap(dst_ptr[3], dst_ptr[6]);
    }
}

template <int ldg_Num_B = 4, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename TiledMma>
__forceinline__ __device__ void gemm_rs_hdim128(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                                                TiledMma tiled_mma, uint32_t *cpy_offset, const uint32_t &tCsB_stride,
                                                const uint32_t &tCrB_stride) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    cute::copy_trans(tCsB(_, _, _0{}), tCrB(_, _, _0{}), tCsB_stride, tCrB_stride, cpy_offset);

#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy_trans(tCsB(_, _, i + 1), tCrB(_, _, i + 1), tCsB_stride, tCrB_stride, cpy_offset);
        }
        if (ldg_Num_B == 2) {
            tensor_swap<2>(tCrB(_, _, i).data(), tCrB_stride);
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <int lds_Tuple = 1, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void copy_trans(Tensor0 const &&src, Tensor1 &&dst, const int &src_stride,
                                           const int &dst_stride, const uint32_t *cpy_offset) {
    /* ***********************************************
    ** TODO: Refine this function to more generalization
    * ***********************************************/
    auto dst_ptr = reinterpret_cast<uint32_t *>(dst.data());
    auto src_addr = reinterpret_cast<uint64_t const>(src.data().ptr_);
    src_addr = src_addr - cpy_offset[4];

    /* ************************************************
    ** The address attribute of src_addr has benn destoried
    ** So we need  to use __attribute__((address_space (3)))
    * *************************************************/
    uint32_t __attribute__((address_space(3))) * src_ptr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        src_ptr[i] = (uint32_t __attribute__((address_space(3))) *)(src_addr) + cpy_offset[i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dst_ptr[2 * i] = src_ptr[i][0];
        dst_ptr[2 * i + 1] = src_ptr[i][1];
    }

#pragma unroll
    for (int j = 1; j < lds_Tuple; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            src_ptr[i] = src_ptr[i] + src_stride;
        }

        dst_ptr = dst_ptr + dst_stride;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            dst_ptr[2 * i] = src_ptr[i][0];
            dst_ptr[2 * i + 1] = src_ptr[i][1];
        }
    }
}

template <int ldg_Num_B = 4, int lds_Tuple = 1, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                                        TiledMma tiled_mma, uint32_t *cpy_offset, const uint32_t &tCsB_stride,
                                        const uint32_t &tCrB_stride) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    flash::copy_trans<lds_Tuple>(tCsB(_, _, _0{}), tCrB(_, _, _0{}), tCsB_stride, tCrB_stride, cpy_offset);

#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            flash::copy_trans<lds_Tuple>(tCsB(_, _, i + 1), tCrB(_, _, i + 1), tCsB_stride, tCrB_stride, cpy_offset);
        }
        if (ldg_Num_B == 2) {
            tensor_swap<lds_Tuple>(tCrB(_, _, i).data(), tCrB_stride);
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

__forceinline__ __host__ mcDeviceProp_t mcGetCurrentDeviceProperties() {
    int deviceId{};
    mcGetDevice(&deviceId);
    mcDeviceProp_t dprops;
    mcGetDeviceProperties(&dprops, deviceId);
    return dprops;
}

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

template <typename Engine, typename Layout>
__forceinline__ __device__ decltype(auto) permute_8x4_b16(Tensor<Engine, Layout> &t) {
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

#define CHECK_MSG(x, ...)                             \
    do {                                              \
        if ((x) == false) {                           \
            throw std::invalid_argument(__VA_ARGS__); \
        }                                             \
    } while (0)
#define CUDA_CHECK(expr)                                                    \
    {                                                                       \
        auto x = (expr);                                                    \
        CHECK_MSG(x == cudaSuccess, #expr + std::string(" check failed!")); \
    }
#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

}  // namespace flash
