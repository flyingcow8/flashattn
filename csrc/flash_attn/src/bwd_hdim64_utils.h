/******************************************************************************
 * Copyright (c) 2024, Metax.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <mctlass/array.h>
#include <cutlass/cutlass.h>
#include <mctlass/numeric_conversion.h>
#include <mctlass/numeric_types.h>

constexpr int alibi_type_slopes = 0;
constexpr int alibi_type_bias = 1;

namespace flash {

__forceinline__ __device__ void shuffle_neighbouring(uint32_t &v) {
    auto tmp = __builtin_mxc_mov_raw_shfl(v, 0x0b1, 0xf, 0xf, false);

    auto low = __builtin_mxc_byte_perm(v, tmp, 0x07060302);
    auto high = __builtin_mxc_byte_perm(tmp, v, 0x05040100);

    v = high;
    if (threadIdx.x & 0x01) {
        v = low;
    }
}

__forceinline__ __device__ void trans(uint32_t &a, uint32_t &b) {
    flash::shuffle_neighbouring(a);
    flash::shuffle_neighbouring(b);
    int group = threadIdx.x % 4;
    uint32_t tmp = b;
    if (group / 2 == 1) {
        tmp = a;
    }

    uint32_t tmp2 = __builtin_mxc_mov_raw_shfl(tmp, 0x04e, 0xf, 0xf, false);
    if (group / 2 == 1) {
        a = tmp2;
    } else {
        b = tmp2;
    }
}

template <typename Tensor1, typename Tensor2>
__forceinline__ __device__ void sts_transpose(Tensor1 &tArA, Tensor2 &tAsA) {
    CUTE_STATIC_ASSERT_V(size<0>(tArA) == _4{});
    CUTE_STATIC_ASSERT_V(size<0>(tArA) == size<0>(tAsA));
    CUTE_STATIC_ASSERT_V(size<1>(tArA) == size<1>(tAsA));
    CUTE_STATIC_ASSERT_V(size<2>(tArA) == size<2>(tAsA));
    int offset = 0;
#pragma unroll
    for (int i = 0; i < size<1>(tArA); i++) {
#pragma unroll
        for (int j = 0; j < size<2>(tArA); j++) {
            uint32_t *ptr = reinterpret_cast<uint32_t *>(tArA.data().get()) + offset;
            trans(ptr[0], ptr[1]);
            cute::copy(tArA(_, i, j), tAsA(_, i, j));

            offset += 2;
        }
    }
}

template <typename Tensor1, typename Tensor2>
__forceinline__ __device__ void lds_transpose(Tensor1 &tAsA, Tensor2 &tArA) {
    CUTE_STATIC_ASSERT_V(size<0>(tArA) == _4{});
    CUTE_STATIC_ASSERT_V(size<0>(tArA) == size<0>(tAsA));
    CUTE_STATIC_ASSERT_V(size<1>(tArA) == size<1>(tAsA));
    CUTE_STATIC_ASSERT_V(size<2>(tArA) == size<2>(tAsA));
    int offset = 0;
#pragma unroll
    for (int i = 0; i < size<1>(tArA); i++) {
#pragma unroll
        for (int j = 0; j < size<2>(tArA); j++) {
            cute::copy(tAsA(_, i, j), tArA(_, i, j));
            uint32_t *ptr = reinterpret_cast<uint32_t *>(tArA.data()) + offset;
            trans(ptr[0], ptr[1]);

            offset += 2;
        }
    }
}

inline __device__ void lds_transpose2(void *reg_ptr, void *sts_ptr) {
    uint32_t tmp[2];

    auto src_ptr = reinterpret_cast<uint32_t *>(sts_ptr);
    tmp[0] = src_ptr[0];
    tmp[1] = src_ptr[1];
    trans(tmp[0], tmp[1]);
    *(reinterpret_cast<uint64_t *>(reg_ptr)) = *(reinterpret_cast<uint64_t *>(tmp));
}

template <typename Engine, typename Layout>
inline __device__ void lds_hidm64_A(Tensor<Engine, Layout> &tensor, void *lds_ptr0, void *lds_ptr1, void *lds_ptr2,
                                    void *lds_ptr3) {
    *(reinterpret_cast<uint64_t *>(tensor.data())) = *(reinterpret_cast<uint64_t *>(lds_ptr0));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 4)) = *(reinterpret_cast<uint64_t *>(lds_ptr1));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 8)) = *(reinterpret_cast<uint64_t *>(lds_ptr2));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 12)) = *(reinterpret_cast<uint64_t *>(lds_ptr3));
}

template <typename Engine, typename Layout, typename Element>
inline __device__ void lds_hidm64_B(Tensor<Engine, Layout> &tensor, Element *lds_ptr0, Element *lds_ptr1,
                                    Element *lds_ptr2, Element *lds_ptr3) {
    *(reinterpret_cast<uint64_t *>(tensor.data())) = *(reinterpret_cast<uint64_t *>(lds_ptr0 + 0 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 4)) = *(reinterpret_cast<uint64_t *>(lds_ptr0 + 1 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 8)) = *(reinterpret_cast<uint64_t *>(lds_ptr0 + 2 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 12)) = *(reinterpret_cast<uint64_t *>(lds_ptr0 + 3 * 16 * 64));

    *(reinterpret_cast<uint64_t *>(tensor.data() + 16)) = *(reinterpret_cast<uint64_t *>(lds_ptr1 + 0 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 20)) = *(reinterpret_cast<uint64_t *>(lds_ptr1 + 1 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 24)) = *(reinterpret_cast<uint64_t *>(lds_ptr1 + 2 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 28)) = *(reinterpret_cast<uint64_t *>(lds_ptr1 + 3 * 16 * 64));

    *(reinterpret_cast<uint64_t *>(tensor.data() + 32)) = *(reinterpret_cast<uint64_t *>(lds_ptr2 + 0 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 36)) = *(reinterpret_cast<uint64_t *>(lds_ptr2 + 1 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 40)) = *(reinterpret_cast<uint64_t *>(lds_ptr2 + 2 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 44)) = *(reinterpret_cast<uint64_t *>(lds_ptr2 + 3 * 16 * 64));

    *(reinterpret_cast<uint64_t *>(tensor.data() + 48)) = *(reinterpret_cast<uint64_t *>(lds_ptr3 + 0 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 52)) = *(reinterpret_cast<uint64_t *>(lds_ptr3 + 1 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 56)) = *(reinterpret_cast<uint64_t *>(lds_ptr3 + 2 * 16 * 64));
    *(reinterpret_cast<uint64_t *>(tensor.data() + 60)) = *(reinterpret_cast<uint64_t *>(lds_ptr3 + 3 * 16 * 64));
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm_B_in_reg(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, void *ldsA_ptr0,
                                              void *ldsA_ptr1, void *ldsA_ptr2, void *ldsA_ptr3, TiledMma tiled_mma) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    auto dst_ptr = tCrA.data();

    lds_transpose2(dst_ptr, ldsA_ptr0);
    lds_transpose2(dst_ptr + 4, ldsA_ptr1);
    cute::gemm(tiled_mma, tCrA(_, _, 0), tCrB(_, _, 0), acc);

    lds_transpose2(dst_ptr + 8, ldsA_ptr2);
    cute::gemm(tiled_mma, tCrA(_, _, 1), tCrB(_, _, 1), acc);

    lds_transpose2(dst_ptr + 12, ldsA_ptr3);
    cute::gemm(tiled_mma, tCrA(_, _, 2), tCrB(_, _, 2), acc);

    cute::gemm(tiled_mma, tCrA(_, _, 3), tCrB(_, _, 3), acc);
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename TiledMma>
__forceinline__ __device__ void gemm_A_in_reg(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 &tCsA,
                                              TiledMma tiled_mma) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    cute::copy(tCsA(_, _, _0{}), tCrA(_, _, _0{}));
    cute::copy(tCsA(_, _, _1{}), tCrA(_, _, _1{}));
    __builtin_mxc_schedbound_begin();
    cute::gemm(tiled_mma, tCrA(_, _, _0{}), tCrB(_, _, _0{}), acc);
    __builtin_mxc_schedbound_end();

    cute::copy(tCsA(_, _, _2{}), tCrA(_, _, _2{}));
    __builtin_mxc_schedbound_begin();
    cute::gemm(tiled_mma, tCrA(_, _, _1{}), tCrB(_, _, _1{}), acc);
    __builtin_mxc_schedbound_end();

    cute::copy(tCsA(_, _, _3{}), tCrA(_, _, _3{}));
    __builtin_mxc_schedbound_begin();
    cute::gemm(tiled_mma, tCrA(_, _, _2{}), tCrB(_, _, _2{}), acc);
    cute::gemm(tiled_mma, tCrA(_, _, _3{}), tCrB(_, _, _3{}), acc);
    __builtin_mxc_schedbound_end();
}

template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_hdim64_opt(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsA,
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
    for (int i = 0; i < size<2>(tCrA) - 1; ++i) {
        if (!A_in_regs) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
        }
        if (!B_in_regs) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }

        __builtin_mxc_schedbound_begin();
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        __builtin_mxc_schedbound_end();
    }

    __builtin_mxc_schedbound_begin();
    cute::gemm(tiled_mma, tCrA(_, _, size<2>(tCrA) - 1), tCrB(_, _, size<2>(tCrA) - 1), acc);
    __builtin_mxc_schedbound_end();
}

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2,
          typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_without_clear(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                                   Tensor<Engine1, Layout1> &D,
                                                   Tensor<Engine2, Layout2> const &identity_MN,
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
                }
            }
        }
    }
}

template <bool Is_even_MN = true, bool Is_even_K = true, typename TiledCopy, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine4, typename Layout5, typename Engine6, typename Layout7,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_4x4_hdmi64(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S0,
                                                Tensor<Engine1, Layout1> &D0, Tensor<Engine4, Layout5> const &S1,
                                                Tensor<Engine6, Layout7> &D1,
                                                Tensor<Engine2, Layout2> const &identity_MN,
                                                Tensor<Engine3, Layout3> const &predicate_K, const int max_MN = 0) {
    if (Is_even_MN) {
#pragma unroll
        for (int k = 0; k < size<2>(S0); ++k) {
            if (Is_even_K || predicate_K(k)) {
                cute::copy(tiled_copy, S0(_, 0, k), D0(_, 0, k));
                cute::copy(tiled_copy, S1(_, 0, k), D1(_, 0, k));
            }
        }
    } else {
        Layout s_l = S0.layout();
        Tensor S0_reshape = make_tensor(S0.data(), make_layout(get<0, 0>(s_l), get<0, 1>(s_l), get<2>(s_l)));
        Tensor S1_reshape = make_tensor(S1.data(), make_layout(get<0, 0>(s_l), get<0, 1>(s_l), get<2>(s_l)));
        Layout d_l = D0.layout();
        Tensor D0_reshape = make_tensor(D0.data(), make_layout(get<0, 0>(d_l), get<0, 1>(d_l), get<2>(d_l)));
        Tensor D1_reshape = make_tensor(D1.data(), make_layout(get<0, 0>(d_l), get<0, 1>(d_l), get<2>(d_l)));
#pragma unroll
        for (int m = 0; m < size<1>(S0_reshape); ++m) {
            if (get<0>(identity_MN(0, 0, 0)) + m < max_MN) {
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
}

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2,
          typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_global_reg_async(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                                      Tensor<Engine1, Layout1> &D,
                                                      Tensor<Engine2, Layout2> const &identity_MN,
                                                      Tensor<Engine3, Layout3> const &predicate_K,
                                                      const int max_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));

    typedef __NATIVE_VECTOR__(2, int) VecType;

#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    *(reinterpret_cast<uint64_t *>(D(_, m, k).data())) =
                        __builtin_mxc_load_global_async64(reinterpret_cast<uint64_t *>(S(_, m, k).data().ptr_));
                }
            }
        }
    }
}

}  // namespace flash