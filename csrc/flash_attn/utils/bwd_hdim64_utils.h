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

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

    __forceinline__ __device__ void shuffle_neighbouring(uint32_t &v){

        auto tmp = __builtin_mxc_mov_raw_shfl(v, 0x0b1, 0xf, 0xf, false);

        auto low = __builtin_mxc_byte_perm(v, tmp, 0x07060302);  // 5,1
        auto high = __builtin_mxc_byte_perm(tmp, v, 0x05040100); // 0,4

        v = high;
        if(threadIdx.x & 0x01){
            v = low;
        }
    }

    __forceinline__ __device__ void trans(uint32_t &a,uint32_t &b){
        flash::shuffle_neighbouring(a);
        flash::shuffle_neighbouring(b);
        int group = threadIdx.x % 4;
        uint32_t tmp = b;
        if(group / 2 == 1){
            tmp = a;
        }

        uint32_t tmp2 = __builtin_mxc_mov_raw_shfl(tmp, 0x04e, 0xf, 0xf, false);
        if(group / 2 == 1){
            a = tmp2;
        }else {
            b = tmp2;
        }
    }

    // transpose and sts_b64
    template<typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void sts_transpose(Tensor1 &tArA,Tensor2 &tAsA) {
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
                // one loop copy 4 half, equals 2 register
                offset += 2;
            }
        }
    }

    template<typename Engine,typename Layout>
    inline __device__ void sts_transpose(const Tensor<Engine,Layout> &tensor,void *sts_ptr0,void *sts_ptr1){
        auto src_ptr = reinterpret_cast<uint32_t *>(tensor.data().ptr_);
        // transpose and sts64
        uint32_t tmp[2];
        tmp[0] = src_ptr[0];
        tmp[1] = src_ptr[1];
        trans(tmp[0],tmp[1]);
        *(reinterpret_cast<uint64_t *>(sts_ptr0)) = *(reinterpret_cast<uint64_t *>(tmp));

        tmp[0] = src_ptr[2];
        tmp[1] = src_ptr[3];
        trans(tmp[0],tmp[1]);
        *(reinterpret_cast<uint64_t *>(sts_ptr1)) = *(reinterpret_cast<uint64_t *>(tmp));
    }

    // lds_b64 and transpose
    template<typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void lds_transpose(Tensor1 &tAsA,Tensor2 &tArA) {
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
                // one loop copy 4 half, equals 2 register
                offset += 2;
            }
        }
    }


    // hdim96
    template<typename Engine,typename Layout>
    inline __device__ void load_A_warp2x2(Tensor<Engine,Layout> &tensor,void *lds_ptr0,void *lds_ptr1,void *lds_ptr2,void *lds_ptr3){
        *(reinterpret_cast<uint64_t *>(tensor.data())) = *(reinterpret_cast<uint64_t *>(lds_ptr0));
        *(reinterpret_cast<uint64_t *>(tensor.data() + 4)) = *(reinterpret_cast<uint64_t *>(lds_ptr1));
        *(reinterpret_cast<uint64_t *>(tensor.data() + 8)) = *(reinterpret_cast<uint64_t *>(lds_ptr2));
        *(reinterpret_cast<uint64_t *>(tensor.data() + 12)) = *(reinterpret_cast<uint64_t *>(lds_ptr3));
    }

    // hdim96
    template<typename Engine,typename Layout>
    inline __device__ void load_A_trans_warp2x2(Tensor<Engine,Layout> &tensor,void *lds_ptr0,void *lds_ptr1,void *lds_ptr2,void *lds_ptr3){

        auto dst_ptr = tensor.data();
        uint32_t tmp[2];

        auto src_ptr = reinterpret_cast<uint32_t *>(lds_ptr0);
        tmp[0] = src_ptr[0];
        tmp[1] = src_ptr[1];
        trans(tmp[0],tmp[1]);
        *(reinterpret_cast<uint64_t *>(dst_ptr)) = *(reinterpret_cast<uint64_t *>(tmp));

        src_ptr = reinterpret_cast<uint32_t *>(lds_ptr1);
        tmp[0] = src_ptr[0];
        tmp[1] = src_ptr[1];
        trans(tmp[0],tmp[1]);
        *(reinterpret_cast<uint64_t *>(dst_ptr + 4)) = *(reinterpret_cast<uint64_t *>(tmp));

        src_ptr = reinterpret_cast<uint32_t *>(lds_ptr2);
        tmp[0] = src_ptr[0];
        tmp[1] = src_ptr[1];
        trans(tmp[0],tmp[1]);
        *(reinterpret_cast<uint64_t *>(dst_ptr + 8)) = *(reinterpret_cast<uint64_t *>(tmp));

        src_ptr = reinterpret_cast<uint32_t *>(lds_ptr3);
        tmp[0] = src_ptr[0];
        tmp[1] = src_ptr[1];
        trans(tmp[0],tmp[1]);
        *(reinterpret_cast<uint64_t *>(dst_ptr + 12)) = *(reinterpret_cast<uint64_t *>(tmp));

    }


////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename Tensor0, typename Tensor1,typename Tensor2,typename Tensor3, typename TiledMma>
__forceinline__ __device__ void gemm_A_in_reg(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 &tCsA, TiledMma tiled_mma) {

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K

    cute::copy(tCsA(_, _, _0{}), tCrA(_, _, _0{}));
    cute::copy(tCsA(_, _, _1{}), tCrA(_, _, _1{}));
    cute::gemm(tiled_mma, tCrA(_, _, _0{}), tCrB(_, _, _0{}), acc);

    cute::copy(tCsA(_, _, _2{}), tCrA(_, _, _2{}));
    cute::gemm(tiled_mma, tCrA(_, _, _1{}), tCrB(_, _, _1{}), acc);

    cute::copy(tCsA(_, _, _3{}), tCrA(_, _, _3{}));
    cute::gemm(tiled_mma, tCrA(_, _, _2{}), tCrB(_, _, _2{}), acc);
    cute::gemm(tiled_mma, tCrA(_, _, _3{}), tCrB(_, _, _3{}), acc);

}

template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma>
__forceinline__ __device__ void gemm_transB(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    cute::copy(tCsB(_, _, _0{}), tCrB(_, _, _0{}));
    cute::copy(tCsA(_, _, _0{}), tCrA(_, _, _0{}));
    #pragma unroll
    for (int k = 0; k < size<2>(tCrA); ++k) {
        bool has_next_loop = k < size<2>(tCrA) - 1;
        #pragma unroll
        for (int n = 0; n < size<1>(tCrB); ++n) {
            if (has_next_loop) cute::copy(tCsB(_, n, k + 1), tCrB(_, n, k + 1));
            uint32_t *tCrB_ptr = reinterpret_cast<uint32_t *>(tCrB(_, n, k).data());
            flash::trans(tCrB_ptr[0], tCrB_ptr[1]);
            __builtin_mxc_schedbound_begin();
            #pragma unroll
            for (int m = 0; m < size<1>(tCrA); ++m) {
                if (has_next_loop) cute::copy(tCsA(_, m, k + 1), tCrA(_, m, k + 1));
                cute::gemm(tiled_mma, tCrA(_, m, k), tCrB(_, n, k), acc(_, m, n));
            }
            __builtin_mxc_schedbound_end();
        }
    }
}



template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_without_clear(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
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
                }
            }
        }
    }
}


} // namespace flash
