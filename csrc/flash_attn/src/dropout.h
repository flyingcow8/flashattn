/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "philox.cuh"
#include "utils.h"

namespace flash {

struct Dropout {
    const unsigned long long seed, offset;
    const uint8_t p_dropout_in_uint8_t;

    __forceinline__ __device__ Dropout(const unsigned long long seed, const unsigned long long offset,
                                       const uint8_t p_dropout_in_uint8_t, const int bid, const int hid, const int tid,
                                       const int nheads)
        : seed(seed),
          offset(offset + (bid * nheads + hid) * 64 + tid % 64),
          p_dropout_in_uint8_t(p_dropout_in_uint8_t) {}

    template <bool encode_dropout_in_sign_bit = false, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_dropout(Tensor<Engine, Layout> &tensor_, int block_row_start,
                                                  int block_col_start, int block_row_stride) {
        Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_dropout(tensor_.layout()));
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) { return keep ? val : (encode_dropout_in_sign_bit ? -val : T(0)); };
        static_assert(decltype(size<2>(tensor))::value % 2 == 0);
        const uint16_t p_dropout_8bit_in_uint16_t = uint16_t(p_dropout_in_uint8_t);
        const uint32_t p_dropout_8bit_in_uint32_t =
            (uint32_t(p_dropout_8bit_in_uint16_t) << 16) | uint32_t(p_dropout_8bit_in_uint16_t);

#pragma unroll
        for (int m = 0; m < size<1>(tensor); ++m, block_row_start += block_row_stride) {
            uint2 rowcol = make_uint2(block_row_start, block_col_start);
#pragma unroll
            for (int n = 0; n < size<2>(tensor) / 2; ++n, ++rowcol.y) {
                uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long &>(rowcol), offset);

                uint8_t(&rnd_8)[16] = reinterpret_cast<uint8_t(&)[16]>(random_uint4);

                if (!encode_dropout_in_sign_bit &&
                    (std::is_same<T, mctlass::half_t>::value || std::is_same<T, mctlass::bfloat16_t>::value)) {
                    uint16_t rnd_16[16];
#pragma unroll
                    for (int i = 0; i < 16; i++) {
                        rnd_16[i] = uint16_t(rnd_8[i]);
                    }
                    uint32_t(&rnd_32)[8] = reinterpret_cast<uint32_t(&)[8]>(rnd_16);
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));

#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            uint32_t mask;

                            tensor_uint32(i) &= mask;
                        }
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 2; j++) {
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            tensor(i, m, n * 2 + j) =
                                encode_dropout(rnd_8[j * 8 + i] <= p_dropout_in_uint8_t, tensor(i, m, n * 2 + j));
                        }
                        Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                    }
                }
            }
        }
    }

    template <bool encode_dropout_in_sign_bit = false, int AtomLayoutNS = 1, typename Engine, typename Layout>
    __forceinline__ __device__ void mc_apply_dropout(Tensor<Engine, Layout> &tensor, int block_row_start,
                                                     int block_col_start, int block_row_stride, int kBlockN,
                                                     int n_block) {
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) { return keep ? val : (encode_dropout_in_sign_bit ? -val : T(0)); };
        static_assert(decltype(size<0>(tensor))::value == 4);
        static_assert(decltype(size<2>(tensor))::value % 2 == 0);

#pragma unroll
        for (int m = 0; m < size<1>(tensor); ++m, block_row_start += block_row_stride) {
            uint2 rowcol = make_uint2(block_row_start, block_col_start);
            uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long &>(rowcol), offset);
            uint8_t(&rnd_8)[16] = reinterpret_cast<uint8_t(&)[16]>(random_uint4);

            int n_offset = ((kBlockN * n_block) % 64) / 16;

#pragma unroll
            for (int n = 0; n < size<2>(tensor); ++n) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if constexpr (AtomLayoutNS == 1) {
                        tensor(i, m, n) =
                            encode_dropout(rnd_8[(n + n_offset) * 4 + i] <= p_dropout_in_uint8_t, tensor(i, m, n));
                    } else if constexpr (AtomLayoutNS == 2) {
                        tensor(i, m, n) = encode_dropout(
                            rnd_8[n * 8 + i + threadIdx.x / 64 / block_row_stride * 4] <= p_dropout_in_uint8_t,
                            tensor(i, m, n));
                    } else if constexpr (AtomLayoutNS == 4) {
                        tensor(i, m, n) = encode_dropout(
                            rnd_8[n * 4 + i + threadIdx.x / 64 / block_row_stride * 4] <= p_dropout_in_uint8_t,
                            tensor(i, m, n));
                    }
                }
            }
        }
    }

    template <bool encode_dropout_in_sign_bit = false, int AtomLayoutMS = 2, int AtomLayoutNS = 2, typename Engine,
              typename Layout>
    __forceinline__ __device__ void mc_apply_dropout(Tensor<Engine, Layout> &tensor, int block_row_start,
                                                     int block_col_start) {
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) { return keep ? val : (encode_dropout_in_sign_bit ? -val : T(0)); };
        static_assert(decltype(size<0>(tensor))::value == 4);
        const int wave_col = threadIdx.x / 64 / AtomLayoutMS;

        if constexpr (AtomLayoutNS == 1) {
#pragma unroll
            for (int m = 0; m < size<1>(tensor); ++m, block_row_start += AtomLayoutMS) {
                uint2 rowcol = make_uint2(block_row_start, block_col_start);
                uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long &>(rowcol), offset);
                uint8_t(&rnd_8)[16] = reinterpret_cast<uint8_t(&)[16]>(random_uint4);
#pragma unroll
                for (int n = 0; n < size<2>(tensor); ++n) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        tensor(i, m, n) = encode_dropout(rnd_8[n * 4 + i] <= p_dropout_in_uint8_t, tensor(i, m, n));
                    }
                }
            }
        } else if constexpr (AtomLayoutNS == 2) {
#pragma unroll
            for (int m = 0; m < size<1>(tensor); ++m, block_row_start += AtomLayoutMS) {
                uint2 rowcol = make_uint2(block_row_start, block_col_start);
                uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long &>(rowcol), offset);
                uint8_t(&rnd_8)[16] = reinterpret_cast<uint8_t(&)[16]>(random_uint4);
#pragma unroll
                for (int n = 0; n < size<2>(tensor); ++n) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (wave_col == 0) {
                            tensor(i, m, n) = encode_dropout(rnd_8[n * 8 + i] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        } else {
                            tensor(i, m, n) =
                                encode_dropout(rnd_8[n * 8 + i + 4] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        }
                    }
                }
            }
        } else if constexpr (AtomLayoutNS == 4) {
#pragma unroll
            for (int m = 0; m < size<1>(tensor); ++m, block_row_start += AtomLayoutMS) {
#pragma unroll
                for (int n = 0; n < size<2>(tensor); ++n, block_col_start += 1) {
                    uint2 rowcol = make_uint2(block_row_start, block_col_start);
                    uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long &>(rowcol), offset);
                    uint8_t(&rnd_8)[16] = reinterpret_cast<uint8_t(&)[16]>(random_uint4);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (wave_col == 0) {
                            tensor(i, m, n) = encode_dropout(rnd_8[i] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        } else if (wave_col == 1) {
                            tensor(i, m, n) = encode_dropout(rnd_8[i + 4] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        } else if (wave_col == 2) {
                            tensor(i, m, n) = encode_dropout(rnd_8[i + 8] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        } else {
                            tensor(i, m, n) = encode_dropout(rnd_8[i + 12] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        }
                    }
                }
            }
        }
    }
};

}  // namespace flash
