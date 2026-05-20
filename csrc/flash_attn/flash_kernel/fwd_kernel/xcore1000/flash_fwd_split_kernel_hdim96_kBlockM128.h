#pragma once

#include <cute/algorithm/copy.hpp>

#include <mctlass/mctlass.h>
#include <mctlass/array.h>
#include <mctlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"
#include "attn_mask.h"


namespace flash {

namespace xcore1000 {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_opt_hdim96_kblockM128(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];
    uint32_t tQrQ[int(Kernel_traits::kRegSize)];
    uint32_t tKrK[int(Kernel_traits::kRegSize / 2)];
    uint32_t tVrV[int(Kernel_traits::kRegSize / 2)];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    static_assert(Kernel_traits::Is_Q_in_regs == true);
    static_assert(Kernel_traits::Share_Q_K_smem == true);

    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    const int kBlockM_stride = m_block * kBlockM;
    if (kBlockM_stride >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((binfo.actual_seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = !Is_local
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (kBlockM_stride + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div(kBlockM_stride + kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
        // Otherwise we might read OOB elements from gK and gV,
        // or get wrong results when we combine gOaccum from different blocks.
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + kBlockM_stride) * params.d_rounded;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + kBlockM_stride;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                     make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy_zero_to_global<Is_even_MN, Is_even_K>(
            tOgOaccum, tOcO, params.d, binfo.actual_seqlen_q - kBlockM_stride);

        auto gLSE = reinterpret_cast<int32_t *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum;
        auto inf = Split ? -INFINITY : INFINITY;
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            __builtin_mxc_stg_b32_predicator(gLSE + row, 0, *reinterpret_cast<int32_t *>(&inf), true, false, false,
                                    get<1>(tOcO(0, m, 0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
        }
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + kBlockM_stride * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
    const index_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : block_table[block_table_idx] * params.k_batch_stride + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = block_table == nullptr
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : block_table[block_table_idx] * params.v_batch_stride + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    //Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    const auto d = params.d;
    int n_block = n_block_max - 1;
    // PREDICATES
    //
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Prologue

    // Copy from Knew to K, optionally apply rotary embedding.
    typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
    auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
    auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
    if constexpr (Append_KV) {
        // Even if we have MQA / GQA, all threadblocks responsible for the same KV head are writing to
        // gmem. Technically it's a race condition, but they all write the same content anyway, and it's safe.
        // We want to do this so that all threadblocks can proceed right after they finish writing the KV cache.
        const index_t row_offset_cossin = ((n_block_max - 1) * kBlockN) * (params.rotary_dim / 2);
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
                                  make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
                                  make_stride(params.rotary_dim / 2, _1{}));
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                      make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                      make_stride(params.rotary_dim / 2, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);

        const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, bidb)
            + ((n_block_max - 1) * kBlockN) * params.knew_row_stride + (bidh / params.h_h_k_ratio) * params.knew_head_stride;
        const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, bidb)
            + ((n_block_max - 1) * kBlockN) * params.vnew_row_stride + (bidh / params.h_h_k_ratio) * params.vnew_head_stride;
        // Subtract seqlen_k_cache * row stride so that conceptually gK and gKnew "line up". When we access them,
        // e.g. if gK has 128 rows and gKnew has 64 rows, we access gK[:128] and gKNew[128:128 + 64].
        // This maps to accessing the first 64 rows of knew_ptr.
        Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.knew_ptr)
                                                + row_offset_knew - binfo.seqlen_k_cache * params.knew_row_stride),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  make_stride(params.knew_row_stride, _1{}));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("knew_ptr = %p, row_offset_knew = %d, gKnew_ptr = %p\n", params.knew_ptr, row_offset_knew, gKnew.data()); }
        Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.vnew_ptr)
                                                + row_offset_vnew - binfo.seqlen_k_cache * params.vnew_row_stride),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  make_stride(params.vnew_row_stride, _1{}));
        Tensor tKgKnew = gmem_thr_copy_QKV.partition_S(gKnew);  // (KCPY, KCPY_N, KCPY_K)
        Tensor tVgVnew = gmem_thr_copy_QKV.partition_S(gVnew);  // (VCPY, VCPY_N, VCPY_K)

        const int n_block_copy_min = std::max(n_block_min, binfo.seqlen_k_cache / kBlockN);
        auto tKgK_data = tKgK.data();
        auto tVgV_data = tVgV.data();
        if (params.rotary_dim == 0) {
            for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
                flash::copy_w_min_idx<Is_even_K>(
                    tVgVnew, tVgV, tKVcKV, d, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
                );
                tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
                flash::copy_w_min_idx<Is_even_K>(
                    tKgKnew, tKgK, tKVcKV, d, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
                );
                tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
                if (block_table == nullptr) {
                    tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                    tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
                } else {
                    if (n_block > n_block_copy_min) {
                        const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                        const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                        const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                        const int block_table_offset_next = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                        const int table_diff = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
                        const int offset_diff = block_table_offset_next - block_table_offset_cur;
                        tVgV.data() = tVgV.data() + table_diff * params.v_batch_stride + offset_diff * params.v_row_stride;
                        tKgK.data() = tKgK.data() + table_diff * params.k_batch_stride + offset_diff * params.k_row_stride;
                    }
                }
            }
        }
        else {
            for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
                flash::copy_w_min_idx<Is_even_K>(
                    tVgVnew, tVgV, tKVcKV, d, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
                );
                tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
                if (params.is_rotary_interleaved) {
                    // Don't clear OOB_K because we're writing to global memory
                    flash::copy_rotary_interleaved<Is_even_K, /*Clear_OOB_K=*/false>(
                        tKgKnew, tKgK, tRgCos, tRgSin, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim
                    );
                    tRgCos.data() = tRgCos.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSin.data() = tRgSin.data() + (-int(kBlockN * params.rotary_dim / 2));
                } else {
                    // Don't clear OOB_K because we're writing to global memory
                    flash::copy_rotary_contiguous<Is_even_K, /*Clear_OOB_K=*/false>(
                        tKgKnew, tKgK, tRgCosCont, tRgSinCont, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim
                    );
                    tRgCosCont.data() = tRgCosCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSinCont.data() = tRgSinCont.data() + (-int(kBlockN * params.rotary_dim / 2));

                }
                tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
                if (block_table == nullptr) {
                    tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                    tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
                } else {
                    if (n_block > n_block_copy_min) {
                        const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                        const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                        const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                        const int block_table_offset_next = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                        const int table_diff = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
                        const int offset_diff = block_table_offset_next - block_table_offset_cur;
                        tVgV.data() = tVgV.data() + table_diff * params.v_batch_stride + offset_diff * params.v_row_stride;
                        tKgK.data() = tKgK.data() + table_diff * params.k_batch_stride + offset_diff * params.k_row_stride;
                    }
                }
            }
        }
        // Need this before we can read in K again, so that we'll see the updated K values.
        flash::barrier_gvm<0>();
        tKgK.data() = tKgK_data;
        tVgV.data() = tVgV_data;
    }

    // Read Q from gmem to smem, optionally apply rotary embedding.
    if (!Append_KV || params.rotary_dim == 0) {
        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, d, binfo.actual_seqlen_q - kBlockM_stride);
        flash::copy_reg_to_share(tQrQ, tQsQ);
    } else {
        const index_t row_offset_cossin = (binfo.seqlen_k_cache + (Is_causal || Is_local ? kBlockM_stride : 0)) * (params.rotary_dim / 2);
        // If not causal, all the queries get the same the cos/sin, taken at location seqlen_k_cache.
        // We do this by setting the row stride of gCos / gSin to 0.
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        if (params.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K>(
                tQgQ, tQsQ, tRgCos, tRgSin, tQcQ, binfo.actual_seqlen_q - kBlockM_stride,
                0, d, params.rotary_dim
            );
        } else {
            flash::copy_rotary_contiguous<Is_even_K>(
                tQgQ, tQsQ, tRgCosCont, tRgSinCont, tQcQ, binfo.actual_seqlen_q - kBlockM_stride,
                0, d, params.rotary_dim
            );
        }
    }

    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, d,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    const uint32_t laneId = __lane_id();
    const int32_t cpy_offset = ((laneId & 0xf) << 1) - (laneId & 0xf);
    const uint32_t tOsVt_stride = 2048;
    const uint32_t tOrVt_stride = 32;

    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
    flash::sync_threads();
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);

    clear(acc_o);
    // Clear the smem tiles to account for predicated off loads
    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tVgV, tVrV, tKVcKV, d,
                                binfo.actual_seqlen_k - n_block * kBlockN);
    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    flash::sync_threads();

    const auto mask_offset = kBlockM_stride + (tidx / 64) * 16 + (tidx & 0xf);
    const auto kWarps_offset = kNWarps * 16;
    const uint32_t perm_mask[2] = {0x05040100, 0x07060302};

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        flash::copy_reg_to_share(tKrK, tKsK);
        clear(acc_s);

        // Advance gV
        if (masking_step > 0) {
            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            } else {
                const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
                const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
                tVgV.data() = tVgV.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
            }
            //flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, d);
            flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tKVcKV, d);
        }
        flash::sync_threads();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

       mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, mask_offset, kWarps_offset, params.custom_alibi);

        flash::copy_reg_to_share(tVrV, tVsV);

        if (n_block > n_block_min) {
            // Advance gK
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                const int block_table_offset_next =(n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                tKgK.data() = tKgK.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
            }
            flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
        }

        // We have key_padding_mask so we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || !Is_even_MN, true, true>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || !Is_even_MN, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        flash::gemm_rs_hdim96(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        flash::copy_reg_to_share(tKrK, tKsK);
        clear(acc_s);
        // Advance gV
        if (block_table == nullptr) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } else {
            const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
            const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
            const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
            const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
            tVgV.data() = tVgV.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
        }
        flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tKVcKV, d);
        flash::sync_threads();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, mask_offset, kWarps_offset, params.custom_alibi);

        flash::copy_reg_to_share(tVrV, tVsV);

        if (n_block > n_block_min) {
            // Advance gK
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                const int block_table_offset_next = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                tKgK.data() = tKgK.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
            }
            flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        //Tensor rP = flash::convert_type<Element>(acc_s);
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        flash::gemm_rs_hdim96(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);

    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    flash::barrier();

    if constexpr(!Split) {
        CONVERT_TENSOR_TYPE(ElementAccum, ElementO, acc_o, rO)
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            auto ptr = reinterpret_cast<uint32_t *>(rO.data().ptr_ + ((i >> 1) << 4) + ((i & 0x1) << 2));
            auto a = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[0]);
            ptr[4] = __builtin_mxc_byte_perm(ptr[4], ptr[0], perm_mask[1]);
            ptr[0] = a;
            auto b = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[0]);
            ptr[5] = __builtin_mxc_byte_perm(ptr[5], ptr[1], perm_mask[1]);
            ptr[1] = b;
            flash::swap(ptr[1], ptr[4]);
        }

        uint32_t sm_col = (((((laneId & 0x7) >> 1) << 1) ^ ((laneId >> 5) << 1)) + ((laneId >> 4) & 0x1)) << 2;
        auto s_ptr = taccOsOaccum.data().ptr_ - sm_col;
        #pragma unroll
        for (uint32_t i = 0; i < 6; ++i) {
            auto ptr = reinterpret_cast<uint64_t *>(rO.data().ptr_ + ((i >> 1) << 4) + ((i & 0x1) << 2));
            uint32_t col = ((laneId >> 4) ^ (((laneId & 0xf) >> 1) & 0x3)) << 3;
            auto sm_ptr = reinterpret_cast<uint64_t *>(s_ptr + col + (i << 11));
            sm_ptr[0] = ptr[0];
            sm_ptr[1] = ptr[2];
        }
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
        const index_t row_offset_oaccum = row_offset_lseaccum * params.d_rounded;

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

        flash::sync_threads();

        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        auto gLSE_ptr = reinterpret_cast<int32_t *>(params.softmax_lse_ptr) + row_offset_lseaccum;
        auto lse_ptr = reinterpret_cast<int32_t *>(lse.data());
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            __builtin_mxc_stg_b32_predicator(gLSE_ptr + row, 0, lse_ptr[mi], true, false, false,
                get<1>(taccOcO_row(0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        flash::copy_reg_to_global<Is_even_MN, Is_even_K>(
            tOrOaccum, tOgOaccum, tOcO, d, binfo.actual_seqlen_q - kBlockM_stride);
    }
    else {
        uint32_t sm_col = (((((laneId & 0x7) >> 1) << 1) ^ ((laneId >> 5) << 1)) + ((laneId >> 4) & 0x1)) << 2;
        auto taccOsOaccum_ptr = taccOsOaccum.data().ptr_ - sm_col;
        #pragma unroll
        for (uint32_t i = 0; i < 3; ++i) {
            auto acc_o_ptr = reinterpret_cast<uint32_t *>(acc_o.data() + ((i >> 1) << 4) + ((i & 0x1) << 2));
            uint32_t col = ((laneId >> 4) ^ (((laneId & 0xf) >> 1) & 0x3)) << 3;
            auto sm_ptr = reinterpret_cast<uint32_t *>(taccOsOaccum_ptr + col + (i << 11));
            sm_ptr[0] = acc_o_ptr[0];
            sm_ptr[1] = acc_o_ptr[8];
            sm_ptr[2] = acc_o_ptr[1];
            sm_ptr[3] = acc_o_ptr[9];
            sm_ptr[4] = acc_o_ptr[2];
            sm_ptr[5] = acc_o_ptr[10];
            sm_ptr[6] = acc_o_ptr[3];
            sm_ptr[7] = acc_o_ptr[11];
        }
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + kBlockM_stride;
        const index_t row_offset_oaccum = row_offset_lseaccum * params.d_rounded;

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.oaccum_ptr) + row_offset_oaccum),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(kHeadDim, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

        flash::sync_threads();

        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));

        int row = laneId >> 3;
        int col = ((laneId & 0x7) ^ ((laneId >> 4) << 1)) << 2;
        const auto wave_id = tidx >> 6;
        auto sOaccum_ptr = sOaccum.data().ptr_ + ((row + (wave_id << 3)) << 5) + col;
        auto tOrOaccum_ptr = tOrOaccum.data();
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            auto sptr = reinterpret_cast<uint128_t *>(sOaccum_ptr + (i << 10));
            auto rptr = reinterpret_cast<uint128_t *>(tOrOaccum_ptr + (i << 2));
            rptr[0] = sptr[0];
        }

        flash::sync_threads();

        #pragma unroll
        for (uint32_t i = 3; i < 6; ++i) {
            auto acc_o_ptr = reinterpret_cast<uint32_t *>(acc_o.data() + ((i >> 1) << 4) + ((i & 0x1) << 2));
            uint32_t col = ((laneId >> 4) ^ (((laneId & 0xf) >> 1) & 0x3)) << 3;
            auto sm_ptr = reinterpret_cast<uint32_t *>(taccOsOaccum_ptr + col + ((i - 3) << 11));
            sm_ptr[0] = acc_o_ptr[0];
            sm_ptr[1] = acc_o_ptr[8];
            sm_ptr[2] = acc_o_ptr[1];
            sm_ptr[3] = acc_o_ptr[9];
            sm_ptr[4] = acc_o_ptr[2];
            sm_ptr[5] = acc_o_ptr[10];
            sm_ptr[6] = acc_o_ptr[3];
            sm_ptr[7] = acc_o_ptr[11];
        }

        flash::sync_threads();

        tOrOaccum_ptr = tOrOaccum.data() + 24;
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            auto sptr = reinterpret_cast<uint128_t *>(sOaccum_ptr + (i << 10));
            auto rptr = reinterpret_cast<uint128_t *>(tOrOaccum_ptr + (i << 2));
            rptr[0] = sptr[0];
        }

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        auto gLSE_ptr = reinterpret_cast<int32_t *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum;
        auto lse_ptr = reinterpret_cast<int32_t *>(lse.data());

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            __builtin_mxc_stg_b32_predicator(gLSE_ptr + row, 0, lse_ptr[mi], true, false, false,
                get<1>(taccOcO_row(0)) == 0 && row < binfo.actual_seqlen_q - kBlockM_stride, 1, MACA_ICMP_EQ);
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        flash::copy_reg_to_global<Is_even_MN, Is_even_K>(
            tOrOaccum, tOgOaccum, tOcO, d, binfo.actual_seqlen_q - kBlockM_stride);
    }

}

} // namespace xcore1000

} // namespace flash
