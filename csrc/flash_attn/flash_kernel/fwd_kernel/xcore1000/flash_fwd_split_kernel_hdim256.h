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

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Is_page_attn, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_opt_hdim256(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

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
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, params.d, binfo.actual_seqlen_q - kBlockM_stride
        );

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
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Is_page_attn
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride; // block addresses are later resolved per-thread

    const index_t row_offset_v = !Is_page_attn
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

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

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyQKVPaged gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    Tensor tKgK_ = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK_ = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV_ = gmem_thr_copy_KV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV_ = gmem_thr_copy_KV.partition_D(sV);

    Tensor tKgK = make_tensor(tKgK_.data(), reshape_thread_tile(tKgK_.layout()));
    Tensor tKsK = make_tensor(tKsK_.data(), reshape_thread_tile(tKsK_.layout()));
    Tensor tVgV = make_tensor(tVgV_.data(), reshape_thread_tile(tVgV_.layout()));
    Tensor tVsV = make_tensor(tVsV_.data(), reshape_thread_tile(tVsV_.layout()));

    if constexpr (Is_page_attn) {
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max, params.page_block_size,
            block_table, params.k_batch_stride, params.k_row_stride);
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride);
    }

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

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV_ = gmem_thr_copy_KV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tKVcKV = make_tensor(tKVcKV_.data(), reshape_thread_tile(tKVcKV_.layout()));

    // Prologue

    // Copy from Knew to K, optionally apply rotary embedding.
    if constexpr (Append_KV) {
        typename Kernel_traits::GmemTiledCopyRotcossinPaged gmem_tiled_copy_rotary;
        auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
        typename Kernel_traits::GmemTiledCopyRotcossinContPaged gmem_tiled_copy_rotary_cont;
        auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);

        // Even if we have MQA / GQA, all threadblocks responsible for the same KV head are writing to
        // gmem. Technically it's a race condition, but they all write the same content anyway, and it's safe.
        // We want to do this so that all threadblocks can proceed right after they finish writing the KV cache.
        const index_t row_offset_cossin = ((n_block_max - 1) * kBlockN + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])) * (params.rotary_dim / 2);
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

        Tensor tRgCos_ = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin_ = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont_ = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont_ = gmem_thr_copy_rotary_cont.partition_S(gSinCont);

        Tensor tRgCos = make_tensor(tRgCos_.data(), reshape_thread_tile(tRgCos_.layout()));
        Tensor tRgSin = make_tensor(tRgSin_.data(), reshape_thread_tile(tRgSin_.layout()));
        Tensor tRgCosCont = make_tensor(tRgCosCont_.data(), reshape_flatten_thread_tile(tRgCosCont_.layout()));
        Tensor tRgSinCont = make_tensor(tRgSinCont_.data(), reshape_flatten_thread_tile(tRgSinCont_.layout()));

        // if (cute::thread(0, 0)) { printf("rotary_cos_ptr = %p, gCos.data() = %p, tRgCos.data() = %p, rotary_dim = %d\n", params.rotary_cos_ptr, gCos.data(), tRgCos.data(), params.rotary_dim); }
        // if (cute::thread(8, 0)) { print_tensor(gCos); }
        // if (cute::thread(0, 0)) { print_tensor(tRgCos); }

        // const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, bidb)
        const index_t row_offset_knew = bidb * params.knew_batch_stride
            + ((n_block_max - 1) * kBlockN) * params.knew_row_stride + (bidh / params.h_h_k_ratio) * params.knew_head_stride;
        // const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, bidb)
        const index_t row_offset_vnew = bidb * params.vnew_batch_stride
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
        typename Kernel_traits::GmemTiledCopyQKVPaged gmem_tiled_copy_KV_new;
        auto gmem_thr_copy_KV_new = gmem_tiled_copy_KV_new.get_thread_slice(tidx);
        Tensor tKgKnew_ = gmem_thr_copy_KV_new.partition_S(gKnew);  // (KCPY, KCPY_N, KCPY_K)
        Tensor tVgVnew_ = gmem_thr_copy_KV_new.partition_S(gVnew);  // (VCPY, VCPY_N, VCPY_K)

        auto tKgKnew = make_tensor(tKgKnew_.data(), reshape_thread_tile(tKgKnew_.layout()));
        auto tVgVnew = make_tensor(tVgVnew_.data(), reshape_thread_tile(tVgVnew_.layout()));

        const int n_block_copy_min = std::max(n_block_min, binfo.seqlen_k_cache / kBlockN);
        auto tKgK_data = tKgK.data();
        auto tVgV_data = tVgV.data();
        for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
            flash::copy_w_min_idx<Is_even_K>(
                tVgVnew, tVgV, tKVcKV, d, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
            );
            tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
            if (params.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K>(
                    tKgKnew, tKgK, tKVcKV, d, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
                );
            } else {
                if constexpr(kBlockM == kBlockN && kBlockN == 64){
                    if (params.is_rotary_interleaved) {
                        // Don't clear OOB_K because we're writing to global memory
                        flash::copy_rotary_interleaved<Is_even_K, /*Clear_OOB_K=*/false>(
                            tKgKnew, tKgK, tRgCos, tRgSin, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                            binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim);
                        tRgCos.data() = tRgCos.data() + (-int(kBlockN * params.rotary_dim / 2));
                        tRgSin.data() = tRgSin.data() + (-int(kBlockN * params.rotary_dim / 2));
                    } else {
                        // Don't clear OOB_K because we're writing to global memory
                        flash::copy_rotary_contiguous<Is_even_K, /*Clear_OOB_K=*/false>(
                            tKgKnew, tKgK, tRgCosCont, tRgSinCont, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                            binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim);
                        tRgCosCont.data() = tRgCosCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                        tRgSinCont.data() = tRgSinCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                    }
                }
            }
            tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
            if constexpr(!Is_page_attn) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                if (n_block > n_block_copy_min) {
                    tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                        block_table, params.v_batch_stride, params.v_row_stride);
                    tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                        block_table, params.k_batch_stride, params.k_row_stride);
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
        typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
        auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
        typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
        auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
        const index_t row_offset_cossin = (binfo.seqlen_k_cache + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb]) + (Is_causal || Is_local ? m_block * kBlockM : 0)) * (params.rotary_dim / 2);
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
        if constexpr(kBlockM == kBlockN && kBlockN == 64){
            if (params.is_rotary_interleaved) {
                flash::copy_rotary_interleaved<Is_even_K>(tQgQ, tQsQ, tRgCos, tRgSin, tQcQ,
                                                          binfo.actual_seqlen_q - m_block * kBlockM, 0, params.d,
                                                          params.rotary_dim);
            } else {
                flash::copy_rotary_contiguous<Is_even_K>(tQgQ, tQsQ, tRgCosCont, tRgSinCont, tQcQ,
                                                         binfo.actual_seqlen_q - m_block * kBlockM, 0, params.d,
                                                         params.rotary_dim);
            }
        }
    }

    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy_global_to_reg<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, d,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    const uint32_t laneId = __lane_id();
    const int cpy_offset = ((laneId & 0xf) << 2) - (laneId & 0xf);
    const uint32_t tOsVt_stride = get<1>(get<1>(tOsVt(_, _, _0{}).layout().stride()));
    const uint32_t tOrVt_stride = get<1>(get<1>(tOrVt(_, _, _0{}).layout().stride()));

    constexpr bool Is_B128 = (kBlockN == 64);
    constexpr int ldg_Num = (kBlockN * kHeadDim / Kernel_traits::kNThreads) / (Is_B128 ? 8 : 4);
    //Every lds_Tuple load 16 elements
    constexpr int lds_Tuple = 4;
    constexpr bool Is_perm_4x4 = true;

    int tVgV_offset[ldg_Num];
    uint32_t *tVsV_ptr[ldg_Num];
    int tVcV[ldg_Num + (!Is_even_K * ldg_Num)]; //pred for V
    #pragma unroll
    for (int i = 0; i < ldg_Num; ++i) {
        /***********************************************************************
         * gv_row means laneId[0~31] read 0~7 rows,laneId[32~63] read 32~39 rows
         * gv_col means reading 128 elements with 16 threads for kBlockN=64
         * For kBlockN=64, ldg_num=8:
         * gv_rows means laneId[0~31] read 0~7 rows,
         * laneId[32~63] read 32~39 rows
         * gv_col means [0~63] read 0~15th 8cols,
         * and for lastest 4-ldg_num, gv_col need to increase 16 8cols
         * For kblock=32, ldg_num=8 with ldg_b64/sts_b64
         * gv_rows means laneId[0~63] read 0~7 rows,
         * gv cols means [0~31] read 0~15th 4cols,
         * and [32~63] read 16~31th 4cols
         ***********************************************************************/
        const int gv_row = (((laneId >> 4) & 0x1) << 2) + (i & 0x3);
        const int gv_col = ((laneId & 0xf) << 2) + ((i >> 2) << 7) + ((laneId >> 5) << 6);
        const int old_gv_row = laneId >> 3;
        const int old_gv_col = (laneId & 0x7) << 3;
        tVgV_offset[i] = (gv_row - old_gv_row) * params.v_row_stride + (gv_col - old_gv_col);

        const int sv_row = gv_row + ((laneId >> 5) << 5) + ((i >> 2) << 6) + ((tidx >> 6) << 3);
        const int sv_col = (laneId & 0xf) << 2;
        tVsV_ptr[i] = reinterpret_cast<uint32_t *>(sV.data().ptr_ + sv_row * 64 + sv_col);
        tVcV[i] = gv_row + ((tidx >> 6) << 3);
        if (!Is_even_K) tVcV[ldg_Num + i] = gv_col;
    }

    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
    flash::sync_threads();
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);

    clear(acc_o);

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
        flash::sync_threads();

        // Advance gV
        if (masking_step > 0) {
            if constexpr(!Is_page_attn) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            } else {
                tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block + 1, params.page_block_size,
                    block_table, params.v_batch_stride, params.v_row_stride);
            }
            flash::copy_global_to_reg_V</*Is_even_MN=*/true, Is_even_K, /*Is_ldg_B128=*/Is_B128, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);
        } else {
            // Clear the smem tiles to account for predicated off loads
            flash::copy_global_to_reg_V<Is_even_MN, Is_even_K,/*Is_ldg_B128=*/Is_B128, ldg_Num>(
                tVgV, tVrV, tVcV, tVgV_offset, d, binfo.actual_seqlen_k - n_block * kBlockN);
        }

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, mask_offset, kWarps_offset, params.custom_alibi);

        flash::copy_reg_to_share_V</*Is_sts_B128=*/Is_B128, ldg_Num>(tVrV, tVsV_ptr, perm_mask);

        if (n_block > n_block_min) {
            // Advance gK
            if constexpr(!Is_page_attn) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                    block_table, params.k_batch_stride, params.k_row_stride);
            }
            flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
        }

        // We have key_padding_mask so we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || !Is_even_MN, true, true>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || !Is_even_MN, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        flash::gemm_rs<Is_perm_4x4, lds_Tuple>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        flash::copy_reg_to_share(tKrK,tKsK);
        clear(acc_s);
        flash::sync_threads();
        // Advance gV
        if constexpr(!Is_page_attn) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } else {
            tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block + 1, params.page_block_size,
                block_table, params.v_batch_stride, params.v_row_stride);
        }
        flash::copy_global_to_reg_V</*Is_even_MN=*/true, Is_even_K,/*Is_ldg_B128=*/Is_B128, ldg_Num>(tVgV, tVrV, tVcV, tVgV_offset, d);

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, mask_offset, kWarps_offset, params.custom_alibi);

        flash::copy_reg_to_share_V</*Is_sts_B128=*/Is_B128, ldg_Num>(tVrV, tVsV_ptr,perm_mask);

        if (n_block > n_block_min) {
            // Advance gK
            if constexpr(!Is_page_attn) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                    block_table, params.k_batch_stride, params.k_row_stride);
            }
            flash::copy_global_to_reg</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKVcKV, d);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        flash::gemm_rs<Is_perm_4x4, lds_Tuple>(acc_o, rP, tOrVt, tOsVt, tiled_mma, cpy_offset, tOsVt_stride, tOrVt_stride);
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
        int old_sm_col = ((((laneId & 0x7) ^ (laneId >> 5)) << 1) + ((laneId >> 4) & 0x1)) << 2;
        auto s_ptr = taccOsOaccum.data().ptr_ - old_sm_col;
        #pragma unroll 4
        for(int i = 0; i < 4; ++i) {
            auto ptr = reinterpret_cast<uint32_t *>(rO.data().ptr_ + 16 * i);
            auto temp_a = __builtin_mxc_byte_perm(ptr[2], ptr[0], perm_mask[0]);
            ptr[2] = __builtin_mxc_byte_perm(ptr[2], ptr[0], perm_mask[1]);
            ptr[0] = temp_a;
            temp_a = __builtin_mxc_byte_perm(ptr[3], ptr[1], perm_mask[0]);
            auto temp_b = __builtin_mxc_byte_perm(ptr[3], ptr[1], perm_mask[1]);
            ptr[1] = __builtin_mxc_byte_perm(ptr[6], ptr[4], perm_mask[0]);
            ptr[3] = __builtin_mxc_byte_perm(ptr[6], ptr[4], perm_mask[1]);

            ptr[4] = temp_a;
            ptr[6] = temp_b;
            temp_a = __builtin_mxc_byte_perm(ptr[7], ptr[5], perm_mask[0]);
            ptr[7] = __builtin_mxc_byte_perm(ptr[7], ptr[5], perm_mask[1]);
            ptr[5] = temp_a;
        }

        #pragma unroll 4
        for(int i = 0; i < 4; ++i) {
            int new_sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (i >> 1))) << 1) + (i & 0x1)) << 2;
            #pragma unroll 4
            for (int j = 0; j < 4; ++j) {
                auto ptr = reinterpret_cast<uint64_t *>(rO.data().ptr_ + j * 16);
                auto sm_ptr = reinterpret_cast<uint64_t *>(s_ptr + new_sm_col) + 1024 * j;
                sm_ptr[0] = ptr[i];
            }
        }

        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                            + kBlockM_stride) * params.d_rounded;
        const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
                ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
            ) + kBlockM_stride;

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.o_row_stride, _1{}));
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
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, d, binfo.actual_seqlen_q - kBlockM_stride
        );
    }
    else {

        int old_sm_col = ((((laneId & 0x7) ^ (laneId >> 5)) << 1) + ((laneId >> 4) & 0x1)) << 2;
        auto s_ptr = taccOsOaccum.data().ptr_ - old_sm_col;
        #pragma unroll 2
        for(int j = 0; j < 2; ++j) {
            auto ptr = acc_o.data() + j * 16;
            #pragma unroll 4
            for (int i = 0; i < 4; ++i) {
                int new_sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (i >> 1))) << 1) + (i & 0x1)) << 2;
                auto sm_ptr = s_ptr + new_sm_col + 4096 * j;
                sm_ptr[0] = ptr[i];
                sm_ptr[1] = ptr[i + 4];
                sm_ptr[2] = ptr[i + 8];
                sm_ptr[3] = ptr[i + 12];
            }
        }

        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + kBlockM_stride * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                            + kBlockM_stride) * params.d_rounded;
        const index_t row_offset_lseaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q) + kBlockM_stride;

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.oaccum_ptr) + (row_offset_oaccum)),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(kHeadDim, _1{}));
        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

        flash::sync_threads();

        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        auto sm_ptr = reinterpret_cast<uint32_t *>(sOaccum.data().ptr_);
        auto r_ptr = reinterpret_cast<uint32_t *>(tOrOaccum.data());
        const int wave_id = tidx >> 6;
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            const int s_row = (laneId >> 4) + (i << 4) + (wave_id << 2);
            const int s_col = ((((laneId & 0xf) ^ (s_row & 0x3) << 1) << 2) + ((wave_id & 0x1) << 5)) & 0x3f;
            auto src_ptr = reinterpret_cast<uint128_t *>(sm_ptr + s_row * 64 + s_col);
            auto dst_ptr = reinterpret_cast<uint128_t *>(r_ptr + 4 * i);
            dst_ptr[0] = src_ptr[0];
        }

        flash::sync_threads();

        #pragma unroll 2
        for(int j = 0; j < 2; ++j) {
            auto ptr = acc_o.data() + (j + 2) * 16;
            #pragma unroll 4
            for (int i = 0; i < 4; ++i) {
                int new_sm_col = ((((laneId & 0x7) ^ (((laneId >> 4) << 1) + (i >> 1))) << 1) + (i & 0x1)) << 2;
                auto sm_ptr = s_ptr + new_sm_col + 4096 * j;
                sm_ptr[0] = ptr[i];
                sm_ptr[1] = ptr[i + 4];
                sm_ptr[2] = ptr[i + 8];
                sm_ptr[3] = ptr[i + 12];
            }
        }

        flash::sync_threads();
        r_ptr = reinterpret_cast<uint32_t *>(tOrOaccum.data()) + 32;
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            const int s_row = (laneId >> 4) + (i << 4) + (wave_id << 2);
            const int s_col = ((((laneId & 0xf) ^ (s_row & 0x3) << 1) << 2) + ((wave_id & 0x1) << 5)) & 0x3f;
            auto src_ptr = reinterpret_cast<uint128_t *>(sm_ptr + s_row * 64 + s_col);
            auto dst_ptr = reinterpret_cast<uint128_t *>(r_ptr + 4 * i);
            dst_ptr[0] = src_ptr[0];
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
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, d, binfo.actual_seqlen_q - kBlockM_stride
        );

    }

}

} // namespace xcore1000

} // namespace flash
