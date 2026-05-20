/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/algorithm/copy.hpp"

#include "mctlass/mctlass.h"
#include "mctlass/layout/layout.h"
#include <mctlass/numeric_types.h>

using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=mctlass::half_t>
struct Flash_kernel_traits {

    using Element = elem_type;
    using ElementAccum = float;
    using index_t = uint64_t;

    static constexpr bool Has_cp_async = false;

    using MMA_Atom_16x16x16 = std::conditional_t<
        std::is_same_v<elem_type, mctlass::half_t>,
        MMA_Atom<MACA_16x16x16_F32F16F16F32>,
        MMA_Atom<MACA_16x16x16_F32BF16BF16F32>
    >;

    using MMA_Atom_16x16x32 = std::conditional_t<
        std::is_same_v<elem_type, mctlass::half_t>,
        MMA_Atom<MACA_16x16x32_F32F16F16F32>,
        MMA_Atom<MACA_16x16x32_F32BF16BF16F32>
    >;
    using MMA_Atom_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, mctlass::half_t>,
        MMA_Atom<MACA_16x64x16_F32F16F16F32>,
        MMA_Atom<MACA_16x64x16_F32BF16BF16F32>
    >;
    using ValLayoutMNK = Layout<Shape<_1, _1, _1>>;

    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
    using UniversalCopyAtomB32 = Copy_Atom<UniversalCopy<uint32_t>, elem_type>;
    using UniversalCopyAtomB64 = Copy_Atom<UniversalCopy<uint64_t>, elem_type>;
    using UniversalCopyAtomB128 = Copy_Atom<UniversalCopy<uint128_t>, elem_type>;
    using LDSB64Trans4x16Atom = Copy_Atom<Copy_Traits<MACA_LDS_TRANS_4X16>, elem_type>;

    // dequant kernel
    using UniversalCopyAtomQuantB32 = Copy_Atom<UniversalCopy<uint32_t>, int8_t>;
    using UniversalCopyAtomQuantB64 = Copy_Atom<UniversalCopy<uint64_t>, int8_t>;
    using UniversalCopyAtomQuantB128 = Copy_Atom<UniversalCopy<uint128_t>, int8_t>;

    using UniversalCopyAtomScaleB16 = Copy_Atom<UniversalCopy<uint16_t>, elem_type>;
    using UniversalCopyAtomScaleB32 = Copy_Atom<UniversalCopy<uint32_t>, elem_type>;
    using UniversalCopyAtomScaleB64 = Copy_Atom<UniversalCopy<uint64_t>, elem_type>;

};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=mctlass::half_t,
         int kHeadDimV_=kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using UniversalCopyAtomB32 = typename Base::UniversalCopyAtomB32;
    using UniversalCopyAtomB64 = typename Base::UniversalCopyAtomB64;
    using UniversalCopyAtomB128 = typename Base::UniversalCopyAtomB128;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;
    using LDSB64Trans4x16Atom = typename Base::LDSB64Trans4x16Atom;
    using ElementSink = mctlass::bfloat16_t;

    static constexpr bool Has_cp_async = Base::Has_cp_async;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kHeadDimV_tail = kHeadDimV % 64 == 0 ? 0 : 32;
    static constexpr int kHeadDimV_pre = kHeadDimV - kHeadDimV_tail;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKSmemV = kHeadDimV % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    static constexpr int MBase = 3;
    static constexpr int SShift = 3;
    static constexpr int SShift_OPT = kBlockKSmem == 32 ? 3 : 4;    // for bank conflict free
    static constexpr int LDSTRANSBSizzle = kBlockKSmem == 32 ? 1 : 2;
    // only C600 hdim128 && K32 support 2-stage
    static constexpr int Num_Stages = (kHeadDim == 128 || kBlockKSmem == 32) ? 2 : 1;
    static constexpr int kAtomLayoutMS = std::min(kBlockM / 16, kNWarps);
    static constexpr int kAtomLayoutMO = kAtomLayoutMS;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_16x16x16,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma_S = TiledMMA<
        typename Base::MMA_Atom_16x16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    // ****for large head_dim kernel****
    using TiledMmaS_k256 = TiledMMA<
        typename Base::MMA_Atom_16x16x16,
        Layout<Shape<Int<kAtomLayoutMS>,_1,_1>>,  // 2x1x1 or 4x1x1
        typename Base::ValLayoutMNK>;

    using TiledMmaO_k256 = TiledMMA<
        typename Base::MMA_Atom_16x16x16,
        Layout<Shape<Int<kAtomLayoutMO>,Int<kNWarps / kAtomLayoutMO>,_1>>,  // 2x2x1 or 4x2x1
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomRowMax = decltype(
        composition(Swizzle<0, 0, 0>{},
                    Layout<Shape<_1, Int<kBlockM>>,
                           Stride<Int<kBlockM>, _1>>{}));
    using SmemLayoutRowMax = decltype(tile_to_shape(
        SmemLayoutAtomRowMax{},
        Shape<Int<kNWarps / kAtomLayoutMS>, Int<kBlockM>>{})); //rowmax 16 value per wave

    using SmemLayoutAtomRowSum = decltype(
        composition(Swizzle<0, 0, 0>{},
                    Layout<Shape<_1, Int<kBlockM>>,
                           Stride<Int<kBlockM>, _1>>{}));
    using SmemLayoutRowSum = decltype(tile_to_shape(
        SmemLayoutAtomRowSum{},
        Shape<Int<kNWarps / kAtomLayoutMS>, Int<kBlockM>>{})); //rowsum 64 value per wave

    using SmemLayoutAtomP = decltype(
        composition(Swizzle<3, 2, 4>{},
                    Layout<Shape<Int<kBlockM>,Int<kBlockN>>,
                           Stride<Int<kBlockN>, _1>>{}));
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        Shape<Int<kBlockM>,Int<kBlockN>>{})); //rowmax_wg0
    // ********************************

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, MBase, SShift>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutK_k_stage = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<Num_Stages>>{}));

    using SmemLayoutAtomK = decltype(
        composition(Swizzle<kSwizzle, MBase, SShift_OPT>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutAtomK324 = decltype(
        composition(Swizzle<3, 2, 4>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutK324 = decltype(tile_to_shape(
        SmemLayoutAtomK324{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK424 = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutK424 = decltype(tile_to_shape(
        SmemLayoutAtomK424{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK242 = decltype(
        composition(Swizzle<2, 4, 2>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutK242 = decltype(tile_to_shape(
        SmemLayoutAtomK242{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomQKVNoSwizzle = Layout<Shape<_16, Int<kBlockKSmem>>,
                                        Stride<Int<kBlockKSmem>, _1>>;
    using SmemLayoutQNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomQKVNoSwizzle{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutKNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomQKVNoSwizzle{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomQKVNoSwizzle{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    using SmemLayoutKNoSwizzle_k_stage = decltype(tile_to_shape(
        SmemLayoutAtomQKVNoSwizzle{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<Num_Stages>>{}));

    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    // This has to be kBlockN and not 8, otherwise we get wrong results for d=128
    using SmemLayoutAtomVtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmemV>, Int<kBlockN>>,
                                                      Stride<_1, Int<kBlockKSmemV>>>;
    using SmemLayoutAtomVtransposed = decltype(
        composition(Swizzle<LDSTRANSBSizzle, 4, 2>{}, SmemLayoutAtomVtransposedNoSwizzle{}));
    using SmemLayoutVtransposed = decltype(tile_to_shape(
        SmemLayoutAtomVtransposed{},
        Shape<Int<kHeadDimV>, Int<kBlockN>>{}));

    // Maybe the VtransposeNoSwizzle just needs to have the right shape
    // And the strides don't matter?
    using SmemLayoutVtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomVtransposedNoSwizzle{},
        Shape<Int<kHeadDimV>, Int<kBlockN>>{}));

    using SmemLayoutVtransposedNoSwizzle_pre = decltype(tile_to_shape(
        Layout<Shape<Int<64>, Int<kBlockN>>,
                Stride<_1, Int<64>>>{},
        Shape<Int<kHeadDimV_pre>, Int<kBlockN>>{}));

    using SmemLayoutVtransposedNoSwizzle_tail = decltype(tile_to_shape(
        Layout<Shape<Int<32>, Int<kBlockN>>,
                Stride<_1, Int<32>>>{},
        Shape<Int<kHeadDimV_tail>, Int<kBlockN>>{}));

    using SmemLayoutVtNoSwizzle = decltype(tile_to_shape(
        Layout<Shape<_16, Int<kBlockKSmemV>>,
               Stride<Int<kBlockKSmemV>, _1>>{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));

    using SmemLayoutVtNoSwizzle_pre = decltype(tile_to_shape(
        Layout<Shape<_16, Int<64>>,
               Stride<Int<64>, _1>>{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV_pre>{})));

    using SmemLayoutVtNoSwizzle_tail = decltype(tile_to_shape(
        Layout<Shape<_16, Int<32>>,
               Stride<Int<32>, _1>>{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV_tail>{})));

    using SmemLayoutAtomV242 = decltype(
        composition(Swizzle<2, 4, 2>{}, Layout<Shape<_16, Int<kBlockKSmemV>>,
                                        Stride<Int<kBlockKSmemV>, _1>>{}));

    using SmemLayoutV242 = decltype(tile_to_shape(
        SmemLayoutAtomV242{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, MBase, SShift>{},
                    Layout<Shape<Int<16>, Int<kBlockKSmemV>>,
                           Stride<Int<kBlockKSmemV>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemLayoutONoSwizzle = decltype(tile_to_shape(
        Layout<Shape<_16, Int<kBlockKSmemV>>,
               Stride<Int<kBlockKSmemV>, _1>>{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<UniversalCopy<uint64_t>, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kBlockKSmemMask = kBlockN % 64 == 0 ? 64 : (kBlockN % 32 == 0 ? 32: 16);
    static constexpr int kSwizzleMask = kBlockKSmemMask == 64 ? 4 : (kBlockKSmemMask == 32 ? 3 : 2);
    using SmemLayoutAtomMask = decltype(
        composition(Swizzle<kSwizzleMask, 2, kSwizzleMask>{},
                    Layout<Shape<Int<kBlockM>, Int<kBlockKSmemMask>>,
                           Stride<Int<kBlockKSmemMask>, _1>>{}));
    using SmemLayoutMask = decltype(tile_to_shape(
        SmemLayoutAtomMask{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));

    static constexpr int kSmemMaskSize = size(SmemLayoutMask{}) * sizeof(Element);
    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutK{}) * sizeof(Element);
    static constexpr int kSmemKSize_k_stage = size(SmemLayoutK_k_stage{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutV{}) * sizeof(Element);
    static constexpr int kSmemKVSize = kSmemKSize + kSmemVSize;
    static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;
    static constexpr int kRegSize = kSmemSize / sizeof(uint32_t) / kNThreads;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemElemsPerLoadB64 = sizeof(cute::uint64_t) / sizeof(Element);

    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static_assert(kHeadDim % kGmemElemsPerLoadB64 == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadB64");
    static_assert(kHeadDimV % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");

    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static constexpr int kGmemThreadsPerRowB64 = kBlockKSmem / kGmemElemsPerLoadB64;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtomB128 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemLayoutAtomB64 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB64>, Int<kGmemThreadsPerRowB64>>,
                                  Stride<Int<kGmemThreadsPerRowB64>, _1>>;

    static constexpr int kGmemThreadsPerRowV = kBlockKSmemV / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRowV == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtomV = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowV>, Int<kGmemThreadsPerRowV>>,
                                   Stride<Int<kGmemThreadsPerRowV>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(UniversalCopyAtomB128{},
                        GmemLayoutAtomB128{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopyB128 = decltype(
        make_tiled_copy(UniversalCopyAtomB128{},
                        GmemLayoutAtomB128{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopyB64 = decltype(
        make_tiled_copy(UniversalCopyAtomB64{},
                        GmemLayoutAtomB64{},
                        Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per read


    // dequant kernel

    using GmemCopyAtomQuantB32 = typename Base::UniversalCopyAtomQuantB32;

    using UniversalCopyAtomQuantB64 = typename Base::UniversalCopyAtomQuantB64;
    using UniversalCopyAtomScaleB16 = typename Base::UniversalCopyAtomScaleB16;
    using UniversalCopyAtomQuantB128 = typename Base::UniversalCopyAtomQuantB128;
    using UniversalCopyAtomScaleB32 = typename Base::UniversalCopyAtomScaleB32;
    using UniversalCopyAtomScaleB64 = typename Base::UniversalCopyAtomScaleB64;

    static constexpr int kGmemElemsQuantPerThreadB64 = sizeof(cute::uint64_t) / sizeof(int8_t);     // 8
    static constexpr int kGmemElemsQuantPerThreadB128 = sizeof(cute::uint128_t) / sizeof(int8_t);   // 16

    static constexpr int kGmemElemsScaleGroup8PerThreadB32 = sizeof(cute::uint32_t) / sizeof(Element);  // 2

    static constexpr int kGmemElemsQuantPerLoadB64 = kHeadDim / kGmemElemsQuantPerThreadB64;
    static constexpr int kGmemElemsQuantPerLoadB128 = kHeadDim / kGmemElemsQuantPerThreadB128;

    static constexpr int kGmemElemsScaleGroup8PerLoadB32 = kHeadDim / 8 / kGmemElemsScaleGroup8PerThreadB32;

    using GmemLayoutAtomQuantB64 =
        Layout<Shape<Int<kNThreads / kGmemElemsQuantPerLoadB64>, Int<kGmemElemsQuantPerLoadB64>>,
               Stride<Int<kGmemElemsQuantPerLoadB64>, _1>>;
    using GmemLayoutAtomQuantB128 =
        Layout<Shape<Int<kNThreads / kGmemElemsQuantPerLoadB128>, Int<kGmemElemsQuantPerLoadB128>>,
               Stride<Int<kGmemElemsQuantPerLoadB128>, _1>>;

    using SmemLayoutAtomScaleB32 = GmemLayoutAtomQuantB128;  // 1 fp16 scale to 8 int8 cache

    using GmemLayoutAtomScaleB16 =
        Layout<Shape<Int<kNThreads / kGmemElemsQuantPerLoadB64>, Int<kGmemElemsQuantPerLoadB64>>,
               Stride<Int<kGmemElemsQuantPerLoadB64>, _1>>;

    using GmemLayoutAtomScaleB32 =
        Layout<Shape<Int<kNThreads / kGmemElemsScaleGroup8PerLoadB32>, Int<kGmemElemsScaleGroup8PerLoadB32>>,
               Stride<Int<kGmemElemsScaleGroup8PerLoadB32>, _1>>;
    using GmemLayoutAtomScaleB64 = Layout<Shape <Int<kNThreads / kGmemElemsPerLoadB64>, Int<kGmemElemsPerLoadB64>>,
                                  Stride<Int<kGmemElemsPerLoadB64>, _1>>;

    using GmemTiledCopyQuantedKV = decltype(make_tiled_copy(UniversalCopyAtomQuantB128{}, GmemLayoutAtomQuantB128{},
                                                            Layout<Shape<_1, _16>>{}));  // Val layout, 16 vals per read

    using GmemTiledCopyScaleKV = decltype(make_tiled_copy(UniversalCopyAtomScaleB32{}, GmemLayoutAtomScaleB32{},
                                                            Layout<Shape<_1, _2>>{}));  // Val layout, 2 vals per read

    using SmemTiledCopyScaleKV =
        decltype(make_tiled_copy(UniversalCopyAtomScaleB32{}, SmemLayoutAtomScaleB32{}, Layout<Shape<_1, _2>>{}));

    using GmemTiledCopy1x4 = decltype(
        make_tiled_copy(UniversalCopyAtomB64{},
                        GmemLayoutAtomB64{},
                        Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per read

    using GmemTiledCopyBsm = decltype(
        make_tiled_copy(Copy_Atom<MACA_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, elem_type>{},
                        GmemLayoutAtomB128{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    // used for k32 kernels
    using GmemTiledCopy_4x2 = decltype(
        make_tiled_copy(UniversalCopyAtomB32{},
                        Layout<Shape <Int<kNThreads / (kBlockKSmem / 2)>, Int<kBlockKSmem / 2>>,
                               Stride<Int<kBlockKSmem / 2>, _1>>{},
                        Layout<Shape <_4, _2>,
                               Stride<_2, _1>>{}));

    using GmemTiledCopyMask = decltype(
        make_tiled_copy(UniversalCopyAtomB64{},
                        Layout<Shape <Int<kNThreads / (kBlockKSmemMask / 4)>, Int<kBlockKSmemMask / 4>>,
                                Stride<Int<kBlockKSmemMask / 4>, _1>>{},
                        Layout<Shape <_1, _4>>{}));

    // used for k64 kernels
    static constexpr bool UseWarpsNx1 = kBlockN >= 16 * kNWarps;
    using GmemLayoutAtomV_4x4 = std::conditional_t<
        UseWarpsNx1,
        Layout<Shape <Int<kNThreads / 16>, _16>,
               Stride<_16, _1>>,
        Layout<Shape<Int<kBlockN / 4>, Shape<_16, Int<kNThreads / 16 / (kBlockN / 4)>>>,
               Stride<_16, Stride<_1, Int<kBlockN * 4>>>>
    >;

    using GmemTiledCopy_4x4 = decltype(
        make_tiled_copy(UniversalCopyAtomB64{},
                        GmemLayoutAtomV_4x4{},
                        Layout<Shape <_4, _4>,
                               Stride<_4, _1>>{}));

    // dequant kernel
    using SmemLayoutVtQuantNoSwizzle = Layout<Shape<Int<kHeadDimV>, Int<kBlockN>>, Stride<_1, Int<kHeadDimV>>>;
    using SmemLayoutVQuantNoSwizzle = Layout<Shape<Int<kBlockN>, Int<kHeadDimV>>, Stride<Int<kHeadDimV>, _1>>;

    using GmemTiledCopyQuant_4x4 = decltype(
        make_tiled_copy(GmemCopyAtomQuantB32{},
                        GmemLayoutAtomV_4x4{},
                        Layout<Shape <_4, _4>,
                               Stride<_4, _1>>{}));


    // from how many rows does each thread have to fetch
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    static constexpr int kGmemRowsPerThreadB64 = kBlockN / (kNThreads / kGmemThreadsPerRowB64);
    // Here we assign a contiguous tile to each thread, rather than a 1x8 row every
    // (kNThreads / kGmemThreadsPerRow) rows, ensuring that the elements assigned to each thread
    // do not cross a page boundary. This way, each thread need only fetch 1 page index per
    // mainloop iteration. R>udimentary testing shows no slowdown.
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(UniversalCopyAtomB128{},
                        GmemLayoutAtomB128{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));

    using GmemTiledCopyVPaged_4x4 = decltype(
        make_tiled_copy(UniversalCopyAtomB64{},
                        GmemLayoutAtomV_4x4{},
                        Layout<Shape <Int<kGmemRowsPerThreadB64>, _4>,
                               Stride<_4, _1>>{}));


    using GmemTiledCopyQKVPagedQuant = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, int8_t>{},
                        GmemLayoutAtomB128{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));

    using GmemTiledCopyO = decltype(
        make_tiled_copy(UniversalCopyAtomB128{},
                        GmemLayoutAtomV{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <Int<kNThreads / 8>, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <Int<kNThreads / 16>, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtomB128;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(UniversalCopyAtomB128{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(UniversalCopyAtomB128{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load
};

// Is_V_in_regs is an option to reduce smem usage, but will increase register pressue.
// No_double_buffer is another option to reduce smem usage, but will slow things down.
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         int AtomLayoutMSdP_=1, int AtomLayoutMdKV_=2, int AtomLayoutMdQ_=2,
         bool Is_V_in_regs_=false, bool Is_K_in_regs_=false,
         bool No_double_buffer_=false, typename elem_type=mctlass::half_t, int kHeadDimV_ = kHeadDim_,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using UniversalCopyAtomB32 = typename Base::UniversalCopyAtomB32;
    using UniversalCopyAtomB64 = typename Base::UniversalCopyAtomB64;
    using UniversalCopyAtomB128 = typename Base::UniversalCopyAtomB128;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;
    using LDSB64Trans4x16Atom = typename Base::LDSB64Trans4x16Atom;

    static constexpr bool Has_cp_async = Base::Has_cp_async;

    static constexpr bool Is_V_in_regs = Is_V_in_regs_;
    static constexpr bool Is_K_in_regs = Is_K_in_regs_;
    static constexpr bool No_double_buffer = No_double_buffer_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 3 : 4;
    static constexpr int kSwizzle_b128 = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static constexpr int AtomLayoutMdQ = AtomLayoutMdQ_;
    static constexpr int AtomLayoutMdKV = AtomLayoutMdKV_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutMdKV == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);
    static constexpr int AtomLayoutNSdP = std::min(kNWarps / AtomLayoutMSdP, kBlockN / 16);

    using TiledMmaSdP = TiledMMA<
        typename Base::MMA_Atom_16x16x16,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<AtomLayoutNSdP>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmaSdP_b128 = TiledMMA<
        typename Base::MMA_Atom_16x16x32,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<AtomLayoutNSdP>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadKV = TiledMMA<
        typename Base::MMA_Atom_16x16x16,
        Layout<Shape<Int<AtomLayoutMdKV>, Int<kNWarps / AtomLayoutMdKV>, _1>>,
        typename Base::ValLayoutMNK>;

    // use lds4x4 + perm4x4 when load B
    using TiledMmadKV_16x64x16 = TiledMMA<
        typename Base::MMA_Atom_16x64x16,
        Layout<Shape<Int<AtomLayoutMdKV>, Int<kNWarps / AtomLayoutMdKV>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadQ = TiledMMA<
        typename Base::MMA_Atom_16x16x16,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQdO = decltype(
        composition(Swizzle<kSwizzle, 2, 4>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO = decltype(tile_to_shape(
        SmemLayoutAtomQdO{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutAtom_b128 = decltype(
        composition(Swizzle<kSwizzle_b128, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO_b128= decltype(tile_to_shape(
        SmemLayoutAtom_b128{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<kSwizzle, 2, 4>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutKV_b128 = decltype(tile_to_shape(
        SmemLayoutAtom_b128{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutAtomKtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmem>, Int<kBlockN>>,
                                                      Stride<_1, Int<kBlockKSmem>>>;
    using SmemLayoutAtomKtransposed = decltype(
        composition(Swizzle<kSwizzle, 2, 4>{}, SmemLayoutAtomKtransposedNoSwizzle{}));
    using SmemLayoutKtransposed = decltype(tile_to_shape(
        SmemLayoutAtomKtransposed{},
        make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));
    // Maybe the KtransposeNoSwizzle just needs to have the right shape
    // And the strides don't matter?
    using SmemLayoutKtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomKtransposedNoSwizzle{},
        make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));

    static constexpr int kPBlockN = kBlockN;
    // static_assert(kBlockN >= 64);
    // TD [2023-03-19]: Idk why kPBlockN = 16 and kSwizzlePdS=3 is the fastest.
    static_assert(kPBlockN == 32 || kPBlockN == 64 || kPBlockN == 128, "only support kPBlockN = 32,64,128");
    static constexpr int kSwizzlePdS = kPBlockN == 32 ? 3 : 4;

    using SmemLayoutAtomPdS_NoSwizzle = Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                                                Stride<Int<kPBlockN>, _1>>;
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, 2, 4>{},
        SmemLayoutAtomPdS_NoSwizzle{}));
    using SmemLayoutPdS_NoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomPdS_NoSwizzle{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));

    using SmemLayoutAtomPdStransposedNoSwizzle = Layout<Shape<Int<kPBlockN>, Int<kBlockM>>,
                                                        Stride<_1, Int<kPBlockN>>>;
    using SmemLayoutAtomPdStransposed = decltype(
        composition(Swizzle<kSwizzlePdS, 2, 4>{},
        SmemLayoutAtomPdStransposedNoSwizzle{}));
    using SmemLayoutPdStransposed = decltype(tile_to_shape(
        SmemLayoutAtomPdStransposed{},
        make_shape(Int<kBlockN>{}, Int<kBlockM>{})));
    using SmemLayoutPdStransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomPdStransposedNoSwizzle{},
        make_shape(Int<kBlockN>{}, Int<kBlockM>{})));


    // for lds_trans
    using SmemLayoutPdS_swz242 = decltype(tile_to_shape(
        composition(Swizzle<2, 4, 2>{}, SmemLayoutAtomPdS_NoSwizzle{}),
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutPdStransposed_swz242 = decltype(tile_to_shape(
        composition(Swizzle<2, 4, 2>{}, SmemLayoutAtomPdStransposedNoSwizzle{}),
        make_shape(Int<kBlockN>{}, Int<kBlockM>{})));

    using SmemCopyAtomPdS = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutAtomQdOtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmem>, Int<kBlockM>>,
                                                        Stride<_1, Int<kBlockKSmem>>>;
    using SmemLayoutAtomQdOtransposed = decltype(
        composition(Swizzle<kSwizzle, 2, 4>{}, SmemLayoutAtomQdOtransposedNoSwizzle{}));
    using SmemLayoutQdOtransposed = decltype(tile_to_shape(
        SmemLayoutAtomQdOtransposed{},
        make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));
    using SmemLayoutQdOtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomQdOtransposedNoSwizzle{},
        make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));

    using SmemLayoutAtomQdOtransposed_b128 = decltype(
        composition(Swizzle<kSwizzle_b128, 3, 3>{}, SmemLayoutAtomQdOtransposedNoSwizzle{}));
    using SmemLayoutQdOtransposed_b128 = decltype(tile_to_shape(
        SmemLayoutAtomQdOtransposed_b128{},
        make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));

    using SmemLayoutAtomdKVNoSwizzle = Layout<Shape<_16, Int<kBlockKSmem>>,
                                            Stride<Int<kBlockKSmem>, _1>>;
    using SmemLayoutAtomdKV = decltype(
        composition(Swizzle<3, 3, 3>{},
                    SmemLayoutAtomdKVNoSwizzle{}));
    using SmemLayoutdKV = decltype(tile_to_shape(
        SmemLayoutAtomdKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutdKVNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomdKVNoSwizzle{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutAtomdQ = decltype(
        composition(Swizzle<0, 0, 0>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;

    // Double buffer for sQ
    static constexpr int kSmemQdOSize = size(SmemLayoutQdO{}) * (No_double_buffer ? 2 : 4) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemdSSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemPSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);
    static constexpr int kSmemSize = kSmemQdOSize
        + (!Is_V_in_regs
           ? kSmemKVSize + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)
           : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)));
    // Copy K to register to reduce the shared memory to 32KB when headdim=64
    static constexpr int kSmemSize1colblock_hdim96 = (kNWarps == 4) ? 32 * 1024 : 16 * 1024;

    static constexpr int kSmemSize1colblock_origin = (kHeadDim == 128) ? kSmemKVSize
        : (
            Is_K_in_regs ?  kSmemKVSize + kSmemdSSize + kSmemPSize : kSmemQdOSize
            + (!Is_V_in_regs ? kSmemKVSize + kSmemdSSize + kSmemPSize : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + kSmemPSize))
        );

    static constexpr int kSmemSize1colblock = No_double_buffer ?
            ((kHeadDim == 96) ? kSmemSize1colblock_hdim96 : kSmemSize1colblock_origin)
            : std::max(kSmemKVSize, kSmemQdOSize + kSmemPSize + kSmemdSSize);

    static constexpr int kGmemElemsPerLoadB128 = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemElemsPerLoadB64 = sizeof(cute::uint64_t) / sizeof(Element);
    static constexpr int kGmemElemsPerLoadB32 = sizeof(cute::uint32_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoadB128 == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadB128");
    static_assert(kHeadDim % kGmemElemsPerLoadB64 == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadB64");
    static_assert(kHeadDim % kGmemElemsPerLoadB32 == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadB32");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr bool QdO_use_b128 = kBlockM * kBlockKSmem / kNThreads % 8 == 0;
    static constexpr bool KV_use_b128 = kBlockN * kBlockKSmem / kNThreads % 8 == 0;
    static constexpr int kGmemThreadsPerRowB128 = kBlockKSmem / kGmemElemsPerLoadB128;
    static constexpr int kGmemThreadsPerRowB64 = kBlockKSmem / kGmemElemsPerLoadB64;
    static constexpr int kGmemThreadsPerRowB32 = kBlockKSmem / kGmemElemsPerLoadB32;
    static constexpr int kGmemThreadsPerRowdO = QdO_use_b128 ? kGmemThreadsPerRowB128 : kGmemThreadsPerRowB64;
    static_assert(kNThreads % kGmemThreadsPerRowB128 == 0, "kNThreads must be a multiple of kGmemThreadsPerRowB128");
    static_assert(kNThreads % kGmemThreadsPerRowB64 == 0, "kNThreads must be a multiple of kGmemThreadsPerRowB64");
    static_assert(kNThreads % kGmemThreadsPerRowB32 == 0, "kNThreads must be a multiple of kGmemThreadsPerRowB32");
    using GmemLayoutAtomB128 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB128>, Int<kGmemThreadsPerRowB128>>,
                                  Stride<Int<kGmemThreadsPerRowB128>, _1>>;
    using GmemLayoutAtomB64 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB64>, Int<kGmemThreadsPerRowB64>>,
                                  Stride<Int<kGmemThreadsPerRowB64>, _1>>;
    using GmemLayoutAtomB32 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB32>, Int<kGmemThreadsPerRowB32>>,
                                  Stride<Int<kGmemThreadsPerRowB32>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    static constexpr int K_4x4_need_blockN = kNThreads / kGmemThreadsPerRowB64 * 4;
    using GmemLayoutAtomK_4x4 = std::conditional_t<
        K_4x4_need_blockN <= kBlockN,
        GmemLayoutAtomB64,
        Layout<Shape<Int<kBlockN / 4>, Shape<Int<kGmemThreadsPerRowB64>, Int<kNThreads / kGmemThreadsPerRowB64 / (kBlockN / 4)>>>,
            Stride<Int<kGmemThreadsPerRowB64>, Stride<_1, Int<kBlockN / 4 * kGmemThreadsPerRowB64>>>>
    >;
    static constexpr int K_4x2_need_blockN = kNThreads / kGmemThreadsPerRowB32 * 4;
    using GmemLayoutAtomK_4x2 = std::conditional_t<
        K_4x2_need_blockN <= kBlockN,
        GmemLayoutAtomB32,
        Layout<Shape<Int<kBlockN / 4>, Shape<Int<kGmemThreadsPerRowB32>, Int<kNThreads / kGmemThreadsPerRowB32 / (kBlockN / 4)>>>,
            Stride<Int<kGmemThreadsPerRowB32>, Stride<_1, Int<kBlockN / 4 * kGmemThreadsPerRowB32>>>>
    >;
    using GmemTiledCopyKV_4x4 = decltype(
    make_tiled_copy(UniversalCopyAtomB64{},
                    GmemLayoutAtomK_4x4{},
                    Layout<Shape<_4, _4>,
                            Stride<_4, _1>>{}));
    using GmemTiledCopyKV_4x2 = decltype(
    make_tiled_copy(UniversalCopyAtomB32{},
                    GmemLayoutAtomK_4x2{},
                    Layout<Shape<_4, _2>,
                            Stride<_2, _1>>{}));
    using GmemTiledCopyK_4x4 = GmemTiledCopyKV_4x4;
    using GmemTiledCopyK_4x2 = GmemTiledCopyKV_4x2;
    static constexpr int QdOb128_need_blockM = kNThreads / kGmemThreadsPerRowB128;
    using GmemLayoutAtomQdO_B128 = std::conditional_t<
        QdOb128_need_blockM <= kBlockM,
        GmemLayoutAtomB128,
        Layout<Shape<Int<kBlockM>, Shape<Int<kGmemThreadsPerRowB128>, Int<kNThreads / kGmemThreadsPerRowB128 / kBlockM>>>,
            Stride<Int<kGmemThreadsPerRowB128>, Stride<_1, Int<kBlockM * kGmemThreadsPerRowB128>>>>
    >;
    using GmemTiledCopyQdO_B128 = decltype(
        make_tiled_copy(UniversalCopyAtomB128{},
                        GmemLayoutAtomQdO_B128{},
                        Layout<Shape <_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopyB128 = decltype(
        make_tiled_copy(UniversalCopyAtomB128{},
                        GmemLayoutAtomB128{},
                        Layout<Shape <_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopyB64 = decltype(
        make_tiled_copy(UniversalCopyAtomB64{},
                        GmemLayoutAtomB64{},
                        Layout<Shape <_1, _4>>{}));  // Val layout, 4 vals per store
    using GmemTiledCopydQ = std::conditional_t<
        QdO_use_b128,
        GmemTiledCopyB128,
        GmemTiledCopyB64
    >;
    using GmemTiledCopyQKV = GmemTiledCopydQ;
    using GmemTiledCopyQdO = GmemTiledCopydQ;
    using GmemTiledCopyQKV_k32_32x32 = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtomB128{},
                        Layout<Shape <_1, _8>>{}));  // Val layout, 8 vals per store;
    using GmemTiledCopydO = std::conditional_t<
        QdO_use_b128,
        GmemTiledCopyB128,
        GmemTiledCopyB64
    >;
    using GmemTiledCopydKV = std::conditional_t<
        KV_use_b128,
        GmemTiledCopyB128,
        GmemTiledCopyB64
    >;
    using GmemTiledCopyV = GmemTiledCopydKV;
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <Int<kNThreads / _8{}>, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <Int<kNThreads / _16{}>, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    using GmemTiledCopydQaccumAtomicAdd = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        Layout<Shape <Int<kNThreads / _32{}>, _32>,  // Thread layout, 8 threads per row
                               Stride<_32, _1>>{},
                        Layout<Shape < _1, _1>>{}));  // Val layout, 1 val per store

    /////////////////////////////// Opt smem layout for hdim128 //////////////////////////
    // optional tiled waves layout: 4x2
    using GmemLayoutAtom4x2 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB128 / 2>, Shape<Int<kGmemThreadsPerRowB128>, _2>>,
                                     Stride<Int<kGmemThreadsPerRowB128>, Stride<_1, Int<kNThreads / 2>>>>;
    using StsLayoutAtomQdO = decltype(
        composition(Swizzle<3, 3, 4>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using StsLayoutQdO = decltype(tile_to_shape(
        StsLayoutAtomQdO{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using LdsLayoutAtomQdO = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using LdsLayoutQdO = decltype(tile_to_shape(
        LdsLayoutAtomQdO{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));


    // for ldgbsm
    using SmemLayoutAtom_ldgbsm = Layout<Shape<_8, Int<kBlockKSmem>>,
                                        Stride<Int<kBlockKSmem>, _1>>;
    // for lds_trans
    using SmemLayoutAtom_ldgbsm_swz242 = decltype(
        composition(Swizzle<2, 4, 2>{},
                    SmemLayoutAtom_ldgbsm{}));
    // for lds
    using SmemLayoutAtom_ldgbsm_swz333 = decltype(
        composition(Swizzle<3, 3, 3>{},
                    SmemLayoutAtom_ldgbsm{}));

    using SmemLayoutQdO_NoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtom_ldgbsm{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutQdO_swz242 = decltype(tile_to_shape(
        SmemLayoutAtom_ldgbsm_swz242{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutQdO_swz333 = decltype(tile_to_shape(
        SmemLayoutAtom_ldgbsm_swz333{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutKV_NoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtom_ldgbsm{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutKV_swz242 = decltype(tile_to_shape(
        SmemLayoutAtom_ldgbsm_swz242{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutKV_swz333 = decltype(tile_to_shape(
        SmemLayoutAtom_ldgbsm_swz333{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutQdOtransposed_swz242 = decltype(tile_to_shape(
        composition(Swizzle<2, 4, 2>{}, SmemLayoutAtomQdOtransposedNoSwizzle{}),
        make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));

    using SmemLayoutQdOtransposed_swz333 = decltype(tile_to_shape(
        composition(Swizzle<3, 3, 3>{}, SmemLayoutAtomQdOtransposedNoSwizzle{}),
        make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));

    using SmemLayoutKtransposed_swz242 = decltype(tile_to_shape(
        composition(Swizzle<2, 4, 2>{}, SmemLayoutAtomKtransposedNoSwizzle{}),
        make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));

    //for hdim_QdO_ldg4x4
    using SmemLayoutAtomQdOSwizzle = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdOSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomQdOSwizzle{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemLayoutQdOtSwizzle = decltype(
        composition(SmemLayoutQdOSwizzle{},
        make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));


    using SmemLayoutAtomKVSwizzle = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutKVSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomKVSwizzle{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutKtSwizzle = decltype(
        composition(SmemLayoutKVSwizzle{},
        make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));


    using GmemTiledCopyBsm1x8 = decltype(
        make_tiled_copy(Copy_Atom<MACA_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, elem_type>{},
                        GmemLayoutAtomB128{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    // tiled waves: 4x1, value: 4x4
    using GmemTiledCopyKV_D128 = decltype(
        make_tiled_copy(UniversalCopyAtomB64{},
                        Layout<Shape<_16, _16>,
                               Stride<_16, _1>>{},
                        Layout<Shape<_4, _4>,
                               Stride<_4, _1>>{}));

    // tiled waves: 2x2, value: 4x4
    using GmemTiledCopyQdO_D128 = decltype(
        make_tiled_copy(UniversalCopyAtomB64{},
                        Layout<Shape<_8, Shape<_16, _2>>,
                               Stride<_16, Stride<_1, _128>>>{},
                        Layout<Shape<_4, _4>,
                               Stride<_4, _1>>{}));
    // for hdim128 32x128 8waves
    using GmemTiledCopyBsm2x8 = decltype(
        make_tiled_copy(Copy_Atom<MACA_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, elem_type>{},
                        GmemLayoutAtom4x2{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    /////////////////////////////////////////////////////////////////////////////////////

    struct VarHeadDim {

        static constexpr int kHeadDimQ = kHeadDim_;
        static constexpr int kHeadDimV = kHeadDimV_;
        static_assert(kHeadDimQ % 32 == 0);
        static_assert(kHeadDimV % 32 == 0);
        static constexpr int kBlockKSmemQ = kHeadDimQ % 64 == 0 ? 64 : 32;
        static constexpr int kBlockKSmemV = kHeadDimV % 64 == 0 ? 64 : 32;

        using SmemLayoutAtomQ = decltype(
            composition(Swizzle<4, 2, 4>{},
                        Layout<Shape<_8, Int<kBlockKSmemQ>>,
                            Stride<Int<kBlockKSmemQ>, _1>>{}));
        using SmemLayoutAtomdO = decltype(
            composition(Swizzle<4, 2, 4>{},
                        Layout<Shape<_8, Int<kBlockKSmemV>>,
                            Stride<Int<kBlockKSmemV>, _1>>{}));
        using SmemLayoutQ = decltype(tile_to_shape(
            SmemLayoutAtomQ{},
            make_shape(Int<kBlockM>{}, Int<kHeadDimQ>{})));
        using SmemLayoutdO = decltype(tile_to_shape(
            SmemLayoutAtomdO{},
            make_shape(Int<kBlockM>{}, Int<kHeadDimV>{})));
        using SmemLayoutAtomQtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmemQ>, Int<kBlockM>>,
                                                            Stride<_1, Int<kBlockKSmemQ>>>;
        using SmemLayoutAtomQtransposed = decltype(
            composition(Swizzle<4, 2, 4>{}, SmemLayoutAtomQtransposedNoSwizzle{}));
        using SmemLayoutQtransposed = decltype(tile_to_shape(
            SmemLayoutAtomQtransposed{},
            make_shape(Int<kHeadDimQ>{}, Int<kBlockM>{})));
        using SmemLayoutQtransposedNoSwizzle = decltype(tile_to_shape(
            SmemLayoutAtomQtransposedNoSwizzle{},
            make_shape(Int<kHeadDimQ>{}, Int<kBlockM>{})));
        using SmemLayoutAtomdOtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmemV>, Int<kBlockM>>,
                                                            Stride<_1, Int<kBlockKSmemV>>>;
        using SmemLayoutAtomdOtransposed = decltype(
            composition(Swizzle<4, 2, 4>{}, SmemLayoutAtomdOtransposedNoSwizzle{}));
        using SmemLayoutdOtransposed = decltype(tile_to_shape(
            SmemLayoutAtomdOtransposed{},
            make_shape(Int<kHeadDimV>{}, Int<kBlockM>{})));
        using SmemLayoutdOtransposedNoSwizzle = decltype(tile_to_shape(
            SmemLayoutAtomdOtransposedNoSwizzle{},
            make_shape(Int<kHeadDimV>{}, Int<kBlockM>{})));

        using SmemLayoutAtomK = decltype(
            composition(Swizzle<4, 2, 4>{},
                        Layout<Shape<Int<kBlockN>, Int<kBlockKSmemQ>>,
                            Stride<Int<kBlockKSmemQ>, _1>>{}));
        using SmemLayoutAtomV = decltype(
            composition(Swizzle<3, 3, 4>{},
                        Layout<Shape<Int<kBlockN>, Int<kBlockKSmemV>>,
                            Stride<Int<kBlockKSmemV>, _1>>{}));
        using SmemLayoutK = decltype(tile_to_shape(
            SmemLayoutAtomK{},
            make_shape(Int<kBlockN>{}, Int<kHeadDimQ>{})));
        using SmemLayoutV = decltype(tile_to_shape(
            SmemLayoutAtomV{},
            make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));

        using SmemLayoutAtomKtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmemQ>, Int<kBlockN>>,
                                                        Stride<_1, Int<kBlockKSmemQ>>>;
        using SmemLayoutAtomKtransposed = decltype(
            composition(Swizzle<0, 0, 0>{}, SmemLayoutAtomKtransposedNoSwizzle{}));
        using SmemLayoutKtransposed = decltype(tile_to_shape(
            SmemLayoutAtomKtransposed{},
            make_shape(Int<kHeadDimQ>{}, Int<kBlockN>{})));
        // Maybe the KtransposeNoSwizzle just needs to have the right shape
        // And the strides don't matter?
        using SmemLayoutKtransposedNoSwizzle = decltype(tile_to_shape(
            SmemLayoutAtomKtransposedNoSwizzle{},
            make_shape(Int<kHeadDimQ>{}, Int<kBlockN>{})));

        using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                           Stride<Int<kPBlockN>, _1>>{}));
        using SmemLayoutPdS = decltype(tile_to_shape(
            SmemLayoutAtomPdS{},
            make_shape(Int<kBlockM>{}, Int<kBlockN>{})));

        using SmemLayoutAtomPdStransposedNoSwizzle = Layout<Shape<Int<kPBlockN>, Int<kBlockM>>,
                                                        Stride<_1, Int<kPBlockN>>>;
        using SmemLayoutAtomPdStransposed = decltype(
        composition(Swizzle<4, 2, 4>{},
            SmemLayoutAtomPdStransposedNoSwizzle{}));
        using SmemLayoutPdStransposed = decltype(tile_to_shape(
            SmemLayoutAtomPdStransposed{},
            make_shape(Int<kBlockN>{}, Int<kBlockM>{})));


        using SmemLayoutAtomdK = decltype(
            composition(Swizzle<3, 3, 4>{},
                        Layout<Shape<_8, Int<kBlockKSmemQ>>,
                            Stride<Int<kBlockKSmemQ>, _1>>{}));
        using SmemLayoutdK = decltype(tile_to_shape(
            SmemLayoutAtomdK{},
            make_shape(Int<kBlockN>{}, Int<kHeadDimQ>{})));

        using SmemLayoutAtomdV = decltype(
            composition(Swizzle<3, 3, 4>{},
                        Layout<Shape<_8, Int<kBlockKSmemV>>,
                            Stride<Int<kBlockKSmemV>, _1>>{}));
        using SmemLayoutdV = decltype(tile_to_shape(
            SmemLayoutAtomdV{},
            make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));
        using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;

        using SmemLayoutAtomdQ = decltype(
            composition(Swizzle<0, 0, 0>{},
                        Layout<Shape<_8, Int<kBlockKSmemQ>>,
                            Stride<Int<kBlockKSmemQ>, _1>>{}));
        using SmemLayoutdQ = decltype(tile_to_shape(
            SmemLayoutAtomdQ{},
            make_shape(Int<kBlockM>{}, Int<kHeadDimQ>{})));
        using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;


        //for xcore1500 new solution
        using SmemLayoutV_swz333 = decltype(tile_to_shape(
            SmemLayoutAtom_ldgbsm_swz333{},
            make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));
        using SmemLayoutV_NoSwizzle = decltype(tile_to_shape(
            SmemLayoutAtom_ldgbsm{},
            make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));

        using SmemLayoutdO_NoSwizzle = decltype(tile_to_shape(
            SmemLayoutAtom_ldgbsm{},
            make_shape(Int<kBlockM>{}, Int<kHeadDimV>{})));
        using SmemLayoutdO_swz242 = decltype(tile_to_shape(
            SmemLayoutAtom_ldgbsm_swz242{},
            make_shape(Int<kBlockM>{}, Int<kHeadDimV>{})));
        using SmemLayoutdOtransposed_swz242 = decltype(tile_to_shape(
            composition(Swizzle<2, 4, 2>{}, SmemLayoutAtomQdOtransposedNoSwizzle{}),
            make_shape(Int<kHeadDimV>{}, Int<kBlockM>{})));


        // Double buffer for sQ
        static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
        static constexpr int kSmemdOSize = size(SmemLayoutdO{}) * sizeof(Element);
        static constexpr int kSmemKSize = size(SmemLayoutK{}) * sizeof(Element);
        static constexpr int kSmemVSize = size(SmemLayoutV{}) * sizeof(Element);
        static constexpr int kSmemdSSize = size(SmemLayoutPdS{}) * sizeof(Element);
        static constexpr int kSmemPSize = size(SmemLayoutPdS{}) * sizeof(Element);
        static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);
        // Copy K to register to reduce the shared memory to 32KB when headdim=64
        static constexpr int kSmemSize1colblock = std::max(
            No_double_buffer ? kSmemQSize + kSmemdOSize + kSmemdSSize * 2 + kSmemPSize : kSmemQSize*2 + kSmemdOSize*2 + kSmemdSSize + kSmemPSize
            , kSmemKSize + kSmemVSize);

        static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
        static constexpr int kGmemElemsPerLoadB64 = sizeof(cute::uint64_t) / sizeof(Element);
        static constexpr int kGmemElemsPerLoadB32 = sizeof(cute::uint32_t) / sizeof(Element);
        static_assert(kHeadDimQ % kGmemElemsPerLoad == 0, "kHeadDimQ must be a multiple of kGmemElemsPerLoad");
        static_assert(kHeadDimV % kGmemElemsPerLoad == 0, "kHeadDimV must be a multiple of kGmemElemsPerLoad");
        // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
        // to affect speed in practice.
        static constexpr int kGmemThreadsPerRowQ = kBlockKSmemQ / kGmemElemsPerLoad;
        static constexpr int kGmemThreadsPerRowV = kBlockKSmemV / kGmemElemsPerLoad;
        static constexpr int kGmemThreadsPerRowB64 = kBlockKSmemQ / kGmemElemsPerLoadB64;
        static constexpr int kGmemThreadsPerRowB32 = kBlockKSmemQ / kGmemElemsPerLoadB32;
        static_assert(kNThreads % kGmemThreadsPerRowQ == 0, "kNThreads must be a multiple of kGmemThreadsPerRowQ");
        static_assert(kNThreads % kGmemThreadsPerRowV == 0, "kNThreads must be a multiple of kGmemThreadsPerRowV");
        static_assert(kNThreads % kGmemThreadsPerRowB64 == 0, "kNThreads must be a multiple of kGmemThreadsPerRowB64");
        static_assert(kNThreads % kGmemThreadsPerRowB32 == 0, "kNThreads must be a multiple of kGmemThreadsPerRowB32");

        using GmemLayoutAtomQ = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowQ>, Int<kGmemThreadsPerRowQ>>,
                                    Stride<Int<kGmemThreadsPerRowQ>, _1>>;
        using GmemLayoutAtomV = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowV>, Int<kGmemThreadsPerRowV>>,
                                    Stride<Int<kGmemThreadsPerRowV>, _1>>;
        using GmemLayoutAtomB64 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB64>, Int<kGmemThreadsPerRowB64>>,
                                    Stride<Int<kGmemThreadsPerRowB64>, _1>>;
        using GmemLayoutAtomB32 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB32>, Int<kGmemThreadsPerRowB32>>,
                                    Stride<Int<kGmemThreadsPerRowB32>, _1>>;
        // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
        // from the same address by the same threadblock. This is slightly faster.
        using Gmem_copy_struct = std::conditional_t<
            Has_cp_async,
            SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
            DefaultCopy
        >;
        using GmemTiledCopyQ = decltype(
            make_tiled_copy(UniversalCopyAtomB64{},
                            GmemLayoutAtomB64{},
                            Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per read
        using GmemTiledCopyK_4x2 = decltype(
        make_tiled_copy(UniversalCopyAtomB32{},
                        GmemLayoutAtomB32{},
                        Layout<Shape<_4, _2>, Stride<_2, _1>>{}));
        using GmemTiledCopyK_4x4 = decltype(
        make_tiled_copy(UniversalCopyAtomB64{},
                    GmemLayoutAtomB64{},
                    Layout<Shape<_4, _4>, Stride<_4, _1>>{}));
        using GmemTiledCopydO = decltype(
            make_tiled_copy(UniversalCopyAtomB128{},
                            GmemLayoutAtomV{},
                            Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
        using GmemTiledCopydOB64 = decltype(
            make_tiled_copy(UniversalCopyAtomB64{},
                            GmemLayoutAtomB64{},
                            Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
        using GmemTiledCopyV = decltype(
            make_tiled_copy(UniversalCopyAtomB128{},
                            GmemLayoutAtomV{},
                            Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
        using GmemTiledCopydK = decltype(
            make_tiled_copy(UniversalCopyAtomB128{},
                            GmemLayoutAtomQ{},
                            Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
        using GmemTiledCopydV = decltype(
            make_tiled_copy(UniversalCopyAtomB128{},
                            GmemLayoutAtomV{},
                            Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
