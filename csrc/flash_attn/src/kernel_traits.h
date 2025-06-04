/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/algorithm/copy.hpp"

#include "mctlass/mctlass.h"
#include "mctlass/layout/layout.h"
#include <mctlass/numeric_types.h>

using namespace cute;

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = mctlass::half_t>
struct Flash_kernel_traits {
#if defined(__MACA_ARCH__)
    using Element = elem_type;
    static constexpr bool Has_cp_async = false;
#else
    using Element = mctlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__MACA_ARCH__)
    using MMA_Atom_Arch =
        std::conditional_t<std::is_same_v<elem_type, mctlass::half_t>, MMA_Atom<MACA_16x16x16_F32F16F16F32>,
                           MMA_Atom<MACA_16x16x16_F32BF16BF16F32>>;
    using ValLayoutMNK = Layout<Shape<_1, _1, _1>>;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using ValLayoutMNK = Layout<Shape<_1, _2, _2>>;
#endif

    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyB64 = Copy_Atom<UniversalCopy<uint64_t>, elem_type>;
    using UniversalCopyAtom32 = Copy_Atom<UniversalCopy<uint32_t>, elem_type>;
};

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_ = false,
          bool Share_Q_K_smem_ = false, typename elem_type = mctlass::half_t, bool Is_Splits_ = false,
          typename Base = Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomB64 = typename Base::SmemCopyB64;
    using UniversalCopyAtom32 = typename Base::UniversalCopyAtom32;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    static constexpr int MBase = 3;
    static constexpr int SShift = 3;

    using TiledMma =
        TiledMMA<typename Base::MMA_Atom_Arch, Layout<Shape<Int<kNWarps>, _1, _1>>, typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(composition(Swizzle<kSwizzle, MBase, SShift>{},

                                                 Layout<Shape<_16, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomVtransposedNoSwizzle =
        Layout<Shape<Int<kBlockKSmem>, Int<kBlockN>>, Stride<_1, Int<kBlockKSmem>>>;
    using SmemLayoutAtomVtransposed =
        decltype(composition(Swizzle<kSwizzle, MBase, SShift>{}, SmemLayoutAtomVtransposedNoSwizzle{}));
    using SmemLayoutVtransposed =
        decltype(tile_to_shape(SmemLayoutAtomVtransposed{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    using SmemLayoutVtransposedNoSwizzle =
        decltype(tile_to_shape(SmemLayoutAtomVtransposedNoSwizzle{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    using SmemLayoutAtomO = decltype(composition(
        Swizzle<kSwizzle, MBase, SShift>{}, Layout<Shape<Int<16>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemCopyAtomO = Copy_Atom<UniversalCopy<uint64_t>, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemSize =
        Share_Q_K_smem ? std::max((Is_Splits_ ? 2 : 1) * kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;
    static constexpr int kRegSize = kSmemSize / sizeof(uint32_t) / kNThreads;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");

    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    using Gmem_copy_struct = std::conditional_t<Has_cp_async, SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DefaultCopy>;
    using GmemTiledCopyQKV =
        decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{}, GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));
    using GmemTiledCopyO =
        decltype(make_tiled_copy(Copy_Atom<DefaultCopy, Element>{}, GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));

    using GmemLayoutAtomOaccum = std::conditional_t<kBlockKSmem == 32, Layout<Shape<_32, _8>, Stride<_8, _1>>,
                                                    Layout<Shape<_16, _16>, Stride<_16, _1>>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{}, GmemLayoutAtomOaccum{}, Layout<Shape<_1, _4>>{}));
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                                                            GmemLayoutAtomRotcossin{}, Layout<Shape<_1, _4>>{}));
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{}, GmemLayoutAtomRotcossin{}, Layout<Shape<_1, _8>>{}));
};

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int AtomLayoutMSdP_ = 1, int AtomLayoutNdKV = 2,
          int AtomLayoutMdQ = 2, bool Is_V_in_regs_ = false, bool Is_K_in_regs_ = false, bool No_double_buffer_ = false,
          typename elem_type = mctlass::half_t,
          typename Base = Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Flash_bwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;
    using SmemCopyB64 = typename Base::SmemCopyB64;
    using UniversalCopyAtom32 = typename Base::UniversalCopyAtom32;

    static constexpr bool Is_V_in_regs = Is_V_in_regs_;
    static constexpr bool Is_K_in_regs = Is_K_in_regs_;
    static constexpr bool No_double_buffer = No_double_buffer_;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutNdKV == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);

    using TiledMmaSdP =
        TiledMMA<typename Base::MMA_Atom_Arch, Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
                 typename Base::ValLayoutMNK>;

    using TiledMmadKV =
        TiledMMA<typename Base::MMA_Atom_Arch, Layout<Shape<Int<AtomLayoutNdKV>, Int<kNWarps / AtomLayoutNdKV>, _1>>,
                 typename Base::ValLayoutMNK>;

    using TiledMmadQ =
        TiledMMA<typename Base::MMA_Atom_Arch, Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,
                 typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQdO =
        decltype(composition(Swizzle<0, 0, 0>{}, Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO = decltype(tile_to_shape(SmemLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutAtomQdO_hdim64 =
        decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO_hdim64 =
        decltype(tile_to_shape(SmemLayoutAtomQdO_hdim64{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutAtomKV = decltype(composition(
        Swizzle<0, 0, 0>{}, Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(

        SmemLayoutAtomKV{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutAtomKV_hdim64 = decltype(composition(
        Swizzle<4, 2, 4>{}, Layout<Shape<Int<kBlockN / kNWarps>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutKV_hdim64 = decltype(tile_to_shape(

        SmemLayoutAtomKV_hdim64{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutAtomKtransposedNoSwizzle =
        Layout<Shape<Int<kBlockKSmem>, Int<kBlockN>>, Stride<_1, Int<kBlockKSmem>>>;
    using SmemLayoutAtomKtransposed = decltype(composition(Swizzle<0, 0, 0>{}, SmemLayoutAtomKtransposedNoSwizzle{}));
    using SmemLayoutKtransposed =
        decltype(tile_to_shape(SmemLayoutAtomKtransposed{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));

    using SmemLayoutKtransposedNoSwizzle =
        decltype(tile_to_shape(SmemLayoutAtomKtransposedNoSwizzle{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));

    using SmemLayoutAtomKtransposed_hdim64 =
        decltype(composition(Swizzle<4, 2, 4>{}, SmemLayoutAtomKtransposedNoSwizzle{}));
    using SmemLayoutKtransposed_hdim64 =
        decltype(tile_to_shape(SmemLayoutAtomKtransposed_hdim64{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));

    static constexpr int kPBlockN = kBlockN;

    static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);

    static constexpr int kSwizzlePdS = 3;
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<0, 0, 0>{}, Layout<Shape<Int<kBlockM>, Int<kPBlockN>>, Stride<Int<kPBlockN>, _1>>{}));
    using SmemLayoutPdS = decltype(tile_to_shape(SmemLayoutAtomPdS{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutAtomPdStransposedNoSwizzle = Layout<Shape<Int<kPBlockN>, Int<kBlockM>>, Stride<_1, Int<kPBlockN>>>;
    using SmemLayoutAtomPdStransposed =
        decltype(composition(Swizzle<0, 0, 0>{}, SmemLayoutAtomPdStransposedNoSwizzle{}));
    using SmemLayoutPdStransposed =
        decltype(tile_to_shape(SmemLayoutAtomPdStransposed{}, make_shape(Int<kBlockN>{}, Int<kBlockM>{})));
    using SmemLayoutPdStransposedNoSwizzle =
        decltype(tile_to_shape(SmemLayoutAtomPdStransposedNoSwizzle{}, make_shape(Int<kBlockN>{}, Int<kBlockM>{})));

    using SmemCopyAtomPdS = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutAtomQdOtransposedNoSwizzle =
        Layout<Shape<Int<kBlockKSmem>, Int<kBlockM>>, Stride<_1, Int<kBlockKSmem>>>;
    using SmemLayoutAtomQdOtransposed =
        decltype(composition(Swizzle<0, 0, 0>{}, SmemLayoutAtomQdOtransposedNoSwizzle{}));
    using SmemLayoutQdOtransposed =
        decltype(tile_to_shape(SmemLayoutAtomQdOtransposed{}, make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));
    using SmemLayoutQdOtransposedNoSwizzle =
        decltype(tile_to_shape(SmemLayoutAtomQdOtransposedNoSwizzle{}, make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));

    using SmemLayoutAtomQdOtransposed_hdim64 =
        decltype(composition(Swizzle<3, 3, 3>{}, SmemLayoutAtomQdOtransposedNoSwizzle{}));
    using SmemLayoutQdOtransposed_hdim64 =
        decltype(tile_to_shape(SmemLayoutAtomQdOtransposed_hdim64{}, make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));

    using SmemLayoutAtomdKV =
        decltype(composition(Swizzle<0, 0, 0>{}, Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdKV = decltype(tile_to_shape(SmemLayoutAtomdKV{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutAtomdKV_hdim64 =
        decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdKV_hdim64 =
        decltype(tile_to_shape(SmemLayoutAtomdKV_hdim64{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutAtomdQ =
        decltype(composition(Swizzle<0, 0, 0>{}, Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(SmemLayoutAtomdQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;

    static constexpr int kSmemQdOSize = size(SmemLayoutQdO{}) * (No_double_buffer ? 2 : 3) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemdSSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemPSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);
    static constexpr int kSmemSize =
        kSmemQdOSize + (!Is_V_in_regs
                            ? kSmemKVSize + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)
                            : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)));

    static constexpr int kSmemSize1colblock =
        (kHeadDim == 128)
            ? kSmemQdOSize * 2
            : (Is_K_in_regs ? kSmemKVSize + kSmemdSSize + kSmemPSize
                            : kSmemQdOSize + (!Is_V_in_regs
                                                  ? kSmemKVSize + kSmemdSSize + kSmemPSize
                                                  : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + kSmemPSize)));
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");

    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    using Gmem_copy_struct = std::conditional_t<Has_cp_async, SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DefaultCopy>;
    using GmemTiledCopyQKV =
        decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{}, GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));
    using GmemTiledCopyKV_hdim64 = decltype(make_tiled_copy(SmemCopyB64{}, Layout<Shape<_16, _16>, Stride<_16, _1>>{},
                                                            Layout<Shape<_4, _4>, Stride<_4, _1>>{}));
    using GmemTiledCopyKV_hdim64_8wave = decltype(make_tiled_copy(
        UniversalCopyAtom32{}, Layout<Shape<_16, _32>, Stride<_32, _1>>{}, Layout<Shape<_4, _2>, Stride<_2, _1>>{}));
    using GmemTiledCopydO =
        decltype(make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{}, GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));
    using GmemTiledCopydKV =
        decltype(make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{}, GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));
    using GmemTiledCopydQ =
        decltype(make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{}, GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));
    using GmemLayoutAtomdQaccum =
        std::conditional_t<kBlockKSmem == 32, Layout<Shape<Int<kNThreads / _8{}>, _8>, Stride<_8, _1>>,
                           Layout<Shape<Int<kNThreads / _16{}>, _16>, Stride<_16, _1>>>;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{}, GmemLayoutAtomdQaccum{}, Layout<Shape<_1, _4>>{}));

    using GmemTiledCopydQaccumAtomicAdd = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        Layout<Shape<Int<kNThreads / _32{}>, _32>, Stride<_32, _1>>{}, Layout<Shape<_1, _1>>{}));

    using StsLayoutAtomQdO =
        decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<_32, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using StsLayoutQdO = decltype(tile_to_shape(StsLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using LdsLayoutAtomQdO =
        decltype(composition(Swizzle<4, 2, 4>{}, Layout<Shape<_16, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using LdsLayoutQdO = decltype(tile_to_shape(LdsLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemLayoutQdOtNoSwizzle =
        decltype(tile_to_shape(Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{},
                               make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutAtomQdOSwizzle =
        decltype(composition(Swizzle<4, 2, 4>{}, Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdOSwizzle =
        decltype(tile_to_shape(SmemLayoutAtomQdOSwizzle{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemLayoutQdOtSwizzle =
        decltype(composition(SmemLayoutQdOSwizzle{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));

    using SmemLayoutAtomKVSwizzle = decltype(composition(
        Swizzle<4, 2, 4>{}, Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutKVSwizzle =
        decltype(tile_to_shape(SmemLayoutAtomKVSwizzle{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutKtSwizzle =
        decltype(composition(SmemLayoutKVSwizzle{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKtNoSwizzle = SmemLayoutKtransposedNoSwizzle;

    using GmemTiledCopyBsm1x8 = decltype(make_tiled_copy(
        Copy_Atom<MACA_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, elem_type>{}, GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));

    using GmemTiledCopyKV_D128 = decltype(make_tiled_copy(SmemCopyB64{}, Layout<Shape<_16, _16>, Stride<_16, _1>>{},
                                                          Layout<Shape<_4, _4>, Stride<_4, _1>>{}));

    using GmemTiledCopyQdO_D128 =
        decltype(make_tiled_copy(SmemCopyB64{}, Layout<Shape<_8, Shape<_16, _2>>, Stride<_16, Stride<_1, _128>>>{},
                                 Layout<Shape<_4, _4>, Stride<_4, _1>>{}));

    using GmemLayoutAtomdQaccum_hdim128_32_32 = Layout<Shape<_32, _4>, Stride<_4, _1>>;

    using GmemTiledCopydQaccum_hdim128_32_32 = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, ElementAccum>{}, GmemLayoutAtomdQaccum_hdim128_32_32{}, Layout<Shape<_1, _4>>{}));

    using GmemTiledCopydQaccumAtomicAdd_hdim128_32_32 = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, ElementAccum>{}, Layout<Shape<_4, _32>, Stride<_32, _1>>{}, Layout<Shape<_1, _1>>{}));
};
