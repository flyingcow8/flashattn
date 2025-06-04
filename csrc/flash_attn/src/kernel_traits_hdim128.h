/******************************************************************************
 * Copyright (c) 2024, Metax.
 ******************************************************************************/
#pragma once

#include "kernel_traits.h"

template <int AtomLayoutNdKV, typename elem_type>
struct Flash_bwd_kernel_traits<128, 32, 128, 8, 2, AtomLayoutNdKV, 2, true, true, true, elem_type>
    : public Flash_kernel_traits<128, 32, 128, 8, elem_type> {
    using Base = Flash_kernel_traits<128, 32, 128, 8, elem_type>;
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;
    using SmemCopyB64 = typename Base::SmemCopyB64;

    static constexpr int kNWarps = 8;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = 32;
    static constexpr int kBlockN = 128;
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockKSmem = 64;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = 3;

    static constexpr int AtomLayoutMSdP = 2;
    static constexpr int AtomLayoutMdQ = 2;

    using TiledMmaSdP =
        TiledMMA<typename Base::MMA_Atom_Arch, Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
                 typename Base::ValLayoutMNK>;

    using TiledMmadKV =
        TiledMMA<typename Base::MMA_Atom_Arch, Layout<Shape<Int<AtomLayoutNdKV>, Int<kNWarps / AtomLayoutNdKV>, _1>>,
                 typename Base::ValLayoutMNK>;

    using TiledMmadQ =
        TiledMMA<typename Base::MMA_Atom_Arch, Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,
                 typename Base::ValLayoutMNK>;

    using StsLayoutAtomQdO =
        decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<_32, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using StsLayoutQdO = decltype(tile_to_shape(StsLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using LdsLayoutAtomQdO =
        decltype(composition(Swizzle<4, 2, 4>{}, Layout<Shape<_16, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using LdsLayoutQdO = decltype(tile_to_shape(LdsLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutQdOtNoSwizzle =
        decltype(tile_to_shape(Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{},
                               make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemLayoutQdO = SmemLayoutQdOtNoSwizzle;

    using SmemLayoutAtomQdOSwizzle =
        decltype(composition(Swizzle<4, 2, 4>{}, Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdOSwizzle =
        decltype(tile_to_shape(SmemLayoutAtomQdOSwizzle{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemLayoutQdOtSwizzle =
        decltype(composition(SmemLayoutQdOSwizzle{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));

    using SmemLayoutAtomKVSwizzle =
        decltype(composition(Swizzle<4, 2, 4>{}, Layout<Shape<_16, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutKVSwizzle =
        decltype(tile_to_shape(SmemLayoutAtomKVSwizzle{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutKtSwizzle =
        decltype(composition(SmemLayoutKVSwizzle{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));

    using SmemLayoutAtomKtransposedNoSwizzle =
        Layout<Shape<Int<kBlockKSmem>, Int<kBlockN>>, Stride<_1, Int<kBlockKSmem>>>;
    using SmemLayoutKtransposedNoSwizzle =
        decltype(tile_to_shape(SmemLayoutAtomKtransposedNoSwizzle{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));
    using SmemLayoutKtNoSwizzle = SmemLayoutKtransposedNoSwizzle;
    using SmemLayoutKV = SmemLayoutKVSwizzle;

    static constexpr int kPBlockN = kBlockN;
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

    using SmemLayoutAtomdKV =
        decltype(composition(Swizzle<0, 0, 0>{}, Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdKV = decltype(tile_to_shape(SmemLayoutAtomdKV{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutAtomdQ =
        decltype(composition(Swizzle<0, 0, 0>{}, Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(SmemLayoutAtomdQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;

    static constexpr int kSmemQdOSize = size(SmemLayoutQdO{}) * 2 * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemdSSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemPSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);

    static constexpr int kSmemSize1colblock = kSmemKVSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static constexpr int kGmemThreadsPerRow_B64 = kBlockKSmem / (sizeof(cute::uint64_t) / sizeof(Element));
    static constexpr int kSmemThreadsPerRow = kBlockKSmem / (sizeof(cute::uint64_t) / sizeof(Element));
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");

    using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    using GmemLayoutAtom4x1 = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow / 2>, Int<kGmemThreadsPerRow>>,
                                     Stride<Int<kGmemThreadsPerRow>, _1>>;

    using GmemLayoutAtom4x2 = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow / 2>, Shape<Int<kGmemThreadsPerRow>, _2>>,
                                     Stride<Int<kGmemThreadsPerRow>, Stride<_1, Int<kNThreads / 2>>>>;

    using Gmem_copy_struct = DefaultCopy;
    using GmemTiledCopyQKV =
        decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{}, GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));
    using GmemTiledCopydO =
        decltype(make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{}, GmemLayoutAtom4x1{}, Layout<Shape<_1, _8>>{}));
    using GmemTiledCopydKV =
        decltype(make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{}, GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));
    using GmemTiledCopydQ =
        decltype(make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{}, GmemLayoutAtom4x2{}, Layout<Shape<_1, _8>>{}));
    using GmemLayoutAtomdQaccum = Layout<Shape<Int<kNThreads / 16 / 2>, _16>, Stride<_16, _1>>;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{}, GmemLayoutAtomdQaccum{}, Layout<Shape<_1, _4>>{}));

    using GmemTiledCopydQaccumAtomicAdd =
        decltype(make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                                 Layout<Shape<Int<kNThreads / 32>, _32>, Stride<_32, _1>>{}, Layout<Shape<_1, _1>>{}));

    using GmemTiledCopyBsm1x8 =
        decltype(make_tiled_copy(Copy_Atom<MACA_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, elem_type>{},
                                 GmemLayoutAtom4x2{}, Layout<Shape<_1, _8>>{}));

    using GmemTiledCopy4x4 =
        decltype(make_tiled_copy(SmemCopyB64{},
                                 Layout<Shape<Int<kNThreads / kGmemThreadsPerRow_B64>, Int<kGmemThreadsPerRow_B64>>,
                                        Stride<Int<kGmemThreadsPerRow_B64>, _1>>{},
                                 Layout<Shape<_4, _4>, Stride<_4, _1>>{}));
};