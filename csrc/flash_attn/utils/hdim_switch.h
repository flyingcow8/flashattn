#pragma once

#include "hdim_config.h"
#include <utility>
#include <iostream>

template<int... Values>
struct HeadDimSwitchImpl;

template<int First, int Second, int... Rest>
struct HeadDimSwitchImpl<First, Second, Rest...> {
    template<typename F>
    static auto apply(int headdim, F&& func) {
        if (headdim <= First) {
          return func(std::integral_constant<int, First>());
        } else {
          return HeadDimSwitchImpl<Second, Rest...>::apply(headdim, std::forward<F>(func));
        }
    }
};

template<int Last>
struct HeadDimSwitchImpl<Last> {
    template<typename F>
    static auto apply(int headdim, F&& func) {
        if (headdim > Last) {
            std::cerr << "HEADDIM exceeds maximum configured value:" << Last << std::endl;
            std::cerr << "HEADDIM is set to 0" << std::endl;
            return func(std::integral_constant<int, 0>());
        }
        return func(std::integral_constant<int, Last>());
    }
};

#define HEADDIM_SWITCH(HEADDIM, ...)                                                 \
    [&](auto&& headdim_) {                                                           \
        return HeadDimSwitchImpl<HDIM_CONFIG>::apply(headdim_, [&](auto kHeadDim) {  \
          constexpr int kHeadDimension = decltype(kHeadDim)::value;                        \
          __VA_ARGS__                                                                \
        });                                                                          \
    }(HEADDIM)
