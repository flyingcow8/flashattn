#pragma once

#include "arch.h"

#define CHECK_HDIM(X) (defined(HDIM_ALL) || defined(HDIM_##X))

#define CHECK_MSG(x, ...) do { if((x) == false) {throw std::invalid_argument(__VA_ARGS__);} }while(0)

/// CONST_PRECOND && COND
#define BOOL_SWITCH_AND_CONST_PRECOND(CONST_PRECOND, COND, CONST_NAME, ...) \
  [&] {                                                                      \
    if constexpr (CONST_PRECOND) {                                           \
      if (COND) {                                                            \
        constexpr static bool CONST_NAME = true;                             \
        return __VA_ARGS__();                                                \
      } else {                                                               \
        constexpr static bool CONST_NAME = false;                            \
        return __VA_ARGS__();                                                \
      }                                                                      \
    } else {                                                                 \
      constexpr static bool CONST_NAME = false;                              \
      return __VA_ARGS__();                                                  \
    }                                                                        \
  }()

/// CONST_PRECOND || COND
#define BOOL_SWITCH_OR_CONST_PRECOND(CONST_PRECOND, COND, CONST_NAME, ...)   \
  [&] {                                                                      \
    if constexpr (CONST_PRECOND) {                                           \
      constexpr static bool CONST_NAME = true;                               \
      return __VA_ARGS__();                                                  \
    } else {                                                                 \
      if (COND) {                                                            \
        constexpr static bool CONST_NAME = true;                             \
        return __VA_ARGS__();                                                \
      } else {                                                               \
        constexpr static bool CONST_NAME = false;                            \
        return __VA_ARGS__();                                                \
      }                                                                      \
    }                                                                        \
  }()

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  BOOL_SWITCH_AND_CONST_PRECOND(true, COND, CONST_NAME, __VA_ARGS__)

#define ROWNUM_SWITCH_ALL(COND, CONST_NAME, ...) \
  [&] {                                      \
    if (COND)                                \
    {                                        \
      constexpr static int CONST_NAME = 2;   \
      return __VA_ARGS__();                  \
    }                                        \
    else                                     \
    {                                        \
      constexpr static int CONST_NAME = 1;   \
      return __VA_ARGS__();                  \
    }                                        \
  }()


#define ROWNUM_SWITCH_1(COND, CONST_NAME, ...)     \
    [&] {                                          \
            constexpr static int CONST_NAME = 1;   \
            return __VA_ARGS__();                  \
    }()

#define ROWNUM_SWITCH_2(COND, CONST_NAME, ...)     \
    [&] {                                          \
            constexpr static int CONST_NAME = 2;   \
            return __VA_ARGS__();                  \
    }()


#define BOOL_SWITCH_TRUE(COND, CONST_NAME, ...)                   \
    [&]{                                                          \
        constexpr static bool CONST_NAME = true;                  \
        return __VA_ARGS__();                                     \
    }()

#define BOOL_SWITCH_PRECOND_TRUE(COND1, COND2, CONST_NAME, ...)    \
    [&]{                                                          \
        constexpr static bool CONST_NAME = true;                  \
        return __VA_ARGS__();                                     \
    }()

#define BOOL_SWITCH_FALSE(COND, CONST_NAME, ...)                  \
    [&]{                                                          \
        constexpr static bool CONST_NAME = false;                  \
        return __VA_ARGS__();                                     \
    }()

#define BOOL_SWITCH_PRECOND_FALSE(COND1, COND2, CONST_NAME, ...)   \
    [&]{                                                          \
        constexpr static bool CONST_NAME = false;                  \
        return __VA_ARGS__();                                     \
    }()

////////////////////////////////////////////////////////////
//
// FP16_SWITCH
//
////////////////////////////////////////////////////////////

#ifdef FA_DTYPE_ALL
  #define FP16_SWITCH(COND, ...)               \
    [&] {                                      \
      if (COND) {                              \
        using elem_type = mctlass::half_t;     \
        return __VA_ARGS__();                  \
      } else {                                 \
        using elem_type = mctlass::bfloat16_t; \
        return __VA_ARGS__();                  \
      }                                        \
    }()

#elif defined(FA_DTYPE_FP16)

  #define FP16_SWITCH(COND, ...)               \
    [&] {                                      \
        using elem_type = mctlass::half_t;     \
        return __VA_ARGS__();                  \
    }()

#else
  #define FP16_SWITCH(COND, ...)               \
    [&] {                                      \
        using elem_type = mctlass::bfloat16_t; \
        return __VA_ARGS__();                  \
    }()
#endif

////////////////////////////////////////////////////////////////
//
// BOOL SWITCH
//
///////////////////////////////////////////////////////////////

#ifdef EVENK_TRUE
    #define EVENK_SWITCH BOOL_SWITCH_TRUE
#elif defined(EVENK_FALSE)
    #define EVENK_SWITCH BOOL_SWITCH_FALSE
#else
    #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef LOCAL_TRUE
    #define LOCAL_SWITCH BOOL_SWITCH_PRECOND_TRUE
#elif defined(LOCAL_FALSE)
    #define LOCAL_SWITCH BOOL_SWITCH_PRECOND_FALSE
#else
    #define LOCAL_SWITCH BOOL_SWITCH_AND_CONST_PRECOND
#endif

#ifdef RETURN_SOFTMAX_TRUE
    #define RETURN_SOFTMAX_SWITCH BOOL_SWITCH_PRECOND_TRUE
#elif defined(RETURN_SOFTMAX_FALSE)
    #define RETURN_SOFTMAX_SWITCH BOOL_SWITCH_PRECOND_FALSE
#else
    #define RETURN_SOFTMAX_SWITCH BOOL_SWITCH_AND_CONST_PRECOND
#endif

#ifdef EVENMN_TRUE
    #define EVENMN_SWITCH BOOL_SWITCH_PRECOND_TRUE
#elif defined(EVENMN_FALSE)
    #define EVENMN_SWITCH BOOL_SWITCH_PRECOND_FALSE
#else
    #define EVENMN_SWITCH BOOL_SWITCH_AND_CONST_PRECOND
#endif

#ifdef ALIBI_TRUE
    #define ALIBI_SWITCH BOOL_SWITCH_TRUE
#elif defined(ALIBI_FALSE)
    #define ALIBI_SWITCH BOOL_SWITCH_FALSE
#else
    #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef ATTN_MASK_TRUE
    #define ATTN_MASK_SWITCH BOOL_SWITCH_PRECOND_TRUE
#elif defined(ATTN_MASK_FALSE)
    #define ATTN_MASK_SWITCH BOOL_SWITCH_PRECOND_FALSE
#else
    #define ATTN_MASK_SWITCH BOOL_SWITCH_AND_CONST_PRECOND
#endif

#ifdef MERGE_ATTN_MASK_LDG_TRUE
    #define MERGE_ATTN_MASK_LDG_SWITCH BOOL_SWITCH_PRECOND_TRUE
#elif defined(MERGE_ATTN_MASK_LDG_FALSE)
    #define MERGE_ATTN_MASK_LDG_SWITCH BOOL_SWITCH_PRECOND_FALSE
#else
    #define MERGE_ATTN_MASK_LDG_SWITCH BOOL_SWITCH_AND_CONST_PRECOND
#endif

#ifdef ROWNUM_TRUE
    #define ROWNUM_SWITCH ROWNUM_SWITCH_2
#elif defined(ROWNUM_FALSE)
    #define ROWNUM_SWITCH ROWNUM_SWITCH_1
#else
    #define ROWNUM_SWITCH ROWNUM_SWITCH_ALL
#endif

#ifdef SOFTCAP_TRUE
    #define SOFTCAP_SWITCH BOOL_SWITCH_PRECOND_TRUE
#elif defined(SOFTCAP_FALSE)
    #define SOFTCAP_SWITCH BOOL_SWITCH_PRECOND_FALSE
#else
    #define SOFTCAP_SWITCH BOOL_SWITCH_AND_CONST_PRECOND
#endif

#ifdef DETERMINISTIC_TRUE
    #define DETERMINISTIC_SWITCH BOOL_SWITCH_TRUE
#elif defined(DETERMINISTIC_FALSE)
    #define DETERMINISTIC_SWITCH BOOL_SWITCH_FALSE
#else
    #define DETERMINISTIC_SWITCH BOOL_SWITCH
#endif

#ifdef BALANCE_TRUE
    #define BALANCE_SWITCH BOOL_SWITCH_PRECOND_TRUE
#elif defined(BALANCE_FALSE)
    #define BALANCE_SWITCH BOOL_SWITCH_PRECOND_FALSE
#else
    #define BALANCE_SWITCH BOOL_SWITCH_AND_CONST_PRECOND
#endif

#ifdef SPLIT_TRUE
    #define SPLIT_SWITCH BOOL_SWITCH
#elif defined(SPLIT_FALSE)
    #define SPLIT_SWITCH BOOL_SWITCH
#else
    #define SPLIT_SWITCH BOOL_SWITCH
#endif

#ifdef APPENDKV_TRUE
    #define APPENDKV_SWITCH BOOL_SWITCH_TRUE
#elif defined(APPENDKV_FALSE)
    #define APPENDKV_SWITCH BOOL_SWITCH_FALSE
#else
    #define APPENDKV_SWITCH BOOL_SWITCH
#endif

#ifdef PAGE_ATTN_TRUE
    #define PAGE_ATTN_SWITCH BOOL_SWITCH_TRUE
#elif defined(PAGE_ATTN_FALSE)
    #define PAGE_ATTN_SWITCH BOOL_SWITCH_FALSE
#else
    #define PAGE_ATTN_SWITCH BOOL_SWITCH
#endif

#ifdef DROPOUT_TRUE
    #define DROPOUT_SWITCH BOOL_SWITCH_TRUE
#elif defined(DROPOUT_FALSE)
    #define DROPOUT_SWITCH BOOL_SWITCH_FALSE
#else
    #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef CAUSAL_TRUE
    #define CAUSAL_SWITCH BOOL_SWITCH_TRUE
#elif defined(CAUSAL_FALSE)
    #define CAUSAL_SWITCH BOOL_SWITCH_FALSE
#else
    #define CAUSAL_SWITCH BOOL_SWITCH
#endif

#ifdef SINK_TRUE
    #define SINK_SWITCH BOOL_SWITCH_TRUE
#elif defined(SINK_FALSE)
    #define SINK_SWITCH BOOL_SWITCH_FALSE
#else
    #define SINK_SWITCH BOOL_SWITCH
#endif

#if defined(XCORE1000)
  #define ARCH_SWITCH ARCH_SWITCH_XCORE1000
#elif defined(XCORE1500)
    #define ARCH_SWITCH ARCH_SWITCH_XCORE1500
#else
    #define ARCH_SWITCH ARCH_SWITCH_ALL
#endif

#define ARCH_SWITCH_ALL(ARCH, ARCH_NAME, ...)                                                    \
  [&] {                                                                                          \
    if (ARCH == 1000) {                                                                          \
      constexpr static Arch ARCH_NAME = Arch::xcore1000;                                         \
      return __VA_ARGS__();                                                                      \
    } else if (ARCH == 1500) {                                                                   \
      constexpr static Arch ARCH_NAME = Arch::xcore1500;                                         \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
       CHECK_MSG(false, "This arch xcore" + std::to_string(ARCH) +                               \
                        " is not supported, please check your arch!");                           \
    }                                                                                            \
  }()

#define ARCH_SWITCH_XCORE1000(ARCH, ARCH_NAME, ...)                                              \
  [&] {                                                                                          \
      constexpr static Arch ARCH_NAME = Arch::xcore1000;                                         \
      return __VA_ARGS__();                                                                      \
  }()

#define ARCH_SWITCH_XCORE1500(ARCH, ARCH_NAME, ...)                                              \
  [&] {                                                                                          \
      constexpr static Arch ARCH_NAME = Arch::xcore1500;                                         \
      return __VA_ARGS__();                                                                      \
  }()

#define COMBINE_BLOCKM_SWITCH(BATCH, HEADQ, SEQLENQ, kBlockM_min, CONST_NAME, ...)           \
  [&] {                                                                                      \
    const int max_block_num = (BATCH * HEADQ * SEQLENQ + kBlockM_min - 1) / kBlockM_min;     \
    if (max_block_num <= 512 || SEQLENQ <= kBlockM_min) {                                    \
        constexpr static int CONST_NAME = kBlockM_min;                                       \
        return __VA_ARGS__();                                                                \
    } else {                                                                                 \
        constexpr static int CONST_NAME = kBlockM_min * 2;                                   \
        return __VA_ARGS__();                                                                \
    }                                                                                        \
  }()
