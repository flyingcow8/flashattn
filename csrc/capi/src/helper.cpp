#include "flash_attn.h"

inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});

    if (max_splits < 64 || batch_nheads_mblocks / 64 > 10) {
        return 1;
    }
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);

            if (eff > max_efficiency) {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 2; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            continue;
        }
        if (efficiency[num_splits - 1] >= 0.96 * max_efficiency) {
            return num_splits;
        }
    }
    return 1;
}

#ifdef __cplusplus
extern "C" {
#endif

int compute_num_splits(int batch_size, int num_heads, int head_size, int max_seqlen_k, int max_seqlen_q) {
    const int block_n = 32;
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;

    const int num_m_blocks = (max_seqlen_q + 32 - 1) / 32;

    int deviceId{};
    mcGetDevice(&deviceId);
    mcDeviceProp_t dprops;
    mcGetDeviceProperties(&dprops, deviceId);

    return num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops.multiProcessorCount, num_n_blocks, 128);
}

int check_seqlenq_ngroups_swapped(int seqlen_q, int num_heads, int num_heads_k, int window_size_left,
                                  int window_size_right, int head_size_og, Tensor_t alibi_slopes_, float p_droput) {
    bool seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 &&
                                   window_size_right < 0 && head_size_og % 8 == 0 && alibi_slopes_ == nullptr;

    return seqlenq_ngroups_swapped;
}

void release_extend_param(mcflashattnExtendParameter_t extend_param) {
    if (extend_param) {
        if (extend_param->data) {
            free(extend_param->data);
        }
        free(extend_param);
    }
}

int head_size_pad(int head_size_og) {
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    if (head_size_og % 8 != 0) return round_multiple(head_size_og, 8);
    return head_size_og;
}

#ifdef __cplusplus
}
#endif