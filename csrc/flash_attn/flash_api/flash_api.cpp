/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#include "flash_api.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";

    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
    m.def("fwd_kvcache_dequant", &mha_fwd_kvcache_dequant, "Forward pass, with quant KV-cache input");

}

#undef PY_DEF
