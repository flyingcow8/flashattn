#include "py_export_capi_inference.h"
#include "py_export_capi_training.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // infer
    m.def("fwd_inference", &mha_fwd_inference_test, "Forward pass");
    m.def("varlen_fwd_inference", &mha_varlen_fwd_inference_test, "Forward pass (variable length)");
    m.def("fwd_kvcache", &mha_fwd_kvcache_test, "Forward pass, with KV-cache");

    // training
    m.def("fwd", &mha_fwd_capi_test, "cpai Forward pass test");
    m.def("varlen_fwd", &mha_varlen_fwd_capi_test, "cpai Forward pass (variable length)");
    m.def("bwd", &mha_bwd_capi_test, "cpai Backward pass");
    m.def("varlen_bwd", &mha_varlen_bwd_capi_test, "cpai Backward pass (variable length)");
}