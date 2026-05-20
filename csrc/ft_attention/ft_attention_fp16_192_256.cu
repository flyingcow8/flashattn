#include "ft_attention_launch_template.h"

template<>
void run_multihead_attention_<192,256,Masked_multihead_attention_params<uint16_t>>(const Masked_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream){
    mmha_launch_kernel<uint16_t, 192, 256, Masked_multihead_attention_params<uint16_t>>(params, stream);
}
