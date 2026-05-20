#include "ft_attention_launch_template.h"

template<>
void run_multihead_attention_<256,256,Masked_multihead_attention_params<float>>(const Masked_multihead_attention_params<float>& params, const cudaStream_t& stream){
    mmha_launch_kernel<float, 256, 256, Masked_multihead_attention_params<float>>(params, stream);
}
