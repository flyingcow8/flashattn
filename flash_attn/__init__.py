__version__ = "2.5.3"

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

from flash_attn.flash_attn_inference_interface import (
    flash_attn_inference_func,
    flash_attn_inference_kvpacked_func,
    flash_attn_inference_qkvpacked_func,
    flash_attn_inference_varlen_func,
    flash_attn_inference_varlen_kvpacked_func,
    flash_attn_inference_varlen_qkvpacked_func,
)