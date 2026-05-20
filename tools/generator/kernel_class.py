import kernel_impl_template as temps
from bool_switch_process import construct_bool_config, generate_bool_macros

SM = [80]
DTYPE_MAP = {
    "bf16": "bfloat16",
    "fp16": "float16",
}

class Kernel:
    _id_counter = 0

    def __init__(self, config, api):
        self.hdim_qk = config["hdim_qk"]
        self.hdim_v = config["hdim_v"]
        self.block_m = config["block_m"]
        self.block_n = config["block_n"]
        self.k_nwarps = config["k_nwarps"]
        self.dtype = "mctlass::"+ ("half_t" if config["dtype"] == "float16" else "bfloat16_t")
        self.dtype_short = next((key for key, value in DTYPE_MAP.items() if value == config["dtype"]),  None )

        self.api = api
        self.id = Kernel._id_counter
        Kernel._id_counter += 1

    @property
    def template(self) -> str:
        raise NotImplementedError("Subclasses should implement this method")

class BwdKernel(Kernel):
    def __init__(self, config, arch):
        super().__init__(config, "bwd")
        self.atom_layout_msdp = config["atom_layout_msdp"]
        self.atom_layout_ndkv = config["atom_layout_ndkv"]
        self.atom_layout_mdq = config["atom_layout_mdq"]
        self.is_v_in_regs = config["is_v_in_regs"]
        self.is_k_in_regs = config["is_k_in_regs"]
        self.no_double_buffer = config["no_double_buffer"]
        self.name=self.kernel_name
        self.arch = arch

    @property
    def template(self) -> str:

        valid_bool_configs = construct_bool_config("bwd",self.hdim_qk)
        macros = generate_bool_macros(valid_bool_configs)
        cpp_macros = [f"#define {item}" for item in macros]
        defined_macros_str = "\n".join(cpp_macros)

        KERNEL_IMPL_TEMPLATE_BWD = temps.KERNEL_IMPL_TEMPLATE_BWD_XCORE1000 if self.arch == 'xcore1000' else temps.KERNEL_IMPL_TEMPLATE_BWD_XCORE1500

        return KERNEL_IMPL_TEMPLATE_BWD.format(
            BOOL_SWITCH_MACROS = defined_macros_str,
            HEAD_DIM_QK=self.hdim_qk,
            HEAD_DIM_V=self.hdim_v,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            K_NWARPS=self.k_nwarps,
            ATOM_LAYOUT_MSDP=self.atom_layout_msdp,
            ATOM_LAYOUT_NDKV=self.atom_layout_ndkv,
            ATOM_LAYOUT_MDQ=self.atom_layout_mdq,
            IS_V_IN_REGS=str(self.is_v_in_regs).lower(),
            IS_K_IN_REGS=str(self.is_k_in_regs).lower(),
            NO_DOUBLE_BUFFER=str(self.no_double_buffer).lower(),
            DTYPE=self.dtype
        )

    @property
    def filename(self) -> str:
        return f"flash_bwd_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_{self.dtype_short}_ \
        {self.is_v_in_regs}_{self.is_k_in_regs}_{self.no_double_buffer}_sm80"

    @property
    def kernel_name(self) -> str:
        return f"flash_bwd_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_ \
        {self.k_nwarps}wave_{self.atom_layout_msdp}x{self.atom_layout_ndkv}x{self.atom_layout_mdq}_ \
        {self.is_v_in_regs}_{self.is_k_in_regs}_{self.dtype_short}"

class FwdKernel(Kernel):
    def __init__(self, config, arch):
        super().__init__(config, "fwd")
        self.Is_Q_in_regs_ = config["Is_Q_in_regs"]
        self.Share_Q_K_smem_ = config["Share_Q_K_smem"]
        self.name=self.kernel_name
        self.arch = arch

    @property
    def template(self) -> str:

        valid_bool_configs = construct_bool_config("fwd",self.hdim_qk)
        macros = generate_bool_macros(valid_bool_configs)
        cpp_macros = [f"#define {item}" for item in macros]
        defined_macros_str = "\n".join(cpp_macros)
        KERNEL_IMPL_TEMPLATE_FWD = temps.KERNEL_IMPL_TEMPLATE_FWD_XCORE1000 if self.arch == 'xcore1000' else temps.KERNEL_IMPL_TEMPLATE_FWD_XCORE1500
        return KERNEL_IMPL_TEMPLATE_FWD.format(
            BOOL_SWITCH_MACROS = defined_macros_str,
            HEAD_DIM_QK=self.hdim_qk,
            HEAD_DIM_V=self.hdim_v,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            K_NWARPS=self.k_nwarps,
            Is_Q_in_regs_=str(self.Is_Q_in_regs_).lower(),
            Share_Q_K_smem_=str(self.Share_Q_K_smem_).lower(),
            DTYPE=self.dtype
        )

    @property
    def filename(self) -> str:
        return f"flash_fwd_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_{self.dtype_short}_ \
        {self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_sm80"

    @property
    def kernel_name(self) -> str:
        return f"flash_fwd_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
        {self.k_nwarps}wave_{self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}_"

class FwdSplitKernel(Kernel):
    def __init__(self, config, arch):
        super().__init__(config, "fwd_splitkv")
        self.Is_Q_in_regs_ = config["Is_Q_in_regs"]
        self.Share_Q_K_smem_ = config["Share_Q_K_smem"]
        self.name=self.kernel_name
        self.arch = arch

    @property
    def template(self) -> str:

        valid_bool_configs = construct_bool_config("fwd_split",self.hdim_qk)
        macros = generate_bool_macros(valid_bool_configs)
        cpp_macros = [f"#define {item}" for item in macros]
        defined_macros_str = "\n".join(cpp_macros)
        KERNEL_IMPL_TEMPLATE_FWD_SPLITKV = temps.KERNEL_IMPL_TEMPLATE_FWD_SPLITKV_XCORE1000 if self.arch == 'xcore1000' else temps.KERNEL_IMPL_TEMPLATE_FWD_SPLITKV_XCORE1500
        return KERNEL_IMPL_TEMPLATE_FWD_SPLITKV.format(
            BOOL_SWITCH_MACROS = defined_macros_str,
            HEAD_DIM_QK=self.hdim_qk,
            HEAD_DIM_V=self.hdim_v,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            K_NWARPS=self.k_nwarps,
            Is_Q_in_regs_=str(self.Is_Q_in_regs_).lower(),
            Share_Q_K_smem_=str(self.Share_Q_K_smem_).lower(),
            DTYPE=self.dtype
        )

    @property
    def filename(self) -> str:
        return f"flash_fwd_splitkv_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_{self.dtype_short}_\
        {self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_sm80"

    @property
    def kernel_name(self) -> str:
        return f"flash_fwd_splitkv_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_{self.k_nwarps}wave_ \
        {self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}_"
        # return f"flash_fwd_splitkv_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_{self.k_nwarps}wave_ \
        # {self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}"
