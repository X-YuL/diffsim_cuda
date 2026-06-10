from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Targets every major NVIDIA architecture from Pascal (2016) onward.
# PTX at compute_90 provides JIT forward-compatibility for future GPUs.
NVCC_GENCODE = [
    # "-gencode=arch=compute_60,code=sm_60",       # Pascal (GTX 10xx, Tesla P100)
    # "-gencode=arch=compute_61,code=sm_61",       # Pascal (GTX 10xx desktop)
    # "-gencode=arch=compute_70,code=sm_70",       # Volta (V100)
    # "-gencode=arch=compute_75,code=sm_75",       # Turing (RTX 20xx, T4)
    # "-gencode=arch=compute_80,code=sm_80",       # Ampere (A100)
    # "-gencode=arch=compute_86,code=sm_86",       # Ampere (RTX 30xx)
    "-gencode=arch=compute_89,code=sm_89",       # Ada Lovelace (RTX 40xx)
    # "-gencode=arch=compute_90,code=sm_90",       # Hopper (H100)
    # "-gencode=arch=compute_90,code=compute_90",  # PTX for future GPUs (JIT compiled)
]

setup(
    name="srbd_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="srbd_cuda_ext",
            sources=[
                "src/srbd_ext.cpp",
                "src/srbd_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",           # safe: SRBD is corrected by alpha-alignment each step
                    "--expt-relaxed-constexpr",
                ] + NVCC_GENCODE,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
