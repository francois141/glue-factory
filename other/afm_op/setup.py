from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='afm_op',
    ext_modules=[
        CUDAExtension(
            name='afm_op',  # submodule inside afm_op package
            sources=[
                'afm_cuda.cu',
                'binding.cpp',
            ],
            include_dirs=['.'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-DCUDA_HAS_FP16=1',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
