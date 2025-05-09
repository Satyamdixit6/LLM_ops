# llm_pipeline/adaptations/custom_cuda_kernel/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_cuda_kernels',
    ext_modules=[
        CUDAExtension(
            name='custom_cuda_kernels_cpp', # Must match TORCH_EXTENSION_NAME in .cu and import
            sources=['custom_kernel.cu'],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })