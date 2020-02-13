from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from os.path import join

project_root = 'PCSA_Module'
sources = [join(project_root, file) for file in ['sa_ext.cpp',
                                                 'sa.cu','reference.cpp']]
'''
with open("README.md", "r") as fh:
    long_description = fh.read()
'''

nvcc_args = [
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]
cxx_args = ['-std=c++11']

setup(
    name='self_cuda',
    #version="0.1.0",
    #author="Clément Pinard",
    #author_email="clement.pinard@ensta-paristech.fr",
    #description="Correlation module for pytorch",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    #url="https://github.com/ClementPinard/Pytorch-Correlation-extension",
    #install_requires=['torch>=1.0.1','numpy'],
    ext_modules=[
        CUDAExtension('self_cuda_backend',
                      sources, extra_compile_args={'cxx': cxx_args,'nvcc': nvcc_args})
    ],
    #package_dir={'': project_root},
    #packages=['spatial_correlation_sampler'],
    cmdclass={
        'build_ext': BuildExtension
    })
