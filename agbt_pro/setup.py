# setup file
# run: python setup.py build_ext --inplace


from distutils.core import setup, Extension
import sys
import numpy as np
from Cython.Build import cythonize


if sys.platform == 'darwin':
    extra_compile_args = ['-stdlib=libc++', '-O3']
else:
    extra_compile_args = ['-std=c++11', '-O3']

extensions = [
    Extension(
        'fairseq.libbleu',
        sources=[
            'fairseq/clib/libbleu/libbleu.cpp',
            'fairseq/clib/libbleu/module.cpp',
        ],
        extra_compile_args=extra_compile_args
    )
]

extensions = extensions \
    + cythonize('fairseq/data/data_utils_fast.pyx') \
        + cythonize('fairseq/data/token_block_utils_fast.pyx')


setup(
    name="necessary_modules",
    ext_modules=extensions,
    include_dirs=[np.get_include()]*3,
)
