from distutils.core import setup
from Cython.Build import cythonize
setup(
    name='_lda_module',
    ext_modules=cythonize('_lda.pyx')
)