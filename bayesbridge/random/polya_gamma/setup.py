from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "polya_gamma",
        ["polya_gamma.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules = cythonize(ext_modules)
)
