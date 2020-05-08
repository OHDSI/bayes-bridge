from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "tilted_stable",
        ["tilted_stable.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules = cythonize(ext_modules)
)
