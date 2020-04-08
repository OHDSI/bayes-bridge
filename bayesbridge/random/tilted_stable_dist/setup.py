from distutils.core import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "rand_exp_tilted_stable",
        ["rand_exp_tilted_stable.pyx"]
    )
]

setup(
    ext_modules = cythonize(ext_modules)
)
