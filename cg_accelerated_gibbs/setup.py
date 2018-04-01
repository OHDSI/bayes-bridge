from distutils.core import setup, Extension
from Cython.Build import cythonize
import subprocess
import os
import numpy as np

# Hack to include the numpy header file.
cmd = 'export CFLAGS="-I ' + np.get_include() + ' $CFLAGS"'
subprocess.run(cmd, shell=True, check=True)
os.environ["CC"] = "clang++ -Xpreprocessor -fopenmp -lomp" # "gcc-6 -fopenmp"

ext_modules = [
    Extension(
        "binary_matmul",
        ["binary_matmul.pyx"],
#        extra_compile_args=['-Xpreprocessor -fopenmp -lomp'],
#        extra_link_args=['-Xpreprocessor -fopenmp -lomp'],
    )
]

setup(
    ext_modules = cythonize(ext_modules)
)
