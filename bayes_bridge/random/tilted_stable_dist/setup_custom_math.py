from distutils.core import setup, Extension
from Cython.Build import cythonize
import subprocess
import numpy as np

# Hack to include the numpy header file.
cmd = 'export CFLAGS="-I ' + np.get_include() + ' $CFLAGS"'
subprocess.run(cmd, shell=True, check=True)

ext_modules = [
    Extension(
        "custom_math",
        ["custom_math.pyx"],
        libraries=["m"],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules = cythonize(ext_modules)
)
