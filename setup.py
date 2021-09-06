import numpy as np
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
from numpy.distutils.misc_util import get_info
from os.path import dirname, join, abspath
from setuptools import setup, find_packages
from setuptools.extension import Extension

path = dirname(__file__)
src_dir = join(dirname(path), '..', 'src')
defs = [('NPY_NO_DEPRECATED_API', 0)]
inc_path = np.get_include()
lib_path = [abspath(join(np.get_include(), '..', '..', 'random', 'lib'))]
lib_path += get_info('npymath')['library_dirs']
np_libs = ['npyrandom', 'npymath']

class CustomBuildExtCommand(build_ext):
    """ build_ext command when numpy headers are needed. """
    def run(self):
        # Import numpy here, only when headers are needed
        import numpy as np
        self.include_dirs.append(np.get_include())
        build_ext.run(self)


ext_modules = [
    Extension(
        'bayesbridge.random.tilted_stable.tilted_stable',
        sources=['bayesbridge/random/tilted_stable/tilted_stable.pyx'],
    ),
    Extension(
        'bayesbridge.random.polya_gamma.polya_gamma',
        sources=['bayesbridge/random/polya_gamma/polya_gamma.pyx'],
    ),
    Extension(
        'bayesbridge.random.normal.normal',
        sources=['bayesbridge/random/normal/normal.pyx'],
        library_dirs=lib_path,
        libraries=np_libs,
        define_macros=defs,
    ),
    Extension(
        'bayesbridge.random.uniform.uniform',
        sources=['bayesbridge/random/uniform/uniform.pyx'],
        library_dirs=lib_path,
        libraries=np_libs,
        define_macros=defs,
    )
]

setup(
    name='bayesbridge',
    version='0.2.4',
    description=\
        'Generates posterior samples under Bayesian sparse regression based on '
        + 'the bridge prior using the CG-accelerated Gibbs sampler of Nishimura '
        + 'et. al. (2018). The linear and logistic model are currently supported.',
    url='https://github.com/aki-nishimura/bayes-bridge',
    author='Akihiko (Aki) Nishimura',
    author_email='aki.nishimura@jhu.edu',
    license='MIT',
    packages=find_packages(exclude=['tests', 'tests.*']),
    cmdclass={'build_ext': CustomBuildExtCommand},
    ext_modules=cythonize(ext_modules),
    setup_requires=['numpy>=1.19'],
    install_requires=[
        'numpy>=1.19', 'scipy'
    ],
    zip_safe=False
)
