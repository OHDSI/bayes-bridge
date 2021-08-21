from setuptools import setup, find_packages
from distutils.extension import Extension
from distutils.command.build_ext import build_ext


class CustomBuildExtCommand(build_ext):
    """ build_ext command when numpy headers are needed. """
    def run(self):
        # Import numpy here, only when headers are needed
        import numpy as np
        self.include_dirs.append(np.get_include())
        build_ext.run(self)

ext_modules = [
    Extension(
        "bayesbridge.random.tilted_stable.tilted_stable",
        sources=["bayesbridge/random/tilted_stable/tilted_stable.c"]
    ),
    Extension(
        "bayesbridge.random.polya_gamma.polya_gamma",
        sources=["bayesbridge/random/polya_gamma/polya_gamma.c"]
    )
]

setup(
    name='bayesbridge',
    version='0.2.2',
    description=\
        'Generates posterior samples under Bayesian sparse regression based on '
        + 'the bridge prior using the CG-accelerated Gibbs sampler of Nishimura '
        + 'et. al. (2018). The linear and logistic model are currently supported.',
    url='https://github.com/aki-nishimura/bayes-bridge',
    author='Akihiko (Aki) Nishimura',
    author_email='aki.nishimura@jhu.edu',
    license='MIT',
    packages=find_packages(exclude=['tests', 'tests.*']),
    cmdclass = {'build_ext': CustomBuildExtCommand},
    ext_modules = ext_modules,
    setup_requires=['numpy'],
    install_requires=[
        'numpy', 'scipy'
    ],
    zip_safe=False
)
