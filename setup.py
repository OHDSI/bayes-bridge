from setuptools import setup, find_packages
from distutils.extension import Extension
import numpy as np

ext_modules = [
    Extension(
        "bayesbridge.random.tilted_stable.tilted_stable",
        sources=["bayesbridge/random/tilted_stable/tilted_stable.c"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "bayesbridge.random.polya_gamma.polya_gamma",
        sources=["bayesbridge/random/polya_gamma/polya_gamma.c"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name='bayesbridge',
    version='0.2.0',
    description=\
        'Generates posterior samples under Bayesian sparse regression based on '
        + 'the bridge prior using the CG-accelerated Gibbs sampler of Nishimura '
        + 'et. al. (2018). The linear and logistic model are currently supported.',
    url='https://github.com/aki-nishimura/bayes-bridge',
    author='Akihiko (Aki) Nishimura',
    author_email='akihiko4@g.ucla.edu',
    license='MIT',
    packages=find_packages(exclude=['tests', 'tests.*']),
    ext_modules = ext_modules,
    install_requires=[
        'numpy', 'scipy'
    ],
    zip_safe=False
)
