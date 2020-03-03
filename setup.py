from setuptools import setup, find_packages
from distutils.extension import Extension

setup(
    name='bayesbridge',
    version='0.1',
    description=\
        'Generates posterior samples under Bayesian sparse regression based on '
        + 'the bridge prior using the CG-accelerated Gibbs sampler of Nishimura '
        + 'et. al. (2018). The linear and logistic model are currently supported.',
    url='https://github.com/aki-nishimura/bayes-bridge',
    author='Akihiko (Aki) Nishimura',
    author_email='akihiko4@g.ucla.edu',
    license='MIT',
    packages=find_packages(exclude=['./tests/*']),
    ext_modules = [
        Extension(
            "bayesbridge.random.tilted_stable_dist.custom_math",
            ["bayesbridge/random/tilted_stable_dist/custom_math.c"]
        )
    ],
    install_requires=[
        'numpy', 'scipy', 'pypolyagamma'
    ],
    zip_safe=False
)
