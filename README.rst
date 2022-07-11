BayesBridge
===========

Python package for Bayesian sparse regression, implementing the standard (Polya-Gamma augmented) Gibbs sampler as well as the CG-accelerated sampler of Nishimura and Suchard (2022). The latter algorithm can be orders of magnitudes faster for a large and sparse design matrix.

Installation
------------
.. code-block:: bash

    pip install bayesbridge

Background
----------
The Bayesian bridge is based on the following prior on the regression coefficients :math:`\beta_j`'s:

..
    .. math::
        \pi(\beta_j \, | \, \tau) \propto \tau^{-1} \exp \big(-|\beta_j / \tau|^\alpha \big) \ \text{ for } \ 0 < \alpha \leq 1

.. raw:: html

    <img src="https://latex.codecogs.com/gif.latex?\pi(\beta_j&space;\,&space;|&space;\,&space;\tau)&space;\propto&space;\tau^{-1}&space;\exp&space;\big(-|\beta_j&space;/&space;\tau|^\alpha&space;\big)&space;\&space;\text{&space;for&space;}&space;\&space;0&space;<&space;\alpha&space;\leq&space;1." title="\pi(\beta_j \, | \, \tau) \propto \tau^{-1} \exp \big(-|\beta_j / \tau|^\alpha \big) \ \text{ for } \ 0 < \alpha \leq 1." />

The Bayesian bridge recovers the the Bayesian lasso when :math:`\alpha = 1` but can provide an improved separation of the significant coefficients from the rest when :math:`\alpha < 1`.

Usage
-----

.. code-block:: python

    from bayesbridge import BayesBridge, RegressionModel, RegressionCoefPrior

    model = RegressionModel(y, X, family='logit')
    prior = RegressionCoefPrior(bridge_exponent=.5)
    bridge = BayesBridge(model, prior)
    samples, mcmc_info = bridge.gibbs(
        n_burnin=100, n_post_burnin=1000, thin=1,
        coef_sampler_type='cholesky' # Try 'cg' for large and sparse X
    )
    coef_samples = samples['coef']

where `y` is a 1-D numpy array and `X` is a 2-D numpy array or scipy sparse matrix.

Currently the linear and logistic model (binomial outcomes) are supported. See `demo.ipynb` for demonstration of further features.

Citation
--------
If you find this package useful, please consider citing:

    Akihiko Nishimura and Marc A. Suchard (2022).
    Prior-preconditioned conjugate gradient method for accelerated Gibbs sampling in "large *n*, large *p*" Bayesian sparse regression. *Journal of the American Statistical Association*.

    Akihiko Nishimura and Marc A. Suchard (2022).
    Shrinkage with shrunken shoulders: Gibbs sampling shrinkage model posteriors with guaranteed convergence rates. *Bayesian Analysis*.