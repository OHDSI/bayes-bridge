# BayesBridge

Python package for Bayesian sparse regression, implementing the standard (Polya-Gamma augmented) Gibbs sampler as well as the CG-accelerated sampler of Nishimura and Suchard (2018). The latter algorithm can be orders of magnitudes faster for a large and sparse design matrix.

### Installation
```bash
    pip install bayesbridge
```

### Background
The Bayesian bridge is based on the following prior on the regression coefficients $\beta_j$'s:
<!---
$$\pi(\beta_j \, | \, \tau) \propto \tau^{-1} \exp \big(-|\beta_j / \tau|^\alpha \big) \ \text{ for } \ 0 < \alpha \leq 1$$
--->
<img src="https://latex.codecogs.com/gif.latex?\pi(\beta_j&space;\,&space;|&space;\,&space;\tau)&space;\propto&space;\tau^{-1}&space;\exp&space;\big(-|\beta_j&space;/&space;\tau|^\alpha&space;\big)&space;\&space;\text{&space;for&space;}&space;\&space;0&space;<&space;\alpha&space;\leq&space;1" title="\pi(\beta_j \, | \, \tau) \propto \tau^{-1} \exp \big(-|\beta_j / \tau|^\alpha \big) \ \text{ for } \ 0 < \alpha \leq 1" />
The Bayesian bridge recovers the the Bayesian lasso when $\alpha = 1$ but can provide an improved separation of the significant coefficients from the rest when $\alpha < 1$.

### Usage

```python
    bridge = BayesBridge(y, X, model='logit')
    mcmc_output = bridge.gibbs(
        n_burnin=100, n_post_burnin=300, thin=1,
        mvnorm_method='direct' # try the option 'cg' if X is large and sparse.
    )
    samples = mcmc_output['samples']
```

where `y` is a 1-D numpy array and `X` is a 2-D numpy array or scipy sparse matrix.

Currently the linear and logistic model (binomial outcomes) are supported. See `bayesbridge_demo.ipynb` for demonstration of further features.

### Citation
If you find this package useful, please cite
> Akihiko Nishimura and Marc A. Suchard (2018).
> Prior-preconditioned conjugate gradient for accelerated Gibbs sampling in "large n & large p" sparse Bayesian logistic regression models. arXiv:1810.12437.
