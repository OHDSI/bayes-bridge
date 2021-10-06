import cupy as cp
import numpy as np
from bayesbridge import BayesBridge, RegressionModel, RegressionCoefPrior

from ..helper import simulate_data

def test_gpu_cpu_similar():
    """Test that the CPU and GPU results are the same"""
    model_type = 'logit'
    sampler = 'cg'
    iters = 10
    seed = 111
    y, X, beta = simulate_data(model=model_type, seed=seed)
    init = {'coef': np.ones(X.shape[1]+1)}
    prior = RegressionCoefPrior()
    model = RegressionModel(y, X, model_type)
    bridge = BayesBridge(model, prior)
    samples_cpu, mcmc_info_cpu = bridge.gibbs(n_iter=iters,
                                              coef_sampler_type=sampler,
                                              init=init,
                                              seed=seed)

    X = cp.sparse.csr_matrix(X)
    model = RegressionModel(y, X, model_type)
    bridge = BayesBridge(model, prior)
    samples_gpu, mcmc_info_gpu = bridge.gibbs(n_iter=iters,
                                              coef_sampler_type=sampler,
                                              init=init,
                                              seed=seed)
    assert np.allclose(samples_gpu['coef'], samples_cpu['coef'], atol=1e-5)


