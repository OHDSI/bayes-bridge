"""Tests for GPU functionality. Depends on cupy being installed, so not run on CI."""
from bayesbridge import BayesBridge, RegressionModel, RegressionCoefPrior
import numpy as np
import pytest
pytest.importorskip("cupy")
import cupy as cp


from ..helper import simulate_data


pytestmark = pytest.mark.skipif(cp is None, reason="cupy unavailable")

@pytest.fixture
def bridge_gpu():
    y, X, beta = simulate_data(model='logit', seed=1)
    return BayesBridge(
        RegressionModel(y, cp.sparse.csr_matrix(X), 'logit'),
        RegressionCoefPrior())


@pytest.fixture
def bridge_cpu():
    y, X, beta = simulate_data(model='logit', seed=1)
    return BayesBridge(
        RegressionModel(y, X, 'logit'),
        RegressionCoefPrior()
    )


def test_use_cupy(bridge_cpu, bridge_gpu):
    """Test use_cupy attribute is set appropriately."""
    assert bridge_gpu.model.design.use_cupy
    assert not bridge_cpu.model.design.use_cupy


def test_similar_output(bridge_cpu, bridge_gpu):
    """Test that the CPU and GPU results are the same."""
    iters = 10
    seed = 1
    sampler = 'cg'
    init = {'coef': np.ones(bridge_gpu.model.n_pred)}
    samples_cpu, mcmc_info_cpu = bridge_cpu.gibbs(
        n_iter=iters, coef_sampler_type=sampler, init=init, seed=seed)
    samples_gpu, mcmc_info_gpu = bridge_gpu.gibbs(
        n_iter=iters, coef_sampler_type=sampler, init=init, seed=seed)
    assert np.allclose(samples_gpu['coef'], samples_cpu['coef'], atol=1e-5)


def test_preferred_sampler(bridge_gpu):
    """Test default sampler for cupy matrices is 'cg'."""
    samples_gpu, mcmc_info_gpu = bridge_gpu.gibbs(n_iter=1)
    assert mcmc_info_gpu['options']['coef_sampler_type'] == 'cg'


def test_unsupported_sampler(bridge_gpu):
    """Test non-'cg' samplers raise errors."""
    with pytest.raises(ValueError):
        bridge_gpu.gibbs(n_iter=1, coef_sampler_type='cholesky')
    with pytest.raises(ValueError):
        bridge_gpu.gibbs(n_iter=1, coef_sampler_type='hmc')
