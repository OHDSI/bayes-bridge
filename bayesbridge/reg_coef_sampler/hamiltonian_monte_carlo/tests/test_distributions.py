import numpy as np

from tests.derivative_tester import numerical_grad_is_close
from .distributions import BivariateGaussian, BivariateSkewNormal

def test_bi_gaussian():
    x = np.ones(2)
    skewnorm = BivariateGaussian()
    f = skewnorm.compute_logp_and_gradient
    assert numerical_grad_is_close(f, x)

def test_skew_normal():
    x = np.ones(2)
    skewnorm = BivariateSkewNormal()
    f = skewnorm.compute_logp_and_gradient
    assert numerical_grad_is_close(f, x)