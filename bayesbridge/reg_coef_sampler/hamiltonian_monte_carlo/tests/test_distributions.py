import numpy as np

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

def numerical_grad_is_close(f, x, atol=10E-6, rtol=10E-6, dx=10E-6):
    """
    Compare the computed gradient to a centered finite difference approximation.

    Params:
    -------
    f : callable
        Returns a value of a function and its gradient
    """
    x = np.array(x, ndmin=1)
    grad_est = np.zeros(len(x))
    for i in range(len(x)):
        x_minus = x.copy()
        x_minus[i] -= dx
        x_plus = x.copy()
        x_plus[i] += dx
        f_minus, _ = f(x_minus)
        f_plus, _ = f(x_plus)
        grad_est[i] = (f_plus - f_minus) / (2 * dx)

    _, grad = f(x)
    return np.allclose(grad, grad_est, atol=atol, rtol=rtol)