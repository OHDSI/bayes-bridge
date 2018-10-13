import sys
sys.path.append(".") # needed if pytest called from the parent directory
sys.path.append("..") # needed if pytest called from this directory.

import numpy as np
import numpy.random
import scipy as sp
import scipy.sparse
from simulate_data import simulate_design
from bayesbridge.model import LogisticModel, CoxModel
from bayesbridge.design_matrix import SparseDesignMatrix, DenseDesignMatrix


def test_logitstic_model_gradient():
    n_trial, y, X, beta = simulate_data(model='logit', seed=0)
    logit_model = LogisticModel(y, X, n_trial)
    f = logit_model.compute_loglik_and_gradient
    assert numerical_grad_is_close(f, beta)

def test_cox_model_gradient():
    _, y, X, beta = simulate_data(model='cox', seed=0)
    cox_model = CoxModel(y, X)
    f = cox_model.compute_loglik_and_gradient
    assert numerical_grad_is_close(f, beta)

def simulate_data(model, seed=None):

    np.random.seed(seed)
    n_obs, n_pred = (100, 50)
    X = simulate_design(n_obs, n_pred, binary_frac=.9)

    beta = np.random.randn(n_pred)
    n_trial = None
    if model == 'logit':
        n_trial = np.random.binomial(np.arange(n_obs) + 1, .5)
        y = LogisticModel.simulate_outcome(n_trial, X, beta)
    elif model == 'cox':
        y = CoxModel.simulate_outcome(X, beta)
        y, X = CoxModel.permute_observations_by_event_time(y, X)
    else:
        raise NotImplementedError()

    if sp.sparse.issparse(X):
        X = SparseDesignMatrix(X)
    else:
        X = DenseDesignMatrix(X)

    return n_trial, y, X, beta

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
