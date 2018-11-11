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
    y, X, beta = simulate_data(model='logit', seed=0)
    n_trial, n_success = y
    logit_model = LogisticModel(n_success, X, n_trial)
    f = logit_model.compute_loglik_and_gradient
    assert numerical_grad_is_close(f, beta)


def test_logitstic_model_hessian_matvec():
    y, X, beta = simulate_data(model='logit', seed=0)
    n_trial, n_success = y
    logit_model = LogisticModel(n_success, X, n_trial)
    f = logit_model.compute_loglik_and_gradient
    hessian_matvec = logit_model.get_hessian_matvec_operator(beta)
    assert numerical_direc_deriv_is_close(f, beta, hessian_matvec, seed=0)


def set_up_cox_model_test(seed=0):
    y, X, beta = simulate_data(model='cox', seed=seed)
    event_order, censoring_time = y
    cox_model = CoxModel(event_order, censoring_time, X)
    return cox_model, beta


def test_cox_model_observation_reordering_and_risk_set_counting():

    event_time = np.array(
        [1, 5, np.inf, 2.5, 2.5, np.inf, 2]
    )
    censoring_time = np.array(
        [np.inf, np.inf, 3, np.inf, np.inf, 2, np.inf]
    )
    X = np.arange(len(event_time))[:, np.newaxis]
    event_time, censoring_time, X = \
        CoxModel.permute_observations_by_event_and_censoring_time(
            event_time, censoring_time, X
        )
    assert np.all(
        event_time == np.array([1, 2, 2.5, 2.5, 5, np.inf, np.inf])
    )
    assert np.all(
        censoring_time == np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 3, 2])
    )
    assert np.all(X == np.array([0, 6, 3, 4, 1, 2, 5])[:, np.newaxis])

    cox_model = CoxModel(event_time, censoring_time, X)
    n_censored_before_event = np.array([0, 0, 1, 1, 2])
    assert np.all(
        cox_model.risk_set_end_index \
            == len(event_time) - 1 - n_censored_before_event
    )
    assert np.all(
        cox_model.n_appearance_in_risk_set == np.array([1, 2, 4, 4, 5, 4, 2])
    ) # Tied events are both considered to be in the risk set.


def test_cox_model_sum_over_risk_set():

    cox_model, beta = set_up_cox_model_test()
    _, hazard_increase, sum_over_risk_set \
        = cox_model._compute_relative_hazard(beta)
    hazard_matrix = cox_model._HazardMultinomialProbMatrix(
        hazard_increase, sum_over_risk_set,
        cox_model.risk_set_end_index, cox_model.n_appearance_in_risk_set
    )
    assert np.allclose(
        hazard_matrix.sum_over_events(),
        np.sum(hazard_matrix.compute_matrix(), 0)
    )


def test_cox_model_gradient():
    cox_model, beta = set_up_cox_model_test()
    f = cox_model.compute_loglik_and_gradient
    assert numerical_grad_is_close(f, beta)


def test_cox_model_hessian_matvec():
    cox_model, beta = set_up_cox_model_test()
    f = cox_model.compute_loglik_and_gradient
    hessian_matvec = cox_model.get_hessian_matvec_operator(beta)
    assert numerical_direc_deriv_is_close(f, beta, hessian_matvec, seed=0)


def simulate_data(model, n_obs=100, n_pred=50, seed=None):

    if seed is not None:
        np.random.seed(seed)
        
    X = simulate_design(n_obs, n_pred, binary_frac=.9)

    beta = np.random.randn(n_pred)
    n_trial = None
    if model == 'logit':
        n_trial = np.random.binomial(np.arange(n_obs) + 1, .5)
        n_success = LogisticModel.simulate_outcome(n_trial, X, beta)
        y = (n_trial, n_success)
    elif model == 'cox':
        event_time, censoring_time = CoxModel.simulate_outcome(X, beta)
        event_time, censoring_time, X = \
            CoxModel.permute_observations_by_event_and_censoring_time(event_time, censoring_time, X)
        n_appearance = \
            CoxModel.count_risk_set_appearance(event_time, censoring_time)
        event_time = event_time[n_appearance >= 1]
        censoring_time = censoring_time[n_appearance >= 1]
        X = X[n_appearance >= 1, :]
        y = (event_time, censoring_time)
    else:
        raise NotImplementedError()

    if sp.sparse.issparse(X):
        X = SparseDesignMatrix(X)
    else:
        X = DenseDesignMatrix(X)

    return y, X, beta


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


def numerical_direc_deriv_is_close(
        f, x, hess_matvec, n_direction=10,
        atol=10E-6, rtol=10E-6, dx=10E-6, seed=None):
    """
    Compare analytically computed directional derivatives of the gradient of 'f'
    (i.e. the Hessian of 'f' applied to vectors) to its numerical approximations.

    Params:
    -------
    f : callable
        Returns a value of a function and its gradient
    """

    x = np.array(x, ndmin=1)

    np.random.seed(seed)
    all_matched = True

    for i in range(n_direction):

        v = np.random.randn(len(x))
        v /= np.sqrt(np.sum(v ** 2))
        _, grad_minus = f(x - dx * v)
        _, grad_plus = f(x + dx * v)
        direc_deriv_est = (grad_plus - grad_minus) / (2 * dx)
        direc_deriv = hess_matvec(v)

        if not np.allclose(direc_deriv, direc_deriv_est, atol=atol, rtol=rtol):
            all_matched = False
            break

    return all_matched
