import sys
sys.path.append(".") # needed if pytest called from the parent directory
sys.path.append("..") # needed if pytest called from this directory.

import numpy as np
import numpy.random
import scipy as sp
import scipy.sparse
from functools import partial
from .derivative_tester \
    import numerical_grad_is_close, numerical_direc_deriv_is_close
from .helper import simulate_data
from bayesbridge.model import LinearModel, LogisticModel, CoxModel


def test_linear_model_gradient_and_hessian():
    y, X, beta = simulate_data(model='linear', seed=0, return_design_mat=True)
    obs_prec = 1.
    linear_model = LinearModel(y, X)
    f = partial(linear_model.compute_loglik_and_gradient, obs_prec=obs_prec)
    hessian_matvec = linear_model.get_hessian_matvec_operator(beta, obs_prec)
    assert numerical_grad_is_close(f, beta)
    assert numerical_direc_deriv_is_close(f, beta, hessian_matvec, seed=0)


def test_logitstic_model_hessian_matvec():
    y, X, beta = simulate_data(model='logit', seed=0, return_design_mat=True)
    n_success, n_trial = y
    logit_model = LogisticModel(n_success, n_trial, X)
    f = logit_model.compute_loglik_and_gradient
    hessian_matvec = logit_model.get_hessian_matvec_operator(beta)
    assert numerical_direc_deriv_is_close(f, beta, hessian_matvec, seed=0)


def set_up_cox_model_test(seed=0):
    y, X, beta = simulate_data(model='cox', seed=seed, return_design_mat=True)
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
        CoxModel._permute_observations_by_event_and_censoring_time(
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
    assert np.all(
        cox_model.risk_set_start_index == np.array([0, 1, 2, 2, 4])
    )

    n_censored_before_event = np.array([0, 0, 1, 1, 2])
    assert np.all(
        cox_model.risk_set_end_index \
            == len(event_time) - 1 - n_censored_before_event
    )
    assert np.all(
        cox_model.n_appearance_in_risk_set == np.array([1, 2, 4, 4, 5, 4, 2])
    ) # Tied events are both considered to be in the risk set.


def test_cox_model_drop_uninformative():
    event_time = np.array(
        [2, 4, np.inf, np.inf]
    )
    censoring_time = np.array(
        [np.inf, np.inf, 3, 1]
    )
    X = np.arange(4)[:, np.newaxis]
    event_time, censoring_time, X = \
        CoxModel._drop_uninformative_observations(event_time, censoring_time, X)
    assert np.all(event_time == np.array([2, 4, np.inf]))
    assert np.all(censoring_time == np.array([np.inf, np.inf, 3]))
    assert np.all(X == np.array([0, 1, 2])[:, np.newaxis])


def test_cox_model_sum_over_risk_set():
    arr = np.array([1, 3, 2])
    start_index = np.array([0, 1])
    end_index = np.array([2, 1])
    assert np.all(
        CoxModel._sum_over_start_end(arr, start_index, end_index) == np.array([6, 3])
    )

def text_cox_model_sum_over_events():

    cox_model, beta = set_up_cox_model_test()
    _, hazard_increase, sum_over_risk_set \
        = cox_model._compute_relative_hazard(beta)
    hazard_matrix = cox_model._HazardMultinomialProbMatrix(
        hazard_increase, sum_over_risk_set,
        cox_model.risk_set_start_index,
        cox_model.risk_set_end_index,
        cox_model.n_appearance_in_risk_set
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