import numpy as np
import scipy as sp
from bayesbridge.model import LinearModel, LogisticModel, CoxModel
from bayesbridge.design_matrix import SparseDesignMatrix, DenseDesignMatrix
from simulate_data import simulate_design


def simulate_data(model, n_obs=100, n_pred=50, seed=None,
                  return_design_mat=False):
    if seed is not None:
        np.random.seed(seed)

    X = simulate_design(n_obs, n_pred, binary_frac=.9)

    beta = np.random.randn(n_pred)
    n_trial = None
    if model == 'linear':
        y = LinearModel.simulate_outcome(X, beta, noise_sd=1.)
    elif model == 'logit':
        n_trial = 1 + np.random.binomial(np.arange(n_obs) + 1, .5)
        n_success = LogisticModel.simulate_outcome(n_trial, X, beta)
        y = (n_success, n_trial)
    elif model == 'cox':
        event_time, censoring_time = CoxModel.simulate_outcome(X, beta)
        event_time, censoring_time, X = \
            CoxModel._permute_observations_by_event_and_censoring_time(
                event_time, censoring_time, X)
        event_time, censoring_time, X = \
            CoxModel._drop_uninformative_observations(event_time,
                                                      censoring_time, X)
        y = (event_time, censoring_time)
    else:
        raise NotImplementedError()

    if return_design_mat:
        if sp.sparse.issparse(X):
            X = SparseDesignMatrix(X, add_intercept=False)
        else:
            X = DenseDesignMatrix(X, add_intercept=False)

    return y, X, beta