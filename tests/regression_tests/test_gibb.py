import numpy as np
import scipy as sp
import scipy.sparse
import math
import sys
sys.path.append("..") # needed if pytest called from the parent directory
sys.path.insert(0, '../..') # needed if pytest called from this directory.

from bayesbridge import BayesBridge, RegressionCoefPrior
from bayesbridge.model import CoxModel

data_folder = 'saved_outputs'
test_combo = [
    ('linear', 'cg', 'dense', False),
    ('logit', 'direct', 'dense', False),
    ('logit', 'direct', 'dense', True),
    ('logit', 'cg', 'sparse', False),
    ('cox', 'hmc', 'sparse', False)
]

def test_gibbs(request):

    test_dirname = request.fspath.dirname
    for model, sampling_method, matrix_format, restart_im_middle in test_combo:
        samples = run_gibbs(model, sampling_method, matrix_format, restart_im_middle)
        assert is_same_as_prev_output(samples, sampling_method, model, test_dirname)

def run_gibbs(model, sampling_method, matrix_format, restart_in_middle=False):

    n_burnin = 0
    n_post_burnin = 10
    thin = 1
    reg_exponent = 0.5

    outcome, X = simulate_data(model, matrix_format)
    n_unshrunk = 1 if model == 'cox' else 0
    prior = RegressionCoefPrior(global_scale_parametrization='raw')
    bridge = BayesBridge(
        outcome, X, model=model, prior=prior,
        n_coef_without_shrinkage=n_unshrunk, prior_sd_for_unshrunk=2.
    )
    init = {
        'global_scale': .01,
        'local_scale': np.ones(X.shape[1] - n_unshrunk)
    }

    if restart_in_middle:
        n_total_post_burnin = n_post_burnin
        n_post_burnin = math.ceil(n_total_post_burnin / 2)

    mcmc_output = bridge.gibbs(
        n_burnin, n_post_burnin, thin, reg_exponent, init,
        sampling_method=sampling_method, seed=0, params_to_save='all'
    )

    if restart_in_middle:
        reinit_bridge = BayesBridge(outcome, X, model=model)
        mcmc_output = reinit_bridge.gibbs_additional_iter(
            mcmc_output, n_total_post_burnin - n_post_burnin, merge=True
        )

    return mcmc_output['samples']

def simulate_data(model, matrix_format):

    np.random.seed(1)
    n = 500
    p = 500

    # True parameters
    sigma_true = 2
    beta_true = np.zeros(p)
    beta_true[:5] = 4
    beta_true[5:15] = 2 ** - np.linspace(0.0, 4.5, 10)

    X = np.random.randn(n, p)
    if model == 'linear':
        outcome = np.dot(X, beta_true) + sigma_true * np.random.randn(n)
    elif model == 'logit':
        mu = (1 + np.exp(- np.dot(X, beta_true))) ** -1
        outcome = (np.random.binomial(1, mu), None)
    elif model == 'cox':
        outcome = CoxModel.simulate_outcome(X, beta_true)
    else:
        raise NotImplementedError()

    if matrix_format == 'sparse':
        X = sp.sparse.csr_matrix(X)

    return outcome, X

def load_data(sampling_method, model, test_dirname):
    filepath = '/'.join([
        test_dirname, data_folder, get_filename(sampling_method, model)
    ])
    return np.load(filepath)

def get_filename(sampling_method, model):
    return '_'.join([
        model, sampling_method, 'samples.npy'
    ])

def save_data(samples, sampling_method, model):
    filepath = data_folder + '/' + get_filename(sampling_method, model)
    np.save(filepath, samples['regress_coef'])

def is_same_as_prev_output(samples, sampling_method, model, test_dirname):
    prev_sample = load_data(sampling_method, model, test_dirname)
    return np.allclose(samples['regress_coef'], prev_sample, rtol=.001, atol=10e-6)


if __name__ == '__main__':
    option = sys.argv[-1]
    if option == 'update':
        for model, sampling_method, matrix_format, restart_im_middle in test_combo:
            samples = run_gibbs(model, sampling_method, matrix_format, restart_im_middle)
            save_data(samples, sampling_method, model)