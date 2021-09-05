import numpy as np
import scipy as sp
import scipy.sparse
import math
import sys
sys.path.append("..") # needed if pytest called from the parent directory
sys.path.insert(0, '../..') # needed if pytest called from this directory.

from bayesbridge import BayesBridge, RegressionModel, RegressionCoefPrior
from bayesbridge.model import LinearModel, LogisticModel, CoxModel

data_folder = 'saved_outputs'
test_combo = [
    ('linear', 'cg', 'dense', False),
    ('logit', 'cholesky', 'dense', False),
    ('logit', 'cholesky', 'dense', True),
    ('logit', 'cg', 'sparse', False),
    ('cox', 'hmc', 'sparse', False)
]

def test_gibbs(request):

    test_dirname = request.fspath.dirname
    for model, sampling_method, matrix_format, restart_im_middle in test_combo:
        samples = run_gibbs(model, sampling_method, matrix_format, restart_im_middle)
        assert is_same_as_prev_output(samples, sampling_method, model, test_dirname)

def run_gibbs(model_type, sampling_method, matrix_format, restart_in_middle=False):

    n_burnin = 0
    n_post_burnin = 10
    thin = 1
    bridge_exponent = 0.25

    outcome, X = simulate_data(model_type, matrix_format)
    prior = RegressionCoefPrior(
        sd_for_intercept=2., regularizing_slab_size=1.,
        bridge_exponent=bridge_exponent
    )
    model = RegressionModel(outcome, X, model_type)
    bridge = BayesBridge(model, prior)

    if restart_in_middle:
        n_total_post_burnin = n_post_burnin
        n_post_burnin = math.ceil(n_total_post_burnin / 2)

    mcmc_output = bridge.gibbs(
        n_burnin + n_post_burnin, n_burnin, thin=thin,
        coef_sampler_type=sampling_method, seed=0, params_to_save='all'
    )

    if restart_in_middle:
        reinit_bridge = BayesBridge(model, prior)
        mcmc_output = reinit_bridge.gibbs_additional_iter(
            mcmc_output, n_post_burnin, merge=True
        )

    return mcmc_output['samples']

def simulate_data(model, matrix_format):

    np.random.seed(1)
    n = 100
    p = 50

    # True parameters
    sigma_true = 2
    beta_true = np.zeros(p)
    beta_true[:4] = 1
    beta_true[4:15] = 2 ** - np.linspace(0.0, 5, 11)

    X = np.random.randn(n, p)

    if model == 'linear':
        outcome = LinearModel.simulate_outcome(X, beta_true, sigma_true)
    elif model == 'logit':
        n_trial = np.ones(n, dtype=np.int32)
        n_success = LogisticModel.simulate_outcome(n_trial, X, beta_true)
        outcome = (n_success, n_trial)
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
    np.save(filepath, samples['coef'][:, -1])

def is_same_as_prev_output(samples, sampling_method, model, test_dirname):
    prev_sample = load_data(sampling_method, model, test_dirname)
    return np.allclose(samples['coef'][:, -1], prev_sample, rtol=.001, atol=10e-6)


if __name__ == '__main__':
    option = sys.argv[-1]
    if option == 'update':
        for model, sampling_method, matrix_format, restart_im_middle in test_combo:
            samples = run_gibbs(model, sampling_method, matrix_format, restart_im_middle)
            save_data(samples, sampling_method, model)