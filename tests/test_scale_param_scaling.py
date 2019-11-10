import sys
sys.path.append(".") # needed if pytest called from the parent directory
sys.path.append("..") # needed if pytest called from this directory.

import numpy as np
from .helper import simulate_data
from bayesbridge.model import LinearModel, LogisticModel, CoxModel
from bayesbridge import BayesBridge


def test_scale_param_scaling():

    y, X, beta = simulate_data(model='logit', seed=0)
    bridge_exp = .25

    bridge = BayesBridge(
        y, X, model='logit',
        add_intercept=True,
        regularizing_slab_size=1.,
        global_scale_parametrization='raw'
    )

    # Two samples should agree since the default prior is scale invariant.
    coef_sample_raw_scaling = get_last_sample_from_gibbs(bridge, bridge_exp)
    bridge.global_scale_parametrization = 'coefficient'
    coef_sample_expected_mag_scaling = get_last_sample_from_gibbs(bridge, bridge_exp)

    assert np.allclose(
        coef_sample_raw_scaling,
        coef_sample_expected_mag_scaling,
        rtol=1e-10
    )

    # Place a prior on the global scale; the two samples should *not* coincide.
    bridge_magnitude = bridge.compute_power_exp_ave_magnitude(bridge_exp)
    gscale_hyper_param = {
        'log10_mean': -2. - np.log10(bridge_magnitude),
        'log10_sd': 1.,
    }
    bridge.global_scale_parametrization = 'raw'
    coef_sample_raw_scaling \
        = get_last_sample_from_gibbs(bridge, bridge_exp, gscale_hyper_param)
    bridge.global_scale_parametrization = 'coefficient'
    coef_sample_expected_mag_scaling \
        = get_last_sample_from_gibbs(bridge, bridge_exp, gscale_hyper_param)

    assert not np.allclose(
        coef_sample_raw_scaling,
        coef_sample_expected_mag_scaling,
        rtol=1e-10
    )

    # After appropriately adjusting the hyper-parameter, the two samples
    # should agree.
    gscale_hyper_param['log10_mean'] += np.log10(bridge_magnitude)
    coef_sample_expected_mag_scaling \
        = get_last_sample_from_gibbs(bridge, bridge_exp, gscale_hyper_param)

    assert np.allclose(
        coef_sample_raw_scaling,
        coef_sample_expected_mag_scaling,
        rtol=1e-10
    )


def get_last_sample_from_gibbs(bridge, exponent, hyper_param=None, seed=0):
    mcmc_output = bridge.gibbs(
        n_burnin=0, n_post_burnin=10,
        sampling_method='direct',
        bridge_exponent=exponent,
        init={'apriori_coef_scale': .1},
        global_scale_prior_hyper_param=hyper_param,
        seed=seed, n_status_update=0
    )
    return mcmc_output['samples']['beta'][:, -1]