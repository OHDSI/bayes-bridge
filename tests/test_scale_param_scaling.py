import sys
sys.path.append(".") # needed if pytest called from the parent directory
sys.path.append("..") # needed if pytest called from this directory.

import numpy as np
from .helper import simulate_data
from bayesbridge.model import LinearModel, LogisticModel, CoxModel
from bayesbridge import BayesBridge, RegressionModel, RegressionCoefPrior


def test_scale_param_scaling():
    """ Check sampler outputs are invariant under global scale parametrization. """

    y, X, beta = simulate_data(model='logit', seed=0)
    model = RegressionModel(y, X, family='logit')
    bridge_exp = .25

    # Two samples should agree since the default prior is scale invariant.
    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        global_scale_parametrization='raw'
    )
    bridge = BayesBridge(model, prior)
    coef_sample_raw_scaling = get_last_sample_from_gibbs(bridge, bridge_exp)

    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        global_scale_parametrization='regress_coef'
    )
    bridge = BayesBridge(model, prior)
    coef_sample_expected_mag_scaling = get_last_sample_from_gibbs(bridge, bridge_exp)

    assert np.allclose(
        coef_sample_raw_scaling,
        coef_sample_expected_mag_scaling,
        rtol=1e-10
    )

    # Place a prior on the global scale; the two samples should *not* coincide.
    bridge_magnitude \
        = RegressionCoefPrior.compute_power_exp_ave_magnitude(bridge_exp)
    gscale_hyper_param = {
        'log10_mean': -2. - np.log10(bridge_magnitude),
        'log10_sd': 1.,
    }
    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        global_scale_prior_hyper_param=gscale_hyper_param,
        global_scale_parametrization='raw'
    )
    bridge = BayesBridge(model, prior)
    coef_sample_raw_scaling \
        = get_last_sample_from_gibbs(bridge, bridge_exp)

    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        global_scale_prior_hyper_param=gscale_hyper_param,
        global_scale_parametrization='regress_coef'
    )
    bridge = BayesBridge(model, prior)
    coef_sample_expected_mag_scaling \
        = get_last_sample_from_gibbs(bridge, bridge_exp)

    assert not np.allclose(
        coef_sample_raw_scaling,
        coef_sample_expected_mag_scaling,
        rtol=1e-10
    )

    # After appropriately adjusting the hyper-parameter, the two samples
    # should agree.
    gscale_hyper_param['log10_mean'] += np.log10(bridge_magnitude)
    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        global_scale_prior_hyper_param=gscale_hyper_param,
        global_scale_parametrization='regress_coef'
    )
    bridge = BayesBridge(model, prior)
    coef_sample_expected_mag_scaling \
        = get_last_sample_from_gibbs(bridge, bridge_exp)

    assert np.allclose(
        coef_sample_raw_scaling,
        coef_sample_expected_mag_scaling,
        rtol=1e-10
    )


def get_last_sample_from_gibbs(bridge, exponent, seed=0):
    mcmc_output = bridge.gibbs(
        n_burnin=0, n_post_burnin=10,
        sampling_method='direct',
        bridge_exponent=exponent,
        init={'apriori_coef_scale': .1},
        seed=seed, n_status_update=0
    )
    return mcmc_output['samples']['regress_coef'][:, -1]