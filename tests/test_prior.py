import sys
sys.path.append(".") # needed if pytest called from the parent directory
sys.path.append("..") # needed if pytest called from this directory.

import numpy as np
from .helper import simulate_data
from bayesbridge.model import LinearModel, LogisticModel, CoxModel
from bayesbridge import BayesBridge, RegressionModel, RegressionCoefPrior


def test_clone():

    kwargs = {
        'bridge_exponent': 1. / 8,
        'n_fixed_effect': 1,
        'sd_for_fixed_effect': 1.11,
        'regularizing_slab_size': 2.22,
        'global_scale_prior_hyper_param': {'log10_mean': - 4., 'log10_sd': 1.}
    }

    prior = RegressionCoefPrior(**kwargs)

    changed_kw = {
        'n_fixed_effect': 3,
        'global_scale_prior_hyper_param': {'log10_mean': - 6., 'log10_sd': 1.5}
    }
    kwargs_alt = kwargs.copy()
    for key, val in changed_kw.items():
        kwargs_alt[key] = val
    cloned = prior.clone(**changed_kw)
    changed_prior = RegressionCoefPrior(**kwargs_alt)

    assert np.all(
        cloned.__dict__.pop('sd_for_fixed')
        == changed_prior.__dict__.pop('sd_for_fixed')
    )
    assert cloned.__dict__ == changed_prior.__dict__


def test_gscale_parametrization():
    """ Check that the Gamma hyper-parameters do not depend on parametrization. """

    gscale_hyper_param = {'log10_mean': - 4., 'log10_sd': 1.}
    bridge_exp = .25

    prior_coef_scale = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        global_scale_prior_hyper_param=gscale_hyper_param,
        _global_scale_parametrization='coef_magnitude'
    )

    unit_bridge_magnitude \
        = RegressionCoefPrior.compute_power_exp_ave_magnitude(bridge_exp)
    gscale_hyper_param['log10_mean'] -= np.log10(unit_bridge_magnitude)
    prior_raw_scale = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        global_scale_prior_hyper_param=gscale_hyper_param,
        _global_scale_parametrization='raw'
    )
    assert (
        prior_coef_scale.param['gscale_neg_power'] == prior_raw_scale.param['gscale_neg_power']
    )


def test_gscale_paramet_invariance():
    """ Check sampler outputs are invariant under global scale parametrization. """

    y, X, beta = simulate_data(model='logit', seed=0)
    model = RegressionModel(y, X, family='logit')
    bridge_exp = .25

    # Two samples should agree since the default prior is scale invariant.
    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        _global_scale_parametrization='raw'
    )
    bridge = BayesBridge(model, prior)
    coef_sample_raw_scaling = get_last_sample_from_gibbs(bridge, bridge_exp)

    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        _global_scale_parametrization='coef_magnitude'
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
        _global_scale_parametrization='raw'
    )
    bridge = BayesBridge(model, prior)
    coef_sample_raw_scaling \
        = get_last_sample_from_gibbs(bridge, bridge_exp)

    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        global_scale_prior_hyper_param=gscale_hyper_param,
        _global_scale_parametrization='coef_magnitude'
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
    prior = prior.clone(global_scale_prior_hyper_param=gscale_hyper_param)
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
        coef_sampler_type='cholesky',
        init={'apriori_coef_scale': .1},
        seed=seed, n_status_update=0
    )
    return mcmc_output['samples']['coef'][:, -1]