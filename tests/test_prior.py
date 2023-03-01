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
    bridge_magnitude \
        = RegressionCoefPrior.compute_power_exp_ave_magnitude(bridge_exp)
    init_gscale = 0.1
    init_lscale = np.ones(X.shape[1])
    init_raw_gscale = init_gscale / bridge_magnitude
    init_raw_lscale = bridge_magnitude * init_lscale
    init = {
        'global_scale': init_gscale,
        'local_scale': init_lscale
    }
    raw_init = {
        'global_scale': init_raw_gscale,
        'local_scale': init_raw_lscale
    }

    # Two samples should agree since the default prior is scale invariant.
    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        _global_scale_parametrization='raw'
    )
    bridge = BayesBridge(model, prior)
    coef_sample_raw_scaling = get_last_sample_from_gibbs(bridge, raw_init)

    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        _global_scale_parametrization='coef_magnitude'
    )
    bridge = BayesBridge(model, prior)
    coef_sample_expected_mag_scaling = get_last_sample_from_gibbs(bridge, init)

    assert np.allclose(
        coef_sample_raw_scaling,
        coef_sample_expected_mag_scaling,
        rtol=1e-10
    )

    # Place a prior on the global scale; the two samples should *not* coincide.

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
        = get_last_sample_from_gibbs(bridge, raw_init)

    prior = RegressionCoefPrior(
        bridge_exponent=bridge_exp,
        regularizing_slab_size=1.,
        global_scale_prior_hyper_param=gscale_hyper_param,
        _global_scale_parametrization='coef_magnitude'
    )
    bridge = BayesBridge(model, prior)
    coef_sample_expected_mag_scaling \
        = get_last_sample_from_gibbs(bridge, init)

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
        = get_last_sample_from_gibbs(bridge, init)

    assert np.allclose(
        coef_sample_raw_scaling,
        coef_sample_expected_mag_scaling,
        rtol=1e-10
    )


def get_last_sample_from_gibbs(bridge, init, seed=0):
    samples, _ = bridge.gibbs(
        n_iter=10, n_burnin=0, init=init,
        coef_sampler_type='cholesky',
        seed=seed, n_status_update=0
    )
    return samples['coef'][:, -1]