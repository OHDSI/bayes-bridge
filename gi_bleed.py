import numpy as np
from bayesbridge import BayesBridge, RegressionModel, RegressionCoefPrior

import util.data_manager as data_manager

dm = data_manager.DataManager(
    path_to_data='../gi_bleed/',
    regress_on='treatment'
)

y, X = dm.read_ohdsi_data()
y = np.squeeze(y.toarray())

model = RegressionModel(
    y, X, family='logit',
    add_intercept=True, center_predictor=True,
)

prior = RegressionCoefPrior(
    bridge_exponent=.5,
    n_fixed_effect=0,
        # Number of coefficients with Gaussian priors of pre-specified sd.
    sd_for_intercept=float('inf'),
        # Set it to float('inf') for a flat prior.
    sd_for_fixed_effect=1.,
    regularizing_slab_size=2.,
        # Weakly constrain the magnitude of coefficients under bridge prior.
)

bridge = BayesBridge(model, prior)

mcmc_output = bridge.gibbs(
    n_iter=30, n_burnin=0, thin=1,
    init={'global_scale': .01},
    coef_sampler_type='cg',
    seed=111
)