import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib.pyplot as plt
from bayesbridge import BayesBridge, RegressionModel, RegressionCoefPrior
from simulate_data import simulate_design, simulate_outcome
from util import mcmc_summarizer

n_obs, n_pred = 10 ** 4, 10 ** 3

X = simulate_design(
    n_obs, n_pred, 
    binary_frac=.9,
    binary_pred_freq=.2,
    shuffle_columns=True,
    format_='sparse',
    seed=111
)

beta_true = np.zeros(n_pred)
beta_true[:5] = 1.5
beta_true[5:10] = 1.
beta_true[10:15] = .5

n_trial = np.ones(X.shape[0]) # Binary outcome.
y = simulate_outcome(
    X, beta_true, intercept=0., 
    n_trial=n_trial, model='logit', seed=1
)

model = RegressionModel(
    y, X, family='logit',
    add_intercept=True, center_predictor=True,
        # Do *not* manually add intercept to or center X.
)

prior = RegressionCoefPrior(
    bridge_exponent=.5,
<<<<<<< Updated upstream
    n_fixed_effect=0, 
        # Number of coefficients with Gaussian priors of pre-specified sd.
    sd_for_intercept=float('inf'),
        # Set it to float('inf') for a flat prior.
    sd_for_fixed_effect=1.,
=======
    n_fixed_effect=1, 
        # Number of coefficients with Gaussian priors of pre-specified sd.
    sd_for_intercept=float('inf'),
        # Set it to float('inf') for a flat prior.
    sd_for_fixed_effect=0.001,
    mean_for_fixed_effect= 50., 
>>>>>>> Stashed changes
    regularizing_slab_size=2.,
        # Weakly constrain the magnitude of coefficients under bridge prior.
)

bridge = BayesBridge(model, prior)

<<<<<<< Updated upstream
samples, mcmc_info = bridge.gibbs(
    n_iter=250, n_burnin=0, thin=1, 
    init={'global_scale': .01},
    params_to_fix = ('global_scale'),
    coef_sampler_type='cg',
    seed=111
)

plt.figure(figsize=(10, 4))
plt.rcParams['font.size'] = 20

plt.plot(samples['logp'])
plt.xlabel('MCMC iteration')
plt.ylabel('Posterior log density')
=======
coef = np.zeros(n_pred)

samples, mcmc_info = bridge.gibbs(
    n_iter=250, n_burnin=0, thin=1, 
    init={'global_scale': .01,
    'coef' : np.zeros(n_pred + 1)},
    coef_sampler_type='cg',
    seed=111
)
coef_samples = samples['coef'][1:, :]

plt.figure(figsize=(12, 5))
plt.rcParams['font.size'] = 20

plt.plot(coef_samples[[1], :].T)
plt.xlabel('MCMC iteration')
plt.ylabel(r'$\beta_j$', rotation=0, labelpad=10)
plt.show()

plt.figure(figsize=(14, 5))
plt.rcParams['font.size'] = 20

n_coef_to_plot = 25

mcmc_summarizer.plot_conf_interval(
    coef_samples, conf_level=.95, 
    n_coef_to_plot=n_coef_to_plot, marker_scale=1.4
)
plt.plot(
    beta_true[:n_coef_to_plot], '--', color='tab:orange',
    label='True value'
)
plt.xlabel(r'Coefficient index $j$')
plt.ylabel(r'$\beta_j$', rotation=0, labelpad=10)
plt.xticks([0, 5, 10, 15, 20])
plt.legend(frameon=False)

>>>>>>> Stashed changes
plt.show()