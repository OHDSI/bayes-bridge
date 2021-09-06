import numpy as np
import math
import time
from .util import simplify_warnings # Monkey patch the warning format
from warnings import warn
from .random import BasicRandom
from .reg_coef_sampler import SparseRegressionCoefficientSampler
from .model import LogisticModel
from .prior import RegressionCoefPrior
from .gibbs_util import MarkovChainManager, SamplerOptions


class BayesBridge():
    """ Implement Gibbs sampler for Bayesian bridge sparse regression. """

    def __init__(self, model, prior=RegressionCoefPrior()):
        """
        Parameters
        ----------
        model : RegressionModel object
        prior : RegressionCoefPrior object
        """

        self.n_obs = model.n_obs
        self.n_pred = model.n_pred
        self.n_unshrunk = prior.n_fixed
        self.prior_sd_for_unshrunk = prior.sd_for_fixed.copy()
        if model.intercept_added:
            self.n_unshrunk += 1
            self.prior_sd_for_unshrunk = np.concatenate((
                [prior.sd_for_intercept], self.prior_sd_for_unshrunk
            ))

        self.model = model
        self.prior = prior
        self.rg = BasicRandom()
        self.manager = MarkovChainManager(
            self.n_obs, self.n_pred, self.n_unshrunk, model.name
        )

    # TODO: write a test to ensure that the output when resuming the Gibbs
    # sampler coincide with that without interruption.
    def gibbs_additional_iter(
            self, prev_mcmc_info, n_add_iter, n_status_update=0,
            merge=False, prev_samples=None):
        """ Resume Gibbs sampler from the last state.

        Parameter
        ---------
        prev_mcmc_info : dict
            MCMC info returned by a previous call to the `gibbs` method.
        n_iter : int
        n_status_update : int
        merge : bool
            If True, merge the Gibbs sampler outputs from the previous and
            new runs and then return.
        prev_samples : dict
            MCMC samples returned by a previous call to the 'gibbs' method.

        Returns
        -------
        new_samples : dict
        new_mcmc_info : dict
        """

        if merge and prev_samples is None:
            raise ValueError(
                "To merge the outputs from previous and new MCMC runs, you "
                "have to supply the optional argument `prev_samples`."
            )

        self.rg.set_state(prev_mcmc_info['_random_gen_state'])

        init = prev_mcmc_info['_markov_chain_state']
        thin, bridge_exp, coef_sampler_type = (
            prev_mcmc_info[key]
            for key in ['thin', 'bridge_exponent', 'coef_sampler_type']
        )
        curvature_est_stabilized = prev_mcmc_info['options']['hmc_curvature_est_stabilized']
        params_to_save = prev_mcmc_info['saved_params']

        # Initalize the regression coefficient sampler with the previous state.
        self.reg_coef_sampler = SparseRegressionCoefficientSampler(
            self.n_pred, self.prior_sd_for_unshrunk, coef_sampler_type,
            curvature_est_stabilized, self.prior.slab_size
        )
        self.reg_coef_sampler.set_internal_state(prev_mcmc_info['_reg_coef_sampler_state'])

        new_samples, new_mcmc_info = self.gibbs(
            n_add_iter, 0, thin, init=init,
            params_to_save=params_to_save,
            n_status_update=n_status_update,
            options=prev_mcmc_info['options'],
            _add_iter_mode=True
        )
        if merge:
            new_samples, new_mcmc_info = self.manager.merge_outputs(
                prev_samples, prev_mcmc_info, new_samples, new_mcmc_info
            )

        return new_samples, new_mcmc_info

    def gibbs(self, n_iter, n_burnin=0, thin=1, seed=None,
              init={'global_scale': 0.1}, params_to_save=('coef', 'global_scale', 'logp'),
              coef_sampler_type=None, n_status_update=0,
              options=None, _add_iter_mode=False):
        """ Generate posterior samples under the specified model and prior.

        Parameters
        ----------
        n_iter : int
            Total number of MCMC iterations i.e. burn-ins + saved posterior draws
        n_burnin : int
            Number of burn-in samples to be discarded
        thin : int
            Number of iterations per saved samples for "thinning" MCMC to reduce
            the output size. In other words, the function saves an MCMC sample
            every `thin` iterations, returning floor(n_iter / thin) samples.
        seed : int
            Seed for random number generator.
        init : dict of numpy arrays
            Specifies, partially or completely, the initial state of Markov chain.
            The partial option allows either specifying the global scale or
            regression coefficients. The former is the recommended default option
            since the global scale parameter is easier to choose, representing
            the prior expected magnitude of regression coefficients and . (But
            the latter might make sense if some preliminary estimate of coefficients
            are available.) Other parameters are then initialized through a
            combination of heuristics and conditional optimization.
        coef_sampler_type : {None, 'cholesky', 'cg', 'hmc'}
            Specifies the sampling method used to update regression coefficients.
            If None, the method is chosen via a crude heuristic based on the
            model type, as well as size and sparsity level of design matrix.
            For linear and logistic models with large and sparse design matrix,
            the conjugate gradient sampler ('cg') is preferred over the
            Cholesky decomposition based sampler ('cholesky'). For other
            models, only Hamiltonian Monte Carlo ('hmc') can be used.
        params_to_save : {'all', tuple or list of str}
            Specifies which parameters to save during MCMC iterations. By default,
            the most relevant parameters --- regression coefficients,
            global scale, posterior log-density --- are saved. Use all to save
            all the parameters (but beaware of the extra memory requirement),
            including local scale and, depending on the model, precision (
            inverse variance) of observations.
        n_status_update : int
            Number of updates to print on stdout during the sampler run.

        Other Parameters
        ----------------
        options : None, dict, SamplerOptions
            SamplerOptions class or a dict whose keywords are used as inputs
            to the class.

        Returns
        -------
        samples : dict
            Contains MCMC samples of the parameters as specified by
            **params_to_save**. The last dimension of the arrays correspond
            to MCMC iterations; for example,
            :code:`samples['coef'][:, 0]`
            is the first MCMC sample of regression coefficients.
        mcmc_info : dict
            Contains information on the MCMC run and sampler settings, enough in
            particular to reproduce and resume the sampling process.
        """

        if not isinstance(options, SamplerOptions):
            options = SamplerOptions.pick_default_and_create(
                coef_sampler_type, options, self.model.name, self.model.design
            )

        if not _add_iter_mode:
            self.rg.set_seed(seed)
            self.reg_coef_sampler = SparseRegressionCoefficientSampler(
                self.n_pred, self.prior_sd_for_unshrunk,
                options.coef_sampler_type, options.curvature_est_stabilized,
                self.prior.slab_size
            )

        if params_to_save == 'all':
            params_to_save = (
                'coef', 'local_scale', 'global_scale', 'logp'
            )
            if self.model.name != 'cox':
                params_to_save += ('obs_prec', )

        n_status_update = min(n_iter, n_status_update)
        start_time = time.time()
        self.manager.stamp_time(start_time)

        # Initial state of the Markov chain
        coef, obs_prec, lscale, gscale, init, initial_optim_info = \
            self.initialize_chain(init, self.prior.bridge_exp)

        # Pre-allocate
        samples = {}
        sampling_info = {}
        self.manager.pre_allocate(
            samples, sampling_info, n_iter - n_burnin, thin, params_to_save,
            options.coef_sampler_type
        )

        # Start Gibbs sampling
        for mcmc_iter in range(1, n_iter + 1):

            coef, info = self.update_regress_coef(
                coef, obs_prec, gscale, lscale, options.coef_sampler_type
            )

            obs_prec = self.update_obs_precision(coef)

            # Draw from gscale | coef and then lscale | gscale, coef.
            # (The order matters.)
            gscale = self.update_global_scale(
                gscale, coef[self.n_unshrunk:], self.prior.bridge_exp,
                method=options.gscale_update
            )

            lscale = self.update_local_scale(
                gscale, coef[self.n_unshrunk:], self.prior.bridge_exp)

            logp = self.compute_posterior_logprob(
                coef, gscale, obs_prec, self.prior.bridge_exp
            )

            self.manager.store_current_state(
                samples, mcmc_iter, n_burnin, thin, coef, lscale, gscale,
                obs_prec, logp, params_to_save
            )
            self.manager.store_sampling_info(
                sampling_info, info, mcmc_iter, n_burnin, thin,
                options.coef_sampler_type
            )
            self.manager.print_status(n_status_update, mcmc_iter, n_iter)

        runtime = time.time() - start_time

        if self.prior._gscale_paramet == 'coef_magnitude':
            gscale, lscale = \
                self.prior.adjust_scale(gscale, lscale, to='coef_magnitude')
            gscale_samples = samples.get('global_scale', 0.)
            lscale_samples = samples.get('local_scale', 0.)
            self.prior.adjust_scale(
                gscale_samples, lscale_samples, to='coef_magnitude'
            ) # Modify in place.

        _markov_chain_state = \
            self.manager.pack_parameters(coef, obs_prec, lscale, gscale)

        _reg_coef_sampling_info = None
        mcmc_info = {
            'init': init,
            'n_iter': n_iter,
            'n_burnin': n_burnin,
            'thin': thin,
            'seed': seed,
            'n_coef_wo_shrinkage': self.n_unshrunk,
            'prior_sd_for_unshrunk': self.prior_sd_for_unshrunk,
            'bridge_exponent': self.prior.bridge_exp,
            'coef_sampler_type': options.coef_sampler_type,
            'saved_params': params_to_save,
            'runtime': runtime,
            'options': options.get_info(),
            '_init_optim_info': initial_optim_info,
            '_reg_coef_sampling_info': sampling_info,
            '_markov_chain_state': _markov_chain_state,
            '_random_gen_state': self.rg.get_state(),
            '_reg_coef_sampler_state': self.reg_coef_sampler.get_internal_state()
        }

        return samples, mcmc_info

    def initialize_chain(self, init, bridge_exp):
        """ Choose the user-specified state if provided, the default ones otherwise."""

        valid_param_name \
            = ('coef', 'local_scale', 'global_scale', 'obs_prec', 'logp')
        for key in init:
            if key not in valid_param_name:
                warn("'{:s}' is not a valid parameter name and "
                     "will be ignored.".format(key))

        coef_only_specified = 'coef' in init and ('global_scale' not in init)

        if 'coef' in init:
            coef = init['coef']
            if not len(coef) == self.n_pred:
                raise ValueError('Invalid initial length of regression coefficient.')
        else:
            coef = np.zeros(self.n_pred)
            if self.model.name in ('linear', 'logit'):
                coef[0] = self.model.calc_intercept_mle()

        obs_prec = self.initialize_obs_precision(init, coef)

        if coef_only_specified:
            gscale = self.update_global_scale(
                None, coef[self.n_unshrunk:], bridge_exp,
                method='optimize'
            )
            lscale = self.update_local_scale(
                gscale, coef[self.n_unshrunk:], bridge_exp
            )
        else:
            if 'global_scale' not in init:
                raise ValueError("Initial global scale must be specified when "
                                 "coefficients aren't specified.")
            if self.prior._gscale_paramet == 'raw':
                warn("Using the raw global scale parametrization; make sure that "
                     "the specified initial value is scaled accordingly.")
            gscale = init['global_scale']
            if 'local_scale' in init:
                lscale = init['local_scale']
                if not len(lscale) == (self.n_pred - self.n_unshrunk):
                    raise ValueError('Invalid initial length of local scale parameter')
            else:
                lscale = np.ones(self.n_pred - self.n_unshrunk) / gscale

        if self.prior._gscale_paramet == 'coef_magnitude':
            # Gibbs sampler requires the raw parametrization, though
            # technically only gscale * lscale matters due to the update order.
            gscale, lscale \
                = self.prior.adjust_scale(gscale, lscale, to='raw')

        if 'coef' not in init:
            # Optimize coefficients and then update other parameters
            coef, info = self.reg_coef_sampler.search_mode(
                coef, lscale, gscale, obs_prec, self.model
            )
            obs_prec = self.update_obs_precision(coef)
            lscale = self.update_local_scale(
                gscale, coef[self.n_unshrunk:], bridge_exp
            )
            optim_info = {
                key: info[key] for key in ['is_success', 'n_design_matvec', 'n_iter']
            }
        else:
            optim_info = None

        init = {
            'coef': coef,
            'obs_prec': obs_prec,
            'local_scale': lscale,
            'global_scale': gscale
        }

        return coef, obs_prec, lscale, gscale, init, optim_info

    def initialize_obs_precision(self, init, coef):
        if 'obs_prec' in init:
            obs_prec = np.ascontiguousarray(init['obs_prec'])
            # Cython requires a C-contiguous array.
            if not len(obs_prec) == self.n_obs:
                raise ValueError('An invalid initial state.')
        elif self.model.name == 'linear':
            obs_prec = np.mean(
                (self.model.y - self.model.design.dot(coef)) ** 2) ** -1
        elif self.model.name == 'logit':
            obs_prec = LogisticModel.compute_polya_gamma_mean(
                self.model.n_trial, self.model.design.dot(coef)
            )
        else:
            obs_prec = None
        return obs_prec

    def update_regress_coef(self, coef, obs_prec, gscale, lscale, sampling_method):

        if sampling_method in ('cholesky', 'cg'):

            if self.model.name == 'linear':
                y_gaussian = self.model.y
                obs_prec = obs_prec * np.ones(self.n_obs)
            elif self.model.name == 'logit':
                y_gaussian = (self.model.n_success - self.model.n_trial / 2) / obs_prec

            coef, info = self.reg_coef_sampler.sample_gaussian_posterior(
                y_gaussian, self.model.design, obs_prec, gscale, lscale,
                sampling_method
            )

        elif sampling_method in ['hmc', 'nuts']:
            coef, info = self.reg_coef_sampler.sample_by_hmc(
                coef, gscale, lscale, self.model, method=sampling_method
            )

        else:
            raise NotImplementedError()

        return coef, info

    def update_obs_precision(self, coef):

        obs_prec = None
        if self.model.name == 'linear':
            resid = self.model.y - self.model.design.dot(coef)
            scale = np.sum(resid ** 2) / 2
            obs_var = scale / self.rg.np_random.gamma(self.n_obs / 2, 1)
            obs_prec = 1 / obs_var
        elif self.model.name == 'logit':
            obs_prec = self.rg.polya_gamma(
                self.model.n_trial.astype(np.intc), self.model.design.dot(coef)
            )

        return obs_prec

    def update_global_scale(
            self, gscale, beta_with_shrinkage, bridge_exp,
            coef_expected_magnitude_lower_bd=.001, method='sample'):
        # :param method: {"sample", "optimize", None}

        if beta_with_shrinkage.size == 0:
            return 1. # arbitrary float value as a placeholder

        lower_bd = coef_expected_magnitude_lower_bd \
                   / self.prior.compute_power_exp_ave_magnitude(bridge_exp)
            # Solve for the value of global shrinkage such that
            # (expected value of regress_coef given gscale) = coef_expected_magnitude_lower_bd.

        if method == 'optimize':
            gscale = self.monte_carlo_em_global_scale(
                beta_with_shrinkage, bridge_exp)

        elif method == 'sample':
            # Conjugate update for phi = 1 / gscale ** bridge_exp
            if np.count_nonzero(beta_with_shrinkage) == 0:
                gscale = 0
            else:
                prior_param = self.prior.param['gscale_neg_power']
                shape, rate = prior_param['shape'], prior_param['rate']
                shape += beta_with_shrinkage.size / bridge_exp
                rate += np.sum(np.abs(beta_with_shrinkage) ** bridge_exp)
                phi = self.rg.np_random.gamma(shape, scale=1 / rate)
                gscale = 1 / phi ** (1 / bridge_exp)

        if (method is not None) and gscale < lower_bd:
            gscale = lower_bd
            warn(
                "The global shrinkage parameter update returned an unreasonably "
                "small value. Returning a specified lower bound value instead."
            )

        return gscale

    def monte_carlo_em_global_scale(
            self, beta_with_shrinkage, bridge_exp):
        """ Maximize the likelihood (not posterior conditional) 'coef | gscale'. """
        phi = len(beta_with_shrinkage) / bridge_exp \
              / np.sum(np.abs(beta_with_shrinkage) ** bridge_exp)
        gscale = phi ** - (1 / bridge_exp)
        return gscale

    def update_local_scale(self, gscale, beta_with_shrinkage, bridge_exp):

        lscale_sq = .5 / self.rg.tilted_stable(
            bridge_exp / 2, (beta_with_shrinkage / gscale) ** 2
        )
        lscale = np.sqrt(lscale_sq)

        # TODO: Pick the lower and upper bound more carefully.
        if np.any(lscale == 0):
            warn(
                "Local scale parameter under-flowed. Replacing with a small number.")
            lscale[lscale == 0] = 10e-16
        elif np.any(np.isinf(lscale)):
            warn(
                "Local scale parameter over-flowed. Replacing with a large number.")
            lscale[np.isinf(lscale)] = 2.0 / gscale

        return lscale

    def compute_posterior_logprob(self, coef, gscale, obs_prec, bridge_exp):

        # Contributions from the likelihood.
        params = [coef] if self.model.name != 'linear' else [coef, obs_prec]
        loglik, _ = self.model.compute_loglik_and_gradient(*params, loglik_only=True)

        # Contributions from the regularization.
        loglik += - .5 * np.sum((coef / self.prior.slab_size) ** 2)

        # Add contributions from the priors.
        prior_logp = 0
        n_shrunk_coef = len(coef) - self.n_unshrunk

        # for coef | gscale.
        prior_logp += \
            - n_shrunk_coef * math.log(gscale) \
            - np.sum(np.abs(coef[self.n_unshrunk:] / gscale) ** bridge_exp)

        # for coefficients without shrinkage.
        prior_logp += - 1 / 2 * np.sum(
            (coef[:self.n_unshrunk] / self.prior_sd_for_unshrunk) ** 2
        )
        prior_logp += - np.sum(np.log(
            self.prior_sd_for_unshrunk[self.prior_sd_for_unshrunk < float('inf')]
        ))
        prior_param = self.prior.param['gscale_neg_power']
        prior_logp += (prior_param['shape'] - 1.) * math.log(gscale) \
                      - prior_param['rate'] * gscale

        logp = loglik + prior_logp

        return logp
